#include "Gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "Parallel.h"
#include "JacobiRow.h"

/*
 * CUDA 实现。
 *
 * 为保证与 CPU 结果逐位一致：
 *  - 每行按 CSR 中的列顺序串行累加（一个线程负责一行），与 CPU 循环顺序相同；
 *  - 单行计算与主机模拟后端共用 JacobiRow.h；
 *  - 编译选项 -fmad=false 禁止把 a*b+c 融合为 FMA（CPU 端对应 -ffp-contract=off）；
 *  - 除法、开方使用默认的 IEEE 精确舍入（不要使用 --use_fast_math）；
 *  - 最大值归约与顺序无关，结果与 CPU 相同。
 */

namespace gpu
{
    namespace
    {
        void check(cudaError_t err, const char* what)
        {
            if (err != cudaSuccess)
            {
                std::ostringstream msg;
                msg << "CUDA error in " << what << ": " << cudaGetErrorString(err);
                std::cerr << "[rank " << par::rank() << "] " << msg.str() << std::endl;
                throw std::runtime_error(msg.str());
            }
        }

        constexpr int BLOCK = 256;

        struct DeviceState
        {
            bool probed = false;
            bool enabled = false;
            int device = -1;
            std::string info = "GPU disabled";
        };

        DeviceState& state()
        {
            static DeviceState s;
            return s;
        }

        void probe()
        {
            DeviceState& s = state();
            if (s.probed)
            {
                return;
            }
            s.probed = true;

            if (const char* env = std::getenv("CFD_USE_GPU"))
            {
                if (std::string(env) == "0")
                {
                    s.info = "GPU disabled by CFD_USE_GPU=0";
                    return;
                }
            }

            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            {
                cudaGetLastError(); // 清除错误状态
                s.info = "no CUDA device found, using CPU";
                return;
            }

            s.device = par::nodeLocalRank() % count;
            check(cudaSetDevice(s.device), "cudaSetDevice");
            cudaDeviceProp prop{};
            check(cudaGetDeviceProperties(&prop, s.device), "cudaGetDeviceProperties");
            std::ostringstream info;
            info << "rank " << par::rank() << " -> GPU " << s.device << " (" << prop.name
                << ", sm_" << prop.major << prop.minor << ")";
            s.info = info.str();
            s.enabled = true;
            // 写到 stderr，便于确认正在使用 GPU（不影响 stdout 日志与输出文件）
            std::cerr << "[GPU] " << s.info << std::endl;
        }

        /* ---------------- 内核 ---------------- */

        // Jacobi 更新：一个线程一行
        template<int NC, bool SPLIT>
        __global__ void jacobiUpdate(int nRows,
                                     const int* __restrict__ rowPtr,
                                     const int* __restrict__ colIdx,
                                     const double* __restrict__ values,
                                     const double* __restrict__ diag,
                                     const double* __restrict__ b,
                                     const double* __restrict__ x0,
                                     double* __restrict__ x)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < nRows)
            {
                jacobiUpdateRow<NC, SPLIT>(i, rowPtr, colIdx, values, diag, b, x0, x);
            }
        }

        // 残差并在块内求最大值。rows == nullptr 时为全部自有行
        template<int NC, bool SPLIT>
        __global__ void residualMax(int n,
                                    const int* __restrict__ rows,
                                    const int* __restrict__ rowPtr,
                                    const int* __restrict__ colIdx,
                                    const double* __restrict__ values,
                                    const double* __restrict__ b,
                                    const double* __restrict__ x,
                                    double* __restrict__ blockMax)
        {
            __shared__ double shared[BLOCK];
            const int k = blockIdx.x * blockDim.x + threadIdx.x;
            double r = 0.0;
            if (k < n)
            {
                r = jacobiResidualRow<NC, SPLIT>(rows ? rows[k] : k, rowPtr, colIdx, values, b, x);
            }
            shared[threadIdx.x] = r;
            __syncthreads();
            for (int stride = BLOCK / 2; stride > 0; stride >>= 1)
            {
                if (threadIdx.x < stride)
                {
                    shared[threadIdx.x] = maxLike(shared[threadIdx.x], shared[threadIdx.x + stride]);
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                blockMax[blockIdx.x] = shared[0];
            }
        }

        // 打包：packed[k] = x[indexes[k]]
        __global__ void gatherKernel(int n, int nc, const int* __restrict__ indexes,
                                     const double* __restrict__ x, double* __restrict__ packed)
        {
            const int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k < n)
            {
                for (int c = 0; c < nc; ++c)
                {
                    packed[k * nc + c] = x[indexes[k] * nc + c];
                }
            }
        }

        // 解包：x[indexes[k]] = packed[k]
        __global__ void scatterKernel(int n, int nc, const int* __restrict__ indexes,
                                      const double* __restrict__ packed, double* __restrict__ x)
        {
            const int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k < n)
            {
                for (int c = 0; c < nc; ++c)
                {
                    x[indexes[k] * nc + c] = packed[k * nc + c];
                }
            }
        }

        int blocksFor(std::size_t n)
        {
            return static_cast<int>((n + BLOCK - 1) / BLOCK);
        }

        /* ---------------- 设备缓冲区 ---------------- */

        template<typename T>
        class DeviceBuffer
        {
        public:
            DeviceBuffer() = default;
            ~DeviceBuffer() { release(); }
            DeviceBuffer(const DeviceBuffer&) = delete;
            DeviceBuffer& operator=(const DeviceBuffer&) = delete;

            void resize(std::size_t n)
            {
                if (n <= capacity_)
                {
                    size_ = n;
                    return;
                }
                release();
                check(cudaMalloc(&ptr_, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
                capacity_ = n;
                size_ = n;
            }

            void upload(const T* host, std::size_t n, cudaStream_t stream)
            {
                resize(n);
                if (n > 0)
                {
                    check(cudaMemcpyAsync(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice, stream),
                          "cudaMemcpyAsync(H2D)");
                }
            }

            void swap(DeviceBuffer& other) noexcept
            {
                std::swap(ptr_, other.ptr_);
                std::swap(capacity_, other.capacity_);
                std::swap(size_, other.size_);
            }

            T* get() { return ptr_; }
            const T* get() const { return ptr_; }
            std::size_t size() const { return size_; }

        private:
            void release()
            {
                if (ptr_ != nullptr)
                {
                    cudaFree(ptr_);
                    ptr_ = nullptr;
                }
                capacity_ = 0;
                size_ = 0;
            }

            T* ptr_ = nullptr;
            std::size_t capacity_ = 0;
            std::size_t size_ = 0;
        };

        std::vector<int> toInt(const std::vector<ULL>& v, const char* what)
        {
            std::vector<int> out(v.size());
            for (std::size_t k = 0; k < v.size(); ++k)
            {
                if (v[k] > static_cast<ULL>(std::numeric_limits<int>::max()))
                {
                    throw std::overflow_error(std::string("gpu: index too large in ") + what);
                }
                out[k] = static_cast<int>(v[k]);
            }
            return out;
        }
    } // namespace

    bool enabled()
    {
        probe();
        return state().enabled;
    }

    std::string deviceInfo()
    {
        probe();
        return state().info;
    }

    /* ======================== JacobiGpu ======================== */

    struct JacobiGpu::Impl
    {
        int nc = 1;
        int nRows = 0;
        int nCols = 0;
        bool split = false;
        cudaStream_t stream = nullptr;

        DeviceBuffer<int> rowPtr, colIdx;
        DeviceBuffer<double> values, diag, b, x0, x;
        DeviceBuffer<int> sendIdx, recvIdx, interiorRows, haloRows;
        DeviceBuffer<double> sendBuf, recvBuf, blockMax;

        std::vector<double> hostBlockMax;
        int pendingBlocks = 0;
    };

    JacobiGpu::JacobiGpu(int nComp)
        : impl_(std::make_unique<Impl>())
    {
        if (nComp != 1 && nComp != 3)
        {
            throw std::invalid_argument("JacobiGpu: only scalar (1) and vector (3) unknowns are supported");
        }
        if (!enabled())
        {
            throw std::runtime_error("JacobiGpu: no GPU available");
        }
        impl_->nc = nComp;
        check(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
    }

    JacobiGpu::~JacobiGpu()
    {
        if (impl_ && impl_->stream != nullptr)
        {
            cudaStreamSynchronize(impl_->stream);
            cudaStreamDestroy(impl_->stream);
        }
    }

    void JacobiGpu::upload(ULL nRows, ULL nCols,
                           const std::vector<ULL>& rowPointer,
                           const std::vector<ULL>& columnIndex,
                           const std::vector<double>& values,
                           const std::vector<double>& diag,
                           const double* b,
                           bool split)
    {
        Impl& m = *impl_;
        if (nCols > static_cast<ULL>(std::numeric_limits<int>::max() / m.nc))
        {
            throw std::overflow_error("JacobiGpu: matrix too large for 32-bit indices");
        }
        m.nRows = static_cast<int>(nRows);
        m.nCols = static_cast<int>(nCols);
        m.split = split;

        // 整型化后的索引需要在拷贝完成前保持有效，因此同步上传
        const std::vector<int> rp = toInt(rowPointer, "rowPointer");
        const std::vector<int> ci = toInt(columnIndex, "columnIndex");
        m.rowPtr.upload(rp.data(), rp.size(), m.stream);
        m.colIdx.upload(ci.data(), ci.size(), m.stream);
        m.values.upload(values.data(), values.size(), m.stream);
        m.diag.upload(diag.data(), diag.size(), m.stream);
        m.b.upload(b, nRows * m.nc, m.stream);
        m.x0.resize(nCols * m.nc);
        m.x.resize(nCols * m.nc);
        check(cudaStreamSynchronize(m.stream), "upload");
    }

    void JacobiGpu::setHalo(const std::vector<ULL>& sendIndexes, const std::vector<ULL>& recvIndexes)
    {
        Impl& m = *impl_;
        const std::vector<int> s = toInt(sendIndexes, "sendIndexes");
        const std::vector<int> r = toInt(recvIndexes, "recvIndexes");
        m.sendIdx.upload(s.data(), s.size(), m.stream);
        m.recvIdx.upload(r.data(), r.size(), m.stream);
        m.sendBuf.resize(s.size() * m.nc);
        m.recvBuf.resize(r.size() * m.nc);
        check(cudaStreamSynchronize(m.stream), "setHalo");
    }

    void JacobiGpu::setRowSets(const std::vector<ULL>& interiorRows, const std::vector<ULL>& haloRows)
    {
        Impl& m = *impl_;
        const std::vector<int> in = toInt(interiorRows, "interiorRows");
        const std::vector<int> ha = toInt(haloRows, "haloRows");
        m.interiorRows.upload(in.data(), in.size(), m.stream);
        m.haloRows.upload(ha.data(), ha.size(), m.stream);
        check(cudaStreamSynchronize(m.stream), "setRowSets");
    }

    void JacobiGpu::setX0(const double* x0)
    {
        Impl& m = *impl_;
        m.x0.upload(x0, static_cast<std::size_t>(m.nCols) * m.nc, m.stream);
        // x 的幽灵单元在第一次交换前也需要有定义的值
        check(cudaMemcpyAsync(m.x.get(), m.x0.get(), static_cast<std::size_t>(m.nCols) * m.nc * sizeof(double),
                              cudaMemcpyDeviceToDevice, m.stream), "setX0");
        check(cudaStreamSynchronize(m.stream), "setX0");
    }

    void JacobiGpu::update()
    {
        Impl& m = *impl_;
        if (m.nRows == 0)
        {
            return;
        }
        const int grid = blocksFor(m.nRows);
#define CFD_LAUNCH_UPDATE(NC, SPLIT) \
        jacobiUpdate<NC, SPLIT><<<grid, BLOCK, 0, m.stream>>>(m.nRows, m.rowPtr.get(), m.colIdx.get(), \
            m.values.get(), m.diag.get(), m.b.get(), m.x0.get(), m.x.get())
        if (m.nc == 1)
        {
            if (m.split) { CFD_LAUNCH_UPDATE(1, true); } else { CFD_LAUNCH_UPDATE(1, false); }
        }
        else
        {
            if (m.split) { CFD_LAUNCH_UPDATE(3, true); } else { CFD_LAUNCH_UPDATE(3, false); }
        }
#undef CFD_LAUNCH_UPDATE
        check(cudaGetLastError(), "jacobiUpdate");
    }

    void JacobiGpu::gatherSend(std::vector<double>& packed)
    {
        Impl& m = *impl_;
        const std::size_t n = m.sendIdx.size();
        packed.resize(n * m.nc);
        if (n == 0)
        {
            return;
        }
        gatherKernel<<<blocksFor(n), BLOCK, 0, m.stream>>>(static_cast<int>(n), m.nc, m.sendIdx.get(),
                                                         m.x.get(), m.sendBuf.get());
        check(cudaGetLastError(), "gatherKernel");
        check(cudaMemcpyAsync(packed.data(), m.sendBuf.get(), packed.size() * sizeof(double),
                              cudaMemcpyDeviceToHost, m.stream), "gatherSend");
        check(cudaStreamSynchronize(m.stream), "gatherSend");
    }

    void JacobiGpu::scatterRecv(const std::vector<double>& packed)
    {
        Impl& m = *impl_;
        const std::size_t n = m.recvIdx.size();
        if (n == 0)
        {
            return;
        }
        if (packed.size() != n * m.nc)
        {
            throw std::invalid_argument("JacobiGpu::scatterRecv: wrong buffer size");
        }
        m.recvBuf.upload(packed.data(), packed.size(), m.stream);
        scatterKernel<<<blocksFor(n), BLOCK, 0, m.stream>>>(static_cast<int>(n), m.nc, m.recvIdx.get(),
                                                          m.recvBuf.get(), m.x.get());
        check(cudaGetLastError(), "scatterKernel");
        // packed 由调用方持有，拷贝需在返回前完成
        check(cudaStreamSynchronize(m.stream), "scatterRecv");
    }

    void JacobiGpu::launchResidual(int rowSet)
    {
        Impl& m = *impl_;
        const int* rows = nullptr;
        std::size_t n = static_cast<std::size_t>(m.nRows);
        if (rowSet == 1)
        {
            rows = m.interiorRows.get();
            n = m.interiorRows.size();
        }
        else if (rowSet == 2)
        {
            rows = m.haloRows.get();
            n = m.haloRows.size();
        }
        m.pendingBlocks = blocksFor(n);
        if (n == 0)
        {
            return;
        }
        m.blockMax.resize(m.pendingBlocks);
        m.hostBlockMax.resize(m.pendingBlocks);
#define CFD_LAUNCH_RESIDUAL(NC, SPLIT) \
        residualMax<NC, SPLIT><<<m.pendingBlocks, BLOCK, 0, m.stream>>>(static_cast<int>(n), rows, \
            m.rowPtr.get(), m.colIdx.get(), m.values.get(), m.b.get(), m.x.get(), m.blockMax.get())
        if (m.nc == 1)
        {
            if (m.split) { CFD_LAUNCH_RESIDUAL(1, true); } else { CFD_LAUNCH_RESIDUAL(1, false); }
        }
        else
        {
            if (m.split) { CFD_LAUNCH_RESIDUAL(3, true); } else { CFD_LAUNCH_RESIDUAL(3, false); }
        }
#undef CFD_LAUNCH_RESIDUAL
        check(cudaGetLastError(), "residualMax");
        check(cudaMemcpyAsync(m.hostBlockMax.data(), m.blockMax.get(), m.pendingBlocks * sizeof(double),
                              cudaMemcpyDeviceToHost, m.stream), "residual D2H");
    }

    double JacobiGpu::finishResidual()
    {
        Impl& m = *impl_;
        if (m.pendingBlocks == 0)
        {
            return 0.0;
        }
        check(cudaStreamSynchronize(m.stream), "finishResidual");
        double result = 0.0;
        for (int k = 0; k < m.pendingBlocks; ++k)
        {
            result = maxLike(result, m.hostBlockMax[k]);
        }
        m.pendingBlocks = 0;
        return result;
    }

    void JacobiGpu::swap()
    {
        Impl& m = *impl_;
        m.x0.swap(m.x);
    }

    void JacobiGpu::download(double* x)
    {
        Impl& m = *impl_;
        check(cudaMemcpyAsync(x, m.x.get(), static_cast<std::size_t>(m.nCols) * m.nc * sizeof(double),
                              cudaMemcpyDeviceToHost, m.stream), "download");
        check(cudaStreamSynchronize(m.stream), "download");
    }

} // namespace gpu
