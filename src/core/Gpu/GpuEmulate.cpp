#include "Gpu.h"

#include <limits>
#include <stdexcept>

#include "JacobiRow.h"

/*
 * GPU 后端的主机端模拟实现（仅用于测试，CMake 选项 CFD_GPU_EMULATE=ON）。
 *
 * 与 Gpu.cu 使用同一套接口和同一份单行计算代码（JacobiRow.h），
 * 只是把"显存"换成主机内存、把内核换成循环。这样在没有 GPU 的机器上
 * 也能验证 GPU 路径的数据流（上传、halo 打包/解包、残差分组、交换、下载）
 * 与 CPU 实现逐位一致。
 */

namespace gpu
{
    bool enabled()
    {
        return true;
    }

    std::string deviceInfo()
    {
        return "GPU emulation on host (CFD_GPU_EMULATE)";
    }

    namespace
    {
        std::vector<int> toInt(const std::vector<ULL>& v)
        {
            std::vector<int> out(v.size());
            for (std::size_t k = 0; k < v.size(); ++k)
            {
                if (v[k] > static_cast<ULL>(std::numeric_limits<int>::max()))
                {
                    throw std::overflow_error("gpu: index too large");
                }
                out[k] = static_cast<int>(v[k]);
            }
            return out;
        }
    }

    struct JacobiGpu::Impl
    {
        int nc = 1;
        int nRows = 0;
        int nCols = 0;
        bool split = false;
        std::vector<int> rowPtr, colIdx, sendIdx, recvIdx, interiorRows, haloRows;
        std::vector<double> values, diag, b, x0, x;
        double pending = 0.0;
    };

    JacobiGpu::JacobiGpu(int nComp)
        : impl_(std::make_unique<Impl>())
    {
        if (nComp != 1 && nComp != 3)
        {
            throw std::invalid_argument("JacobiGpu: only scalar (1) and vector (3) unknowns are supported");
        }
        impl_->nc = nComp;
    }

    JacobiGpu::~JacobiGpu() = default;

    void JacobiGpu::upload(ULL nRows, ULL nCols,
                           const std::vector<ULL>& rowPointer,
                           const std::vector<ULL>& columnIndex,
                           const std::vector<double>& values,
                           const std::vector<double>& diag,
                           const double* b,
                           bool split)
    {
        Impl& m = *impl_;
        m.nRows = static_cast<int>(nRows);
        m.nCols = static_cast<int>(nCols);
        m.split = split;
        m.rowPtr = toInt(rowPointer);
        m.colIdx = toInt(columnIndex);
        m.values = values;
        m.diag = diag;
        m.b.assign(b, b + nRows * m.nc);
        m.x0.assign(nCols * m.nc, 0.0);
        m.x.assign(nCols * m.nc, 0.0);
    }

    void JacobiGpu::setHalo(const std::vector<ULL>& sendIndexes, const std::vector<ULL>& recvIndexes)
    {
        impl_->sendIdx = toInt(sendIndexes);
        impl_->recvIdx = toInt(recvIndexes);
    }

    void JacobiGpu::setRowSets(const std::vector<ULL>& interiorRows, const std::vector<ULL>& haloRows)
    {
        impl_->interiorRows = toInt(interiorRows);
        impl_->haloRows = toInt(haloRows);
    }

    void JacobiGpu::setX0(const double* x0)
    {
        Impl& m = *impl_;
        m.x0.assign(x0, x0 + static_cast<std::size_t>(m.nCols) * m.nc);
        m.x = m.x0;
    }

    void JacobiGpu::update()
    {
        Impl& m = *impl_;
        for (int i = 0; i < m.nRows; ++i)
        {
            if (m.nc == 1)
            {
                if (m.split) jacobiUpdateRow<1, true>(i, m.rowPtr.data(), m.colIdx.data(), m.values.data(), m.diag.data(), m.b.data(), m.x0.data(), m.x.data());
                else jacobiUpdateRow<1, false>(i, m.rowPtr.data(), m.colIdx.data(), m.values.data(), m.diag.data(), m.b.data(), m.x0.data(), m.x.data());
            }
            else
            {
                if (m.split) jacobiUpdateRow<3, true>(i, m.rowPtr.data(), m.colIdx.data(), m.values.data(), m.diag.data(), m.b.data(), m.x0.data(), m.x.data());
                else jacobiUpdateRow<3, false>(i, m.rowPtr.data(), m.colIdx.data(), m.values.data(), m.diag.data(), m.b.data(), m.x0.data(), m.x.data());
            }
        }
    }

    void JacobiGpu::gatherSend(std::vector<double>& packed)
    {
        Impl& m = *impl_;
        packed.resize(m.sendIdx.size() * m.nc);
        for (std::size_t k = 0; k < m.sendIdx.size(); ++k)
        {
            for (int c = 0; c < m.nc; ++c)
            {
                packed[k * m.nc + c] = m.x[m.sendIdx[k] * m.nc + c];
            }
        }
    }

    void JacobiGpu::scatterRecv(const std::vector<double>& packed)
    {
        Impl& m = *impl_;
        if (packed.size() != m.recvIdx.size() * m.nc)
        {
            throw std::invalid_argument("JacobiGpu::scatterRecv: wrong buffer size");
        }
        for (std::size_t k = 0; k < m.recvIdx.size(); ++k)
        {
            for (int c = 0; c < m.nc; ++c)
            {
                m.x[m.recvIdx[k] * m.nc + c] = packed[k * m.nc + c];
            }
        }
    }

    void JacobiGpu::launchResidual(int rowSet)
    {
        Impl& m = *impl_;
        const std::vector<int>* rows = rowSet == 1 ? &m.interiorRows : rowSet == 2 ? &m.haloRows : nullptr;
        const int n = rows ? static_cast<int>(rows->size()) : m.nRows;
        double result = 0.0;
        for (int k = 0; k < n; ++k)
        {
            const int i = rows ? (*rows)[k] : k;
            double r = 0.0;
            if (m.nc == 1)
            {
                r = m.split ? jacobiResidualRow<1, true>(i, m.rowPtr.data(), m.colIdx.data(), m.values.data(), m.b.data(), m.x.data())
                            : jacobiResidualRow<1, false>(i, m.rowPtr.data(), m.colIdx.data(), m.values.data(), m.b.data(), m.x.data());
            }
            else
            {
                r = m.split ? jacobiResidualRow<3, true>(i, m.rowPtr.data(), m.colIdx.data(), m.values.data(), m.b.data(), m.x.data())
                            : jacobiResidualRow<3, false>(i, m.rowPtr.data(), m.colIdx.data(), m.values.data(), m.b.data(), m.x.data());
            }
            result = maxLike(result, r);
        }
        m.pending = result;
    }

    double JacobiGpu::finishResidual()
    {
        return impl_->pending;
    }

    void JacobiGpu::swap()
    {
        impl_->x0.swap(impl_->x);
    }

    void JacobiGpu::download(double* x)
    {
        Impl& m = *impl_;
        std::copy(m.x.begin(), m.x.end(), x);
    }

} // namespace gpu
