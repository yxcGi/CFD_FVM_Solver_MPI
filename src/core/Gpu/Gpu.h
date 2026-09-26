#ifndef GPU_H_
#define GPU_H_

/*
 * GPU 加速层
 *
 * 求解器其余部分只通过这里访问 GPU，本头文件不包含任何 CUDA 头文件：
 *
 *   gpu::enabled()      是否使用 GPU（编译时启用 CUDA、运行时检测到设备、且未被环境变量关闭）
 *   gpu::deviceInfo()   当前进程使用的设备描述
 *   gpu::JacobiGpu      在 GPU 上执行 Jacobi 迭代（CSR SpMV + 更新 + 残差）
 *
 * 编译时未启用 CUDA（未定义 CFD_USE_CUDA）时，enabled() 恒为 false，
 * 求解器走原来的 CPU 实现，行为与 CPU 版本完全一致。
 *
 * 运行时开关：环境变量 CFD_USE_GPU=0 可强制使用 CPU。
 * 多进程（MPI）时每个进程使用一块 GPU：设备号 = 节点内进程序号 % 节点 GPU 数。
 *
 * 结果一致性：内核按与 CPU 相同的顺序累加每一行，且编译时关闭 FMA 融合（-fmad=false），
 * 除法与开方使用 IEEE 精确舍入，因此 GPU 结果与 CPU（串行/MPI）结果逐位一致。
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace gpu
{
    using ULL = unsigned long long;

    // 是否使用 GPU（首次调用时选择设备）
    bool enabled();

    // 设备描述（未启用时返回说明文字）
    std::string deviceInfo();

    /**
     * @brief GPU 上的 Jacobi 迭代器。
     *
     * 数据以 double 数组表示，每个未知量 nComp 个分量（标量 1，矢量 3）。
     * 解向量长度为 nCols（自有 + 幽灵单元），矩阵只有 nRows（自有单元）行。
     *
     * 典型流程（每次线性求解）：
     *   upload() -> setHalo()/setRowSets() -> setX0()
     *   循环 { update(); gatherSend(); ...MPI...; scatterRecv(); maxResidual(); swap(); }
     *   download()
     */
    class JacobiGpu
    {
    public:
        explicit JacobiGpu(int nComp);
        ~JacobiGpu();
        JacobiGpu(const JacobiGpu&) = delete;
        JacobiGpu& operator=(const JacobiGpu&) = delete;

        // 上传矩阵（CSR）、对角元、右端项。split 为 true 时使用"跳过对角元"的更新写法
        void upload(ULL nRows, ULL nCols,
                    const std::vector<ULL>& rowPointer,
                    const std::vector<ULL>& columnIndex,
                    const std::vector<double>& values,
                    const std::vector<double>& diag,
                    const double* b,
                    bool split);

        // 幽灵单元交换计划（拼接后的发送 / 接收局部编号）
        void setHalo(const std::vector<ULL>& sendIndexes, const std::vector<ULL>& recvIndexes);

        // 内部行（不依赖幽灵单元）与边界行，用于通信/计算重叠
        void setRowSets(const std::vector<ULL>& interiorRows, const std::vector<ULL>& haloRows);

        // 设置上一步解（长度 nCols * nComp）
        void setX0(const double* x0);

        // x = Jacobi(x0)（只写自有部分，异步）
        void update();

        // 取出 x 中需要发送给邻居的值（同步，按 setHalo 的发送顺序拼接）
        void gatherSend(std::vector<double>& packed);

        // 把接收到的值写入 x 的幽灵单元
        void scatterRecv(const std::vector<double>& packed);

        // 异步计算一组行的最大残差：rowSet = 0 全部自有行，1 内部行，2 边界行
        void launchResidual(int rowSet);

        // 等待 launchResidual 完成并返回该组行的最大残差
        double finishResidual();

        // 交换 x0 与 x
        void swap();

        // 下载 x（长度 nCols * nComp）
        void download(double* x);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace gpu

#endif // GPU_H_
