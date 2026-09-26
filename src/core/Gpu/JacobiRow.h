#ifndef JACOBI_ROW_H_
#define JACOBI_ROW_H_

/*
 * Jacobi 单行计算（GPU 内核与主机端模拟后端共用同一份代码）。
 *
 * 运算顺序与 Solver.hpp 中 CPU 实现逐项对应，配合关闭 FMA 融合
 * （主机 -ffp-contract=off，设备 -fmad=false），结果逐位一致。
 * 数据按 double 数组存放，每个未知量 NC 个分量（标量 1，矢量 3）。
 */

#include <cmath>

#ifdef __CUDACC__
#define CFD_HD __host__ __device__ __forceinline__
#else
#define CFD_HD inline
#endif

namespace gpu
{
    // 第 i 行的 Jacobi 更新：x_i = f(x0)
    template<int NC, bool SPLIT>
    CFD_HD void jacobiUpdateRow(int i,
                                const int* rowPtr, const int* colIdx, const double* values,
                                const double* diag, const double* b,
                                const double* x0, double* x)
    {
        const int begin = rowPtr[i];
        const int end = rowPtr[i + 1];
        const double d = diag[i];

        if constexpr (SPLIT)
        {
            // x_i = ( b_i - sum_{j != i} A_ij x_j ) / A_ii
            double rhs[NC];
            for (int c = 0; c < NC; ++c)
            {
                rhs[c] = b[i * NC + c];
            }
            for (int k = begin; k < end; ++k)
            {
                const int col = colIdx[k];
                if (col == i)
                {
                    continue;
                }
                const double v = values[k];
                for (int c = 0; c < NC; ++c)
                {
                    rhs[c] -= v * x0[col * NC + c];
                }
            }
            for (int c = 0; c < NC; ++c)
            {
                x[i * NC + c] = rhs[c] / d;
            }
        }
        else
        {
            // sum = sum_j A_ij x_j - A_ii x_i ; x_i = (b_i - sum) / A_ii
            double sum[NC];
            for (int c = 0; c < NC; ++c)
            {
                sum[c] = 0.0;
            }
            for (int k = begin; k < end; ++k)
            {
                const int col = colIdx[k];
                const double v = values[k];
                for (int c = 0; c < NC; ++c)
                {
                    sum[c] += v * x0[col * NC + c];
                }
            }
            for (int c = 0; c < NC; ++c)
            {
                sum[c] -= x0[i * NC + c] * d;
                x[i * NC + c] = (b[i * NC + c] - sum[c]) / d;
            }
        }
    }

    // 第 i 行的残差 |b_i - (A x)_i|（矢量取模）
    template<int NC, bool SPLIT>
    CFD_HD double jacobiResidualRow(int i,
                                    const int* rowPtr, const int* colIdx, const double* values,
                                    const double* b, const double* x)
    {
        double ax[NC];
        for (int c = 0; c < NC; ++c)
        {
            ax[c] = 0.0;
        }
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k)
        {
            const int col = colIdx[k];
            const double v = values[k];
            for (int c = 0; c < NC; ++c)
            {
                ax[c] += v * x[col * NC + c];
            }
        }
        if constexpr (NC == 1)
        {
            return std::fabs(b[i] - ax[0]);
        }
        else
        {
            // CPU：split 写法为 (b - ax).magnitude()，否则为 (ax - b).magnitude()
            // magnitude = sqrt(x*x + y*y + z*z)
            double mag2 = 0.0;
            for (int c = 0; c < NC; ++c)
            {
                const double diff = SPLIT ? (b[i * NC + c] - ax[c]) : (ax[c] - b[i * NC + c]);
                const double sq = diff * diff;
                mag2 = (c == 0) ? sq : mag2 + sq;
            }
            return std::sqrt(mag2);
        }
    }

    // 与 std::max(a, b) 语义相同：b 为 NaN 时保留 a
    CFD_HD double maxLike(double a, double b)
    {
        return (a < b) ? b : a;
    }

} // namespace gpu

#endif // JACOBI_ROW_H_
