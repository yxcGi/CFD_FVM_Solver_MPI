#ifndef SOLVER_H_
#define SOLVER_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <thread>

#include "SparseMatrix.hpp"
#include "Field.hpp"
#include "threadpool.hpp"
#include "Parallel/Parallel.h"
#include "Gpu/Gpu.h"


/**
 * @brief 线性方程组迭代求解器
 *
 * 支持串行、线程池并行（共享内存）与 MPI 并行（分布式内存），三者可组合：
 *
 *  - MPI 并行：矩阵按行分布在各进程上（每个进程只保存自有单元的行），
 *    每次 Jacobi 迭代交换一次幽灵单元的解（非阻塞通信，与内部行的残差计算重叠），
 *    最大残差通过 Allreduce 取全局最大值。由于 Jacobi 每行的更新只依赖上一步的解，
 *    且每行的列顺序与串行相同，并行结果与串行结果逐位一致。
 *  - GPU 并行：编译时启用 CUDA 且运行时检测到 GPU 时，Jacobi 迭代在 GPU 上进行
 *    （矩阵每次求解上传一次，迭代全程留在显存中，只有幽灵单元数据经主机做 MPI 交换）。
 *    GPU 内核按与 CPU 相同的顺序计算每一行，结果与 CPU 逐位一致。
 *    环境变量 CFD_USE_GPU=0 可强制使用 CPU。
 *  - 线程池并行：setParallel() 开启。单进程时与原实现相同；
 *    多进程时每个进程默认使用 1 个线程，可通过环境变量 CFD_THREADS_PER_PROCESS 调整。
 *
 * 说明：原实现中串行分支与线程分支的 Jacobi 更新写法不同（舍入不同）。
 * 为保证与原串行程序的结果逐位一致，这里保留两种写法，并按原实现的规则
 * （由全局矩阵规模和机器核数决定）选择使用哪一种。
 */
template <typename Tp>
class Solver
{
    using ULL = unsigned long long;
    static constexpr double DEFAULT_EPSILON = 1e-6; // 默认的精度
    static constexpr int DEFAULT_OUTPUT_INTERVAL = 20;
public:
    // 求解方法
    enum class Method
    {
        Jacobi,             // 雅可比迭代
        GaussSeidel,        // 高斯赛德尔迭代
        AMG,                // 代数多重网格法
    };

public:
    Solver() = delete;
    explicit Solver(SparseMatrix<Tp>& matrix, Method method, int maxmaxIterationNum);
    Solver(const Solver&) = delete;
    Solver(Solver&&) = delete;
    ~Solver() = default;

public:
    // 初始化
    void init(const std::vector<Tp>& x0);
    void init(Tp value0);
    void init(const Field<Tp>& initField);
    // 获取初始化后的残差，判据：两次外迭代解的相对变化量
    Scalar Error();

    // 并行
    void setParallel();

    // 求解方程
    void solve();

    // 设置容差
    void setTolerance(Scalar tolerance);

    // 亚松弛
    void relax(Scalar alpha);

    // 是否有效（初始化与否）
    bool isValid() const;

private:
    // Jacobi 迭代（串行 / 线程 / MPI 统一实现）
    void JacobiSolve();

    // Jacobi 迭代的 GPU 实现（结果写入 x_，与 CPU 迭代相同）
    void JacobiIterateGpu(const std::vector<Scalar>& diag,
                          bool splitFormula,
                          const par::HaloExchange* halo,
                          const std::vector<ULL>& interiorRows,
                          const std::vector<ULL>& haloRows);

    // 原实现线程池分支使用的线程数（由矩阵全局规模和核数决定）
    static unsigned originalThreadNum(ULL globalSize);

    // 分布式网格（多进程）
    bool isDistributed() const;


private:
    SparseMatrix<Tp>& equation_;  // 方程矩阵（稀疏矩阵，压不压缩都行）
    std::vector<Tp> x0_;        // 上一步的解（并行时包含幽灵单元）
    std::vector<Tp> x_;         // 本步的解（并行时包含幽灵单元）
    Method method_;             // 求解方法
    int maxIterationNum_;       // 最大迭代次数
    Scalar tolerance_;           // 残差
    Field<Tp>* filed_{ nullptr };           // 待求解的场
    // 每多少步输出一次
    int outputInterval_;
    bool isParallel_{ false };  // 是否并行
    bool isInitialized_{ false };// 是否赋初始值
};






template<typename Tp>
inline Solver<Tp>::Solver(SparseMatrix<Tp>& matrix, Method method, int maxmaxIterationNum)
    : equation_(matrix)
    , method_(method)
    , maxIterationNum_(maxmaxIterationNum)
    , tolerance_(DEFAULT_EPSILON)
    , outputInterval_(DEFAULT_OUTPUT_INTERVAL)
{
    // 检查矩阵是否有效
    if (!equation_.isValid())
    {
        std::cerr << "Solver<Tp>::Solver(const SparseMatrix<Tp>& matrix) Error: matrix is not valid" << std::endl;
        throw std::invalid_argument("matrix is not valid");
    }

    // 矩阵有效，判断是否压缩，没压缩，则进行压缩
    if (!equation_.isCompressed())
    {
        equation_.compress();
    }
}

template<typename Tp>
inline bool Solver<Tp>::isDistributed() const
{
    const Mesh* mesh = equation_.getMesh();
    return mesh != nullptr && mesh->isDistributed();
}

template<typename Tp>
inline void Solver<Tp>::init(const std::vector<Tp>& x0)
{
    // 检查长度是否合法：可以是自有单元长度，也可以是局部存储长度（含幽灵单元）
    if (x0.size() != equation_.size() && x0.size() != equation_.colSize())
    {
        std::cerr << "Solver<Tp>::init(const std::vector<Tp>& x0) Error: length of x0 is not equal to matrix size, length of x0i is illegal" << std::endl;
        throw std::invalid_argument("length of x0 is not equal to matrix size, length of x0i is illegal");
    }
    // 赋初值
    x0_ = x0;
    x0_.resize(equation_.colSize());
    x_.resize(equation_.colSize());

    // 获取场，如果没有就是空指针
    filed_ = &equation_.getField();

    isInitialized_ = true;
}

template<typename Tp>
inline void Solver<Tp>::init(Tp value0)
{
    // 不能重复初始化
    if (isInitialized_)
    {
        std::cerr << "Solver<Tp>::init(Tp value0) Error: cannot initialize again" << std::endl;
        throw std::runtime_error("cannot initialize again");
    }

    x0_.resize(equation_.colSize(), value0);
    x_.resize(equation_.colSize());
    isInitialized_ = true;
}

template<typename Tp>
inline void Solver<Tp>::init(const Field<Tp>& initField)
{
    if (isInitialized_)
    {
        std::cerr << "Solver<Tp>::init(const Field<Tp>& initField) Error: cannot initialize again" << std::endl;
        throw std::runtime_error("cannot initialize again");
    }

    // 判断场是否有效
    if (!initField.isValid())
    {
        std::cerr << "Solver<Tp>::init(const Field<Tp>& initField) Error: initField is not valid" << std::endl;
        throw std::invalid_argument("initField is not valid");
    }

    Mesh* equationMesh = equation_.getMesh();
    Mesh* initFieldMesh = initField.getMesh();

    // 判断是否为同一个场
    // 如果稀疏矩阵不是由网格构造，则可忽略
    if (equationMesh != nullptr &&
        initFieldMesh != equationMesh)
    {
        std::cerr << "The mesh of the field is not equal to the mesh of the matrix" << std::endl;
        throw std::invalid_argument("The mesh of the field is not equal to the mesh of the matrix");
    }

    // 判断长度是否合理
    if (equation_.size() != initFieldMesh->getCellNumber())
    {
        std::cerr << "Solver<Tp>::init(const Field<Tp>& initField) Error: The size of equation matrix is not equal to the number of cells in the mesh" << std::endl;
        throw std::invalid_argument("The size of equation matrix is not equal to the number of cells in the mesh");
    }

    // 从field中的cellField_0_获取初始值
    x0_ = initField.getCellField_0().getData();
    x0_.resize(equation_.colSize());
    x_.resize(equation_.colSize());
    isInitialized_ = true;
}

template<typename Tp>
inline Scalar Solver<Tp>::Error()
{
    ULL size = equation_.size();
    const Field<Tp>& phi = equation_.getField();

    // 获取本步解和上一部解
    const std::vector<Tp>& x = phi.getCellField().getData();    // 本步解
    const std::vector<Tp>& x0 = phi.getCellField_0().getData(); // 上一部解

    // 采用相对误差方式；并行时按全局单元顺序求和，与串行结果一致
    std::vector<double> terms(2 * size);
    for (ULL i = 0; i < size; ++i)
    {
        if constexpr (std::is_same_v<Tp, Scalar>)   // 标量场
        {
            terms[2 * i] = (x[i] - x0[i]) * (x[i] - x0[i]);
            terms[2 * i + 1] = x[i] * x[i];
        }
        else if constexpr (std::is_same_v<Tp, Vector<Scalar>>)  // 矢量场用相减后的模长平方
        {
            terms[2 * i] = (x[i] - x0[i]).magnitudeSquared();
            terms[2 * i + 1] = x[i].magnitudeSquared();
        }
    }

    Scalar temp1{};     // 分子
    Scalar temp2{};     // 分母
    if (equation_.getMesh() != nullptr)
    {
        const std::vector<double> sums = equation_.getMesh()->getCellOrdering().orderedSums(terms, 2);
        temp1 = sums[0];
        temp2 = sums[1];
    }
    else
    {
        for (ULL i = 0; i < size; ++i)
        {
            temp1 += terms[2 * i];
            temp2 += terms[2 * i + 1];
        }
    }

    if (temp1 == 0) // 第一次直接返回最大值
    {
        return std::numeric_limits<Scalar>::max();
    }

    return std::sqrt(temp1 / (temp2 + 1e-30));
}



template<typename Tp>
inline void Solver<Tp>::setParallel()
{
    isParallel_ = true;
}

template<typename Tp>
inline void Solver<Tp>::solve()
{
    if (!isInitialized_)
    {
        std::cerr << "Solver<Tp>::solve() Error: cannot solve without initialization" << std::endl;
        throw std::runtime_error("cannot solve without initialization");
    }

    if (isParallel_ && method_ != Solver::Method::Jacobi)
    {
        std::cerr << "Solver<Tp>::ParallelSolve() Error: only Jacobi method is supported"
            << std::endl;
        throw std::invalid_argument("only Jacobi method is supported");
    }

    JacobiSolve();
}

template<typename Tp>
inline void Solver<Tp>::setTolerance(Scalar tolerance)
{
    if (!isInitialized_)
    {
        std::cerr << "Solver<Tp>::setTolerance() Error: cannot set tolerance without initialization" << std::endl;
        throw std::runtime_error("cannot set tolerance without initialization");
    }
    tolerance_ = tolerance;
}

template<typename Tp>
inline void Solver<Tp>::relax(Scalar alpha)
{
    if (!isInitialized_)
    {
        std::cerr << "Solver<Tp>::relax() Error: cannot relax without initialization" << std::endl;
        throw std::runtime_error("cannot relax without initialization");
    }

    // 判断alpha是否在0-1之间
    if (alpha <= 0 || alpha >= 1)
    {
        std::cerr << "Solver<Tp>::relax() Error: alpha must be in (0, 1)" << std::endl;
        throw std::runtime_error("alpha must be in (0, 1)");
    }



    // 将方程对角线元素除以alpha（只处理自有行）
    ULL size = equation_.size();
    std::vector<Scalar> diag(size);
    for (ULL i = 0; i < size; ++i)
    {
        // 对角线记录的是已经除以alpha后的值
        diag[i] = (equation_(i, i) /= alpha);
    }

    // 右端项
    for (ULL i = 0; i < size; ++i)
    {
        equation_.addB(i, (1 - alpha) * diag[i] * x0_[i]);
    }
}

template<typename Tp>
inline bool Solver<Tp>::isValid() const
{
    return isInitialized_;
}

template<typename Tp>
inline unsigned Solver<Tp>::originalThreadNum(ULL globalSize)
{
    // 与原 ParallelSolve() 的线程数规则相同：
    // Jacobi + CSR SpMV 通常受内存带宽限制，矩阵太小时不值得开太多线程。
    const unsigned hardwareThreadNum = std::thread::hardware_concurrency();
    const unsigned maxThreadNum = hardwareThreadNum > 1U ? hardwareThreadNum - 1U : 1U;

    constexpr ULL MIN_ROWS_PER_TASK = 1024;

    const ULL usefulThreadNum = std::max<ULL>(
        1ULL,
        (globalSize + MIN_ROWS_PER_TASK - 1ULL) / MIN_ROWS_PER_TASK
    );

    return static_cast<unsigned>(
        std::min<ULL>(
            static_cast<ULL>(maxThreadNum),
            usefulThreadNum
        )
        );
}

template<typename Tp>
inline void Solver<Tp>::JacobiSolve()
{
    const ULL size = equation_.size();          // 自有行数

    const std::vector<ULL>& rowPointer = equation_.getRowPointer();
    const std::vector<ULL>& columnIndex = equation_.getColIndexs();
    const std::vector<Scalar>& values = equation_.getValues();
    const std::vector<Tp>& b = equation_.getB();

    const bool distributed = isDistributed();
    const Mesh* mesh = equation_.getMesh();
    const par::HaloExchange* halo = distributed ? &mesh->getHalo() : nullptr;
    const ULL globalSize = distributed ? mesh->getGlobalCellNumber() : size;

    // 原实现：setParallel() 且线程数 > 1 时走线程分支（"分裂"写法），否则走串行写法
    const unsigned originalThreads = originalThreadNum(globalSize);
    const bool splitFormula = isParallel_ && originalThreads > 1U;

    // 先确认不存在0元素行
    for (ULL row = 0; row < size; ++row)
    {
        if (rowPointer[row] == rowPointer[row + 1])
        {
            std::cerr << "Solver<Tp>::solve() Error: The line " << row << " of the matrix is a zero element row" << std::endl;
            throw std::runtime_error("matrix has a zero element row");
        }
    }

    // 获取矩阵对角元素
    std::vector<Scalar> diag(size, Scalar{});
    for (ULL row = 0; row < size; ++row)
    {
        diag[row] = equation_(row, row);
        if (splitFormula)
        {
            bool hasDiagonal = false;
            for (ULL index = rowPointer[row]; index < rowPointer[row + 1]; ++index)
            {
                if (columnIndex[index] == row)
                {
                    hasDiagonal = true;
                    break;
                }
            }
            if (!hasDiagonal || std::abs(diag[row]) < static_cast<Scalar>(1.0e-30))
            {
                std::cerr << "Solver<Tp>::solve() Error: invalid diagonal at row "
                    << row << std::endl;
                throw std::runtime_error("invalid diagonal element");
            }
        }
    }

    if (method_ != Solver::Method::Jacobi)
    {
        // 原实现中 GaussSeidel / AMG 尚未实现（串行分支不做任何迭代）
        return;
    }

    // 幽灵单元初值
    if (halo != nullptr)
    {
        halo->exchange(x0_);
    }

    // 线程分支写回 cellField_0 时使用求解前的初始解
    std::vector<Tp> oldSolution;
    if (splitFormula)
    {
        oldSolution = x0_;
    }

    // 线程数：单进程时与原实现一致；多进程时默认每个进程 1 个线程
    unsigned threadNum = 1U;
    if (splitFormula)
    {
        threadNum = originalThreads;
        if (par::isParallel())
        {
            threadNum = 1U;
            if (const char* env = std::getenv("CFD_THREADS_PER_PROCESS"))
            {
                threadNum = static_cast<unsigned>(std::max(1, std::atoi(env)));
            }
        }
    }

    // 行分类：内部行（只依赖自有单元）与边界行（依赖幽灵单元）
    std::vector<ULL> interiorRows;
    std::vector<ULL> haloRows;
    if (distributed)
    {
        for (ULL row = 0; row < size; ++row)
        {
            bool touchesGhost = false;
            for (ULL index = rowPointer[row]; index < rowPointer[row + 1]; ++index)
            {
                if (columnIndex[index] >= size)
                {
                    touchesGhost = true;
                    break;
                }
            }
            (touchesGhost ? haloRows : interiorRows).push_back(row);
        }
    }

    // 求解结束：把解写回场
    auto writeBack = [&]() {
        if (filed_ != nullptr)
        {
            if (splitFormula)
            {
                // cellField    ：本次线性求解后的新解
                // cellField_0  ：本次线性求解前的旧解
                filed_->getCellField().getData() = x_;
                filed_->getCellField_0().getData() = oldSolution;
            }
            else
            {
                // 新值给cellField_0的，再与cellField交换
                filed_->getCellField_0().getData() = std::move(x_);
                filed_->getCellField_0().getData().
                    swap(filed_->getCellField().getData());
            }
        }
    };

    // 单行 Jacobi 更新
    auto updateRow = [&](ULL i) {
        if (splitFormula)
        {
            // x_i^{k+1} = ( b_i - sum_{j != i} A_ij x_j^k ) / A_ii
            Tp rhs = b[i];
            for (ULL index = rowPointer[i]; index < rowPointer[i + 1]; ++index)
            {
                const ULL col = columnIndex[index];
                if (col == i)
                {
                    continue;
                }
                rhs -= values[index] * x0_[col];
            }
            x_[i] = rhs / diag[i];
        }
        else
        {
            Tp sum{};
            // 只遍历当前行的非0元素
            for (ULL colId = rowPointer[i]; colId < rowPointer[i + 1]; ++colId)
            {
                sum += values[colId] * x0_[columnIndex[colId]];
            }
            sum -= x0_[i] * diag[i];
            // 计算当前步解
            x_[i] = (b[i] - sum) / diag[i];
        }
    };

    // 单行残差
    auto residualRow = [&](ULL i) -> Scalar {
        if constexpr (std::is_same_v<Tp, Scalar>)
        {
            Scalar ax{};
            for (ULL index = rowPointer[i]; index < rowPointer[i + 1]; ++index)
            {
                ax += values[index] * x_[columnIndex[index]];
            }
            return std::abs(b[i] - ax);
        }
        else if constexpr (std::is_same_v<Tp, Vector<Scalar>>)
        {
            Vector<Scalar> ax{};
            for (ULL index = rowPointer[i]; index < rowPointer[i + 1]; ++index)
            {
                ax += values[index] * x_[columnIndex[index]];
            }
            return splitFormula ? (b[i] - ax).magnitude() : (ax - b[i]).magnitude();
        }
        else
        {
            static_assert(std::is_same_v<Tp, Scalar> || std::is_same_v<Tp, Vector<Scalar>>,
                          "Solver: unsupported Tp type");
            return Scalar{};
        }
    };

    // 线程划分
    ThreadPool* pool = nullptr;
    if (threadNum > 1U)
    {
        pool = &ThreadPool::getInstance();
        // 如果线程池已经启动，新的 ThreadPool::start() 会直接 return，不会重复创建线程。
        pool->start(threadNum);
    }

    // 对 [0, n) 按线程均分后执行 func(begin, end)
    auto runChunks = [&](ULL n, auto&& func) {
        if (pool == nullptr || n < threadNum)
        {
            func(0ULL, n);
            return;
        }
        std::vector<std::future<void>> futures(threadNum);
        const ULL base = n / threadNum;
        const ULL remain = n % threadNum;
        ULL begin = 0;
        for (unsigned tid = 0; tid < threadNum; ++tid)
        {
            const ULL count = base + (tid < remain ? 1ULL : 0ULL);
            futures[tid] = pool->submitTask(func, begin, begin + count);
            begin += count;
        }
        for (auto& f : futures)
        {
            f.get();
        }
    };

    auto updateAll = [&]() {
        runChunks(size, [&](ULL begin, ULL end) {
            for (ULL row = begin; row < end; ++row)
            {
                updateRow(row);
            }
        });
    };

    // 对行列表（rows == nullptr 表示全部自有行）求最大残差
    auto maxResidualOf = [&](const std::vector<ULL>* rows) -> Scalar {
        const ULL n = rows ? rows->size() : size;
        if (pool == nullptr || n < threadNum)
        {
            Scalar localMax{};
            for (ULL k = 0; k < n; ++k)
            {
                localMax = std::max(localMax, residualRow(rows ? (*rows)[k] : k));
            }
            return localMax;
        }
        std::vector<std::future<Scalar>> futures(threadNum);
        const ULL base = n / threadNum;
        const ULL remain = n % threadNum;
        ULL begin = 0;
        auto task = [&](ULL b0, ULL e0) -> Scalar {
            Scalar localMax{};
            for (ULL k = b0; k < e0; ++k)
            {
                localMax = std::max(localMax, residualRow(rows ? (*rows)[k] : k));
            }
            return localMax;
        };
        for (unsigned tid = 0; tid < threadNum; ++tid)
        {
            const ULL count = base + (tid < remain ? 1ULL : 0ULL);
            futures[tid] = pool->submitTask(task, begin, begin + count);
            begin += count;
        }
        Scalar localMax{};
        for (auto& f : futures)
        {
            localMax = std::max(localMax, f.get());
        }
        return localMax;
    };

    if (gpu::enabled())
    {
        JacobiIterateGpu(diag, splitFormula, halo, interiorRows, haloRows);
        writeBack();
        return;
    }

    for (int it = 0; it < maxIterationNum_; ++it)
    {
        // 1. Jacobi 更新（只读 x0_，只写 x_ 的自有部分）
        updateAll();

        // 2. 残差：发起 x_ 幽灵单元交换，同时计算内部行残差
        Scalar maxResidual{};
        if (halo != nullptr)
        {
            halo->start(x_);
            maxResidual = maxResidualOf(&interiorRows);
            halo->finish(x_);
            maxResidual = std::max(maxResidual, maxResidualOf(&haloRows));
            maxResidual = par::allReduceMax(maxResidual);
        }
        else
        {
            maxResidual = maxResidualOf(nullptr);
        }

        // 残差满足要求或到最大迭代次数则返回，本次求解结束
        if (maxResidual < tolerance_ ||
            it == maxIterationNum_ - 1)
        {
            writeBack();
            return;
        }

        // Jacobi 下一轮只需要交换新旧解（x_ 的幽灵单元已是最新值）
        x0_.swap(x_);
    }
}


template<typename Tp>
inline void Solver<Tp>::JacobiIterateGpu(const std::vector<Scalar>& diag,
                                         bool splitFormula,
                                         const par::HaloExchange* halo,
                                         const std::vector<ULL>& interiorRows,
                                         const std::vector<ULL>& haloRows)
{
    static_assert(std::is_same_v<Tp, Scalar> || std::is_same_v<Tp, Vector<Scalar>>,
                  "Solver: unsupported Tp type");
    constexpr int nc = static_cast<int>(par::nComponents<Tp>());

    const ULL size = equation_.size();
    const ULL colSize = equation_.colSize();

    // 矢量场：CPU 实现中 Vector / 0（|diag| < 1e-12，见 Vector::isZero）会抛出异常，这里保持相同行为
    if constexpr (std::is_same_v<Tp, Vector<Scalar>>)
    {
        for (ULL row = 0; row < size; ++row)
        {
            if (std::abs(diag[row]) < 1e-12)
            {
                std::cerr << "Error: Division by zero in Vector::operator/." << std::endl;
                throw std::invalid_argument("Division by zero");
            }
        }
    }

    gpu::JacobiGpu device(nc);
    device.upload(size, colSize,
                  equation_.getRowPointer(), equation_.getColIndexs(), equation_.getValues(),
                  diag, reinterpret_cast<const double*>(equation_.getB().data()),
                  splitFormula);

    const bool overlap = (halo != nullptr && !halo->empty());
    if (overlap)
    {
        std::vector<ULL> sendIndexes;
        std::vector<ULL> recvIndexes;
        for (const auto& list : halo->getSendIndexes())
        {
            sendIndexes.insert(sendIndexes.end(), list.begin(), list.end());
        }
        for (const auto& list : halo->getRecvIndexes())
        {
            recvIndexes.insert(recvIndexes.end(), list.begin(), list.end());
        }
        device.setHalo(sendIndexes, recvIndexes);
        device.setRowSets(interiorRows, haloRows);
    }

    // x0_ 的幽灵单元已在调用前交换
    device.setX0(reinterpret_cast<const double*>(x0_.data()));

    std::vector<double> sendPacked;
    std::vector<double> recvPacked;
    for (int it = 0; it < maxIterationNum_; ++it)
    {
        // 1. Jacobi 更新
        device.update();

        // 2. 残差：发起 x 幽灵单元交换，同时在 GPU 上计算内部行残差
        Scalar maxResidual{};
        if (halo != nullptr)
        {
            if (overlap)
            {
                device.gatherSend(sendPacked);
                halo->startPacked(sendPacked, nc);
                device.launchResidual(1);
                halo->finishPacked(recvPacked, nc);
                maxResidual = device.finishResidual();
                device.scatterRecv(recvPacked);
                device.launchResidual(2);
                maxResidual = std::max(maxResidual, device.finishResidual());
            }
            else
            {
                device.launchResidual(0);
                maxResidual = device.finishResidual();
            }
            maxResidual = par::allReduceMax(maxResidual);
        }
        else
        {
            device.launchResidual(0);
            maxResidual = device.finishResidual();
        }

        if (maxResidual < tolerance_ ||
            it == maxIterationNum_ - 1)
        {
            break;
        }

        device.swap();
    }

    x_.resize(colSize);
    device.download(reinterpret_cast<double*>(x_.data()));
}


#endif // SOLVER_H_
