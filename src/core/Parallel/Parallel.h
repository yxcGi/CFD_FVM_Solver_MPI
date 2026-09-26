#ifndef PARALLEL_H_
#define PARALLEL_H_

/*
 * MPI 并行通信层
 *
 * 该层对 MPI 做了一层很薄的封装，求解器其余部分只通过这里访问 MPI：
 *
 *   par::Environment     MPI 初始化/结束（每个 main() 开头构造一次）
 *   par::rank()/size()   进程号/进程数
 *   par::allReduceMax()  全局最大值归约
 *   par::HaloExchange    幽灵单元（halo）数据交换
 *   par::GlobalOrdering  按全局编号收集数据、按串行顺序求和
 *
 * 若编译时未启用 MPI（未定义 CFD_USE_MPI），所有接口退化为单进程实现，
 * 程序与原串行版本行为完全一致。
 */

#include <cstddef>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace par
{
    using ULL = unsigned long long;

    /* ======================== 进程环境 ======================== */

    /**
     * @brief MPI 运行环境（RAII）。
     *
     * 构造时调用 MPI_Init，析构时调用 MPI_Finalize。
     * 非 0 号进程的 std::cout 会被静默，保证屏幕输出与串行版本一致；
     * std::cerr 保留，便于定位错误。
     */
    class Environment
    {
    public:
        Environment(int& argc, char**& argv);
        Environment();
        ~Environment();
        Environment(const Environment&) = delete;
        Environment& operator=(const Environment&) = delete;
    };

    // 当前进程号（未初始化 MPI 时为 0）
    int rank();
    // 进程总数（未初始化 MPI 时为 1）
    int size();
    // 是否为 0 号进程
    bool isMaster();
    // 是否多进程运行
    bool isParallel();
    // 同一节点上的进程数（用于决定每个进程可用的线程数）
    int nodeLocalSize();

    // 终止所有进程
    [[noreturn]] void abort(int errorCode = 1);
    void barrier();

    /* ======================== 归约 ======================== */
    double allReduceMax(double value);
    double allReduceSum(double value);
    ULL allReduceSum(ULL value);

    /* ======================== 点对点（字节流） ======================== */
    void sendBytes(const std::vector<char>& buffer, int dest, int tag);
    std::vector<char> recvBytes(int source, int tag);
    void broadcastBytes(std::vector<char>& buffer, int root = 0);

    /* ======================== 数据类型分量 ======================== */

    /**
     * @brief 将 Scalar / Vector / Tensor 视为若干个 double 进行传输。
     *
     * 这些类型只包含 double 成员（Vector: 3 个，Tensor: 9 个），
     * 按字节拷贝即可逐位保持数值。
     */
    template<typename Tp>
    constexpr std::size_t nComponents()
    {
        static_assert(sizeof(Tp) % sizeof(double) == 0,
                      "par: only double-based types can be communicated");
        return sizeof(Tp) / sizeof(double);
    }

    template<typename Tp>
    inline void packValue(const Tp& value, double* dest)
    {
        std::memcpy(dest, &value, sizeof(Tp));
    }

    template<typename Tp>
    inline void unpackValue(const double* src, Tp& value)
    {
        std::memcpy(static_cast<void*>(&value), src, sizeof(Tp));
    }


    /* ======================== Halo 交换 ======================== */

    /**
     * @brief 幽灵单元数据交换计划。
     *
     * 每个邻居进程 p 对应两个列表：
     *   sendIndexes[p]：本进程自有单元的局部编号，其值需要发送给 p
     *   recvIndexes[p]：本进程幽灵单元的局部编号，其值从 p 接收
     * 两端列表按全局单元编号升序一一对应。
     *
     * 支持非阻塞交换：start() 发起通信，finish() 等待并写回幽灵单元，
     * 两者之间可以进行不依赖幽灵单元的计算（通信/计算重叠）。
     */
    class HaloExchange
    {
    public:
        HaloExchange() = default;

        void setup(std::vector<int> neighbours,
                   std::vector<std::vector<ULL>> sendIndexes,
                   std::vector<std::vector<ULL>> recvIndexes);

        bool empty() const { return neighbours_.empty(); }
        const std::vector<int>& getNeighbours() const { return neighbours_; }
        const std::vector<std::vector<ULL>>& getSendIndexes() const { return sendIndexes_; }
        const std::vector<std::vector<ULL>>& getRecvIndexes() const { return recvIndexes_; }

        // 阻塞式交换：data 为局部存储（自有 + 幽灵）
        template<typename Tp>
        void exchange(std::vector<Tp>& data) const
        {
            if (neighbours_.empty())
            {
                return;
            }
            start(data);
            finish(data);
        }

        // 发起非阻塞交换
        template<typename Tp>
        void start(const std::vector<Tp>& data) const
        {
            if (neighbours_.empty())
            {
                return;
            }
            constexpr std::size_t nc = nComponents<Tp>();
            prepareBuffers(nc);
            for (std::size_t n = 0; n < neighbours_.size(); ++n)
            {
                double* buf = sendBuffers_[n].data();
                for (std::size_t k = 0; k < sendIndexes_[n].size(); ++k)
                {
                    packValue(data[sendIndexes_[n][k]], buf + k * nc);
                }
            }
            postCommunication(nc);
        }

        // 等待交换完成并写回幽灵单元
        template<typename Tp>
        void finish(std::vector<Tp>& data) const
        {
            if (neighbours_.empty())
            {
                return;
            }
            constexpr std::size_t nc = nComponents<Tp>();
            waitCommunication();
            for (std::size_t n = 0; n < neighbours_.size(); ++n)
            {
                const double* buf = recvBuffers_[n].data();
                for (std::size_t k = 0; k < recvIndexes_[n].size(); ++k)
                {
                    unpackValue(buf + k * nc, data[recvIndexes_[n][k]]);
                }
            }
        }

    private:
        void prepareBuffers(std::size_t nComponents) const;
        void postCommunication(std::size_t nComponents) const;
        void waitCommunication() const;

    private:
        std::vector<int> neighbours_;
        std::vector<std::vector<ULL>> sendIndexes_;
        std::vector<std::vector<ULL>> recvIndexes_;

        // 通信缓冲区（mutable：交换不改变计划本身）
        mutable std::vector<std::vector<double>> sendBuffers_;
        mutable std::vector<std::vector<double>> recvBuffers_;
        mutable std::vector<char> requests_;   // 存放 MPI_Request 的原始内存
        mutable bool inFlight_ = false;
    };


    /* ======================== 全局排序 ======================== */

    /**
     * @brief 描述"各进程的若干局部条目在串行程序中的全局顺序"。
     *
     * 用途：
     *  1. 把分布式的单元场收集到 0 号进程，并按全局单元编号排列（结果输出）；
     *  2. 按串行程序的循环顺序做求和（收敛判据），保证与串行结果逐位一致。
     *
     * 每个进程只保存本进程条目数；0 号进程额外保存所有进程条目的全局位置。
     */
    class GlobalOrdering
    {
    public:
        GlobalOrdering() = default;

        // 串行（单进程）情形：条目顺序即全局顺序
        void setupSerial(ULL nLocal);

        // 并行情形：nLocal 为本进程条目数；globalPositions 仅在 0 号进程有效，
        // 按进程顺序拼接，globalPositions[k] 为拼接序列第 k 个条目在全局序列中的位置
        void setup(ULL nLocal, ULL nGlobal, std::vector<ULL> globalPositions);

        ULL getLocalNumber() const { return nLocal_; }
        ULL getGlobalNumber() const { return nGlobal_; }

        /**
         * @brief 收集到 0 号进程。
         * @param local 本进程条目（长度 >= nLocal，只取前 nLocal 个）
         * @return 0 号进程上按全局顺序排列的数组；其余进程返回空数组
         */
        template<typename Tp>
        std::vector<Tp> gatherToMaster(const std::vector<Tp>& local) const
        {
            constexpr std::size_t nc = nComponents<Tp>();
            std::vector<double> packed(nLocal_ * nc);
            for (ULL i = 0; i < nLocal_; ++i)
            {
                packValue(local[i], packed.data() + i * nc);
            }
            std::vector<double> global = gatherDoubles(packed, nc);
            std::vector<Tp> result;
            if (isMaster())
            {
                result.resize(nGlobal_);
                for (ULL i = 0; i < nGlobal_; ++i)
                {
                    unpackValue(global.data() + i * nc, result[i]);
                }
            }
            return result;
        }

        /**
         * @brief 按全局顺序依次累加（与串行 for 循环的累加顺序一致）。
         *
         * terms 按条目交错存放 nSums 个求和项：
         *   terms[i * nSums + s] 为第 i 个条目对第 s 个和的贡献。
         * 返回 nSums 个和，所有进程得到相同结果。
         */
        std::vector<double> orderedSums(const std::vector<double>& terms, std::size_t nSums) const;

    private:
        // 收集并按全局顺序重排（仅 0 号进程返回有效数据）
        std::vector<double> gatherDoubles(const std::vector<double>& local, std::size_t nc) const;

    private:
        ULL nLocal_ = 0;
        ULL nGlobal_ = 0;
        bool serial_ = true;
        std::vector<int> counts_;          // 0 号进程：各进程条目数
        std::vector<ULL> globalPositions_; // 0 号进程：拼接序列 -> 全局位置
    };

} // namespace par

#endif // PARALLEL_H_
