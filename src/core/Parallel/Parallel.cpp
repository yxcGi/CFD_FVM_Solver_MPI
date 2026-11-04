#include "Parallel.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <streambuf>

#ifdef CFD_USE_MPI
#include <mpi.h>
#endif

namespace par
{
    namespace
    {
        // 丢弃所有输出的流缓冲区，用于静默非 0 号进程的 std::cout
        class NullBuffer : public std::streambuf
        {
        protected:
            int overflow(int c) override { return traits_type::not_eof(c); }
            std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
        };

        NullBuffer nullBuffer;
        std::streambuf* savedCoutBuffer = nullptr;
        bool ownsMpi = false;
        int cachedRank = 0;
        int cachedSize = 1;
        int cachedNodeLocalSize = 1;
        int cachedNodeLocalRank = 0;

        void setupProcessInfo()
        {
#ifdef CFD_USE_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &cachedRank);
            MPI_Comm_size(MPI_COMM_WORLD, &cachedSize);

            MPI_Comm nodeComm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
            MPI_Comm_size(nodeComm, &cachedNodeLocalSize);
            MPI_Comm_rank(nodeComm, &cachedNodeLocalRank);
            MPI_Comm_free(&nodeComm);
#endif
            if (cachedRank != 0)
            {
                savedCoutBuffer = std::cout.rdbuf(&nullBuffer);
            }

            if (cachedSize > 1)
            {
                // 任一进程出现未捕获异常时终止全部进程，避免其余进程死等
                std::set_terminate([]() {
                    std::cerr << "[rank " << cachedRank << "] terminate called, aborting all processes" << std::endl;
                    if (auto e = std::current_exception())
                    {
                        try { std::rethrow_exception(e); }
                        catch (const std::exception& ex) { std::cerr << "  what(): " << ex.what() << std::endl; }
                        catch (...) {}
                    }
                    abort(1);
                });
            }
        }

#ifdef CFD_USE_MPI
        bool mpiActive()
        {
            int initialized = 0;
            int finalized = 0;
            MPI_Initialized(&initialized);
            MPI_Finalized(&finalized);
            return initialized && !finalized;
        }
#endif
    }

    /* ======================== Environment ======================== */

    Environment::Environment(int& argc, char**& argv)
    {
#ifdef CFD_USE_MPI
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            MPI_Init(&argc, &argv);
            ownsMpi = true;
        }
#else
        (void)argc;
        (void)argv;
#endif
        setupProcessInfo();
    }

    Environment::Environment()
    {
#ifdef CFD_USE_MPI
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            MPI_Init(nullptr, nullptr);
            ownsMpi = true;
        }
#endif
        setupProcessInfo();
    }

    Environment::~Environment()
    {
        std::cout.flush();
        if (savedCoutBuffer != nullptr)
        {
            std::cout.rdbuf(savedCoutBuffer);
            savedCoutBuffer = nullptr;
        }
#ifdef CFD_USE_MPI
        if (ownsMpi && mpiActive())
        {
            MPI_Finalize();
            ownsMpi = false;
        }
#endif
    }

    int rank() { return cachedRank; }
    int size() { return cachedSize; }
    bool isMaster() { return cachedRank == 0; }
    bool isParallel() { return cachedSize > 1; }
    int nodeLocalSize() { return cachedNodeLocalSize; }
    int nodeLocalRank() { return cachedNodeLocalRank; }

    void abort(int errorCode)
    {
#ifdef CFD_USE_MPI
        if (mpiActive())
        {
            MPI_Abort(MPI_COMM_WORLD, errorCode);
        }
#endif
        std::exit(errorCode);
    }

    void barrier()
    {
#ifdef CFD_USE_MPI
        if (cachedSize > 1)
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif
    }

    /* ======================== 归约 ======================== */

    double allReduceMax(double value)
    {
#ifdef CFD_USE_MPI
        if (cachedSize > 1)
        {
            double result = value;
            MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            return result;
        }
#endif
        return value;
    }

    double allReduceSum(double value)
    {
#ifdef CFD_USE_MPI
        if (cachedSize > 1)
        {
            double result = value;
            MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            return result;
        }
#endif
        return value;
    }

    ULL allReduceSum(ULL value)
    {
#ifdef CFD_USE_MPI
        if (cachedSize > 1)
        {
            unsigned long long result = value;
            MPI_Allreduce(&value, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            return result;
        }
#endif
        return value;
    }

    /* ======================== 点对点 ======================== */

#ifdef CFD_USE_MPI
    namespace
    {
        // 大消息分块，避免 int 计数溢出
        constexpr std::size_t MAX_CHUNK = static_cast<std::size_t>(1) << 30;
    }
#endif

    void sendBytes(const std::vector<char>& buffer, int dest, int tag)
    {
#ifdef CFD_USE_MPI
        unsigned long long n = buffer.size();
        MPI_Send(&n, 1, MPI_UNSIGNED_LONG_LONG, dest, tag, MPI_COMM_WORLD);
        for (std::size_t offset = 0; offset < buffer.size(); offset += MAX_CHUNK)
        {
            const int chunk = static_cast<int>(std::min(MAX_CHUNK, buffer.size() - offset));
            MPI_Send(buffer.data() + offset, chunk, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
        }
#else
        (void)buffer;
        (void)dest;
        (void)tag;
        throw std::runtime_error("par::sendBytes: MPI is not enabled");
#endif
    }

    std::vector<char> recvBytes(int source, int tag)
    {
#ifdef CFD_USE_MPI
        unsigned long long n = 0;
        MPI_Recv(&n, 1, MPI_UNSIGNED_LONG_LONG, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<char> buffer(n);
        for (std::size_t offset = 0; offset < buffer.size(); offset += MAX_CHUNK)
        {
            const int chunk = static_cast<int>(std::min(MAX_CHUNK, buffer.size() - offset));
            MPI_Recv(buffer.data() + offset, chunk, MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        return buffer;
#else
        (void)source;
        (void)tag;
        throw std::runtime_error("par::recvBytes: MPI is not enabled");
#endif
    }

    void broadcastBytes(std::vector<char>& buffer, int root)
    {
#ifdef CFD_USE_MPI
        if (cachedSize == 1)
        {
            return;
        }
        unsigned long long n = buffer.size();
        MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, root, MPI_COMM_WORLD);
        buffer.resize(n);
        for (std::size_t offset = 0; offset < buffer.size(); offset += MAX_CHUNK)
        {
            const int chunk = static_cast<int>(std::min(MAX_CHUNK, buffer.size() - offset));
            MPI_Bcast(buffer.data() + offset, chunk, MPI_BYTE, root, MPI_COMM_WORLD);
        }
#else
        (void)buffer;
        (void)root;
#endif
    }

    /* ======================== HaloExchange ======================== */

    void HaloExchange::setup(std::vector<int> neighbours,
                             std::vector<std::vector<ULL>> sendIndexes,
                             std::vector<std::vector<ULL>> recvIndexes)
    {
        if (neighbours.size() != sendIndexes.size() || neighbours.size() != recvIndexes.size())
        {
            throw std::invalid_argument("HaloExchange::setup: inconsistent neighbour lists");
        }
        neighbours_ = std::move(neighbours);
        sendIndexes_ = std::move(sendIndexes);
        recvIndexes_ = std::move(recvIndexes);
        sendBuffers_.assign(neighbours_.size(), {});
        recvBuffers_.assign(neighbours_.size(), {});
#ifdef CFD_USE_MPI
        requests_.assign(2 * neighbours_.size() * sizeof(MPI_Request), 0);
#endif
        inFlight_ = false;
    }

    void HaloExchange::prepareBuffers(std::size_t nc) const
    {
        if (inFlight_)
        {
            throw std::logic_error("HaloExchange: start() called while a previous exchange is in flight");
        }
        for (std::size_t n = 0; n < neighbours_.size(); ++n)
        {
            sendBuffers_[n].resize(sendIndexes_[n].size() * nc);
            recvBuffers_[n].resize(recvIndexes_[n].size() * nc);
        }
    }

    void HaloExchange::postCommunication(std::size_t nc) const
    {
        (void)nc;
#ifdef CFD_USE_MPI
        constexpr int HALO_TAG = 7001;
        MPI_Request* req = reinterpret_cast<MPI_Request*>(requests_.data());
        const std::size_t nn = neighbours_.size();
        for (std::size_t n = 0; n < nn; ++n)
        {
            MPI_Irecv(recvBuffers_[n].data(), static_cast<int>(recvBuffers_[n].size()), MPI_DOUBLE,
                      neighbours_[n], HALO_TAG, MPI_COMM_WORLD, &req[n]);
        }
        for (std::size_t n = 0; n < nn; ++n)
        {
            MPI_Isend(sendBuffers_[n].data(), static_cast<int>(sendBuffers_[n].size()), MPI_DOUBLE,
                      neighbours_[n], HALO_TAG, MPI_COMM_WORLD, &req[nn + n]);
        }
        inFlight_ = true;
#else
        throw std::runtime_error("HaloExchange: MPI is not enabled");
#endif
    }

    void HaloExchange::waitCommunication() const
    {
#ifdef CFD_USE_MPI
        if (!inFlight_)
        {
            throw std::logic_error("HaloExchange: finish() called without start()");
        }
        MPI_Request* req = reinterpret_cast<MPI_Request*>(requests_.data());
        MPI_Waitall(static_cast<int>(2 * neighbours_.size()), req, MPI_STATUSES_IGNORE);
        inFlight_ = false;
#endif
    }

    void HaloExchange::startPacked(const std::vector<double>& packed, std::size_t nc) const
    {
        if (neighbours_.empty())
        {
            return;
        }
        if (packed.size() != totalSendCount() * nc)
        {
            throw std::invalid_argument("HaloExchange::startPacked: wrong buffer size");
        }
        prepareBuffers(nc);
        std::size_t offset = 0;
        for (std::size_t n = 0; n < neighbours_.size(); ++n)
        {
            std::copy(packed.begin() + offset, packed.begin() + offset + sendBuffers_[n].size(),
                      sendBuffers_[n].begin());
            offset += sendBuffers_[n].size();
        }
        postCommunication(nc);
    }

    void HaloExchange::finishPacked(std::vector<double>& packed, std::size_t nc) const
    {
        if (neighbours_.empty())
        {
            packed.clear();
            return;
        }
        waitCommunication();
        packed.resize(totalRecvCount() * nc);
        std::size_t offset = 0;
        for (std::size_t n = 0; n < neighbours_.size(); ++n)
        {
            std::copy(recvBuffers_[n].begin(), recvBuffers_[n].end(), packed.begin() + offset);
            offset += recvBuffers_[n].size();
        }
    }

    std::size_t HaloExchange::totalSendCount() const
    {
        std::size_t total = 0;
        for (const auto& list : sendIndexes_)
        {
            total += list.size();
        }
        return total;
    }

    std::size_t HaloExchange::totalRecvCount() const
    {
        std::size_t total = 0;
        for (const auto& list : recvIndexes_)
        {
            total += list.size();
        }
        return total;
    }

    /* ======================== GlobalOrdering ======================== */

    void GlobalOrdering::setupSerial(ULL nLocal)
    {
        serial_ = true;
        nLocal_ = nLocal;
        nGlobal_ = nLocal;
        counts_.clear();
        globalPositions_.clear();
    }

    void GlobalOrdering::setup(ULL nLocal, ULL nGlobal, std::vector<ULL> globalPositions)
    {
        serial_ = (cachedSize == 1);
        nLocal_ = nLocal;
        nGlobal_ = nGlobal;
        globalPositions_ = std::move(globalPositions);
        counts_.clear();
#ifdef CFD_USE_MPI
        if (!serial_)
        {
            int myCount = static_cast<int>(nLocal_);
            if (cachedRank == 0)
            {
                counts_.resize(cachedSize);
            }
            MPI_Gather(&myCount, 1, MPI_INT, counts_.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (cachedRank == 0)
            {
                ULL total = 0;
                for (int c : counts_)
                {
                    total += static_cast<ULL>(c);
                }
                if (total != globalPositions_.size() || total != nGlobal_)
                {
                    throw std::runtime_error("GlobalOrdering::setup: inconsistent global positions");
                }
            }
        }
#endif
    }

    std::vector<double> GlobalOrdering::gatherDoubles(const std::vector<double>& local, std::size_t nc) const
    {
        if (serial_)
        {
            return std::vector<double>(local.begin(), local.begin() + nLocal_ * nc);
        }
#ifdef CFD_USE_MPI
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> gathered;
        if (cachedRank == 0)
        {
            counts.resize(cachedSize);
            displs.resize(cachedSize);
            int offset = 0;
            for (int p = 0; p < cachedSize; ++p)
            {
                counts[p] = counts_[p] * static_cast<int>(nc);
                displs[p] = offset;
                offset += counts[p];
            }
            gathered.resize(static_cast<std::size_t>(offset));
        }
        MPI_Gatherv(local.data(), static_cast<int>(nLocal_ * nc), MPI_DOUBLE,
                    gathered.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        std::vector<double> ordered;
        if (cachedRank == 0)
        {
            ordered.resize(nGlobal_ * nc);
            for (ULL k = 0; k < nGlobal_; ++k)
            {
                const ULL pos = globalPositions_[k];
                for (std::size_t c = 0; c < nc; ++c)
                {
                    ordered[pos * nc + c] = gathered[k * nc + c];
                }
            }
        }
        return ordered;
#else
        return {};
#endif
    }

    std::vector<double> GlobalOrdering::orderedSums(const std::vector<double>& terms, std::size_t nSums) const
    {
        std::vector<double> sums(nSums, 0.0);
        const std::vector<double> ordered = gatherDoubles(terms, nSums);
        if (serial_ || cachedRank == 0)
        {
            for (ULL i = 0; i < nGlobal_; ++i)
            {
                for (std::size_t s = 0; s < nSums; ++s)
                {
                    sums[s] += ordered[i * nSums + s];
                }
            }
        }
#ifdef CFD_USE_MPI
        if (!serial_)
        {
            MPI_Bcast(sums.data(), static_cast<int>(nSums), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
#endif
        return sums;
    }

} // namespace par
