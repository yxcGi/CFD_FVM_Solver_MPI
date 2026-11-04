#include "Gpu.h"

#include <stdexcept>

/*
 * 未启用 CUDA 时的实现：enabled() 恒为 false，求解器使用 CPU。
 */

namespace gpu
{
    bool enabled()
    {
        return false;
    }

    std::string deviceInfo()
    {
        return "built without CUDA, using CPU";
    }

    struct JacobiGpu::Impl
    {
    };

    JacobiGpu::JacobiGpu(int)
    {
        throw std::runtime_error("JacobiGpu: built without CUDA support");
    }

    JacobiGpu::~JacobiGpu() = default;

    void JacobiGpu::upload(ULL, ULL, const std::vector<ULL>&, const std::vector<ULL>&,
                           const std::vector<double>&, const std::vector<double>&, const double*, bool) {}
    void JacobiGpu::setHalo(const std::vector<ULL>&, const std::vector<ULL>&) {}
    void JacobiGpu::setRowSets(const std::vector<ULL>&, const std::vector<ULL>&) {}
    void JacobiGpu::setX0(const double*) {}
    void JacobiGpu::update() {}
    void JacobiGpu::gatherSend(std::vector<double>&) {}
    void JacobiGpu::scatterRecv(const std::vector<double>&) {}
    void JacobiGpu::launchResidual(int) {}
    double JacobiGpu::finishResidual() { return 0.0; }
    void JacobiGpu::swap() {}
    void JacobiGpu::download(double*) {}

} // namespace gpu
