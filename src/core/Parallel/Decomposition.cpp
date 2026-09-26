#include "Decomposition.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace par
{
    namespace
    {
        using ULL = unsigned long long;

        double component(const Vector<double>& p, int axis)
        {
            return axis == 0 ? p.x() : (axis == 1 ? p.y() : p.z());
        }

        void bisect(const std::vector<Vector<double>>& centers,
                    std::vector<ULL>::iterator begin,
                    std::vector<ULL>::iterator end,
                    int firstPart,
                    int nParts,
                    std::vector<int>& part)
        {
            if (nParts == 1)
            {
                for (auto it = begin; it != end; ++it)
                {
                    part[*it] = firstPart;
                }
                return;
            }

            // 包围盒最长方向
            double lo[3] = {
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max()
            };
            double hi[3] = {
                std::numeric_limits<double>::lowest(),
                std::numeric_limits<double>::lowest(),
                std::numeric_limits<double>::lowest()
            };
            for (auto it = begin; it != end; ++it)
            {
                for (int a = 0; a < 3; ++a)
                {
                    const double v = component(centers[*it], a);
                    lo[a] = std::min(lo[a], v);
                    hi[a] = std::max(hi[a], v);
                }
            }
            int axis = 0;
            for (int a = 1; a < 3; ++a)
            {
                if (hi[a] - lo[a] > hi[axis] - lo[axis])
                {
                    axis = a;
                }
            }

            // 按比例切分：左侧 nLeft 个分区
            const int nLeft = nParts / 2;
            const ULL n = static_cast<ULL>(end - begin);
            const ULL nLeftCells = (n * static_cast<ULL>(nLeft) + static_cast<ULL>(nParts) / 2) / static_cast<ULL>(nParts);
            auto middle = begin + static_cast<std::ptrdiff_t>(nLeftCells);

            // 以 (坐标, 全局编号) 排序，保证结果确定
            std::nth_element(begin, middle, end, [&](ULL a, ULL b) {
                const double va = component(centers[a], axis);
                const double vb = component(centers[b], axis);
                if (va != vb)
                {
                    return va < vb;
                }
                return a < b;
            });

            bisect(centers, begin, middle, firstPart, nLeft, part);
            bisect(centers, middle, end, firstPart + nLeft, nParts - nLeft, part);
        }
    }

    std::vector<int> decomposeRCB(const std::vector<Vector<double>>& centers, int nParts)
    {
        if (nParts <= 0)
        {
            throw std::invalid_argument("decomposeRCB: number of parts must be positive");
        }
        if (static_cast<ULL>(nParts) > centers.size())
        {
            throw std::invalid_argument("decomposeRCB: more parts than cells");
        }

        std::vector<int> part(centers.size(), 0);
        std::vector<ULL> ids(centers.size());
        for (ULL i = 0; i < ids.size(); ++i)
        {
            ids[i] = i;
        }
        bisect(centers, ids.begin(), ids.end(), 0, nParts, part);
        return part;
    }
}
