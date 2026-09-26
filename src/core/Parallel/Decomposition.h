#ifndef DECOMPOSITION_H_
#define DECOMPOSITION_H_

#include <vector>
#include "Vector.hpp"

namespace par
{
    /**
     * @brief 递归坐标二分（Recursive Coordinate Bisection, RCB）区域分解。
     *
     * 按单元中心坐标递归地沿包围盒最长方向切分，使各子区域单元数按
     * 分区数成比例（支持任意分区数，不要求 2 的幂）。
     * 结果确定：同样的网格与分区数总是得到同样的分解。
     *
     * @param centers 全局单元中心
     * @param nParts 分区数
     * @return 每个单元所属的分区号
     */
    std::vector<int> decomposeRCB(const std::vector<Vector<double>>& centers, int nParts);
}

#endif // DECOMPOSITION_H_
