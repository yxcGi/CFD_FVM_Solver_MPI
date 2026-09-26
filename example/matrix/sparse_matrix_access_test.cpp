#include "Parallel/Parallel.h"
#include <exception>
#include <iostream>

#include "Geometry/Mesh.h"
#include "SparseMatrix.hpp"

using Scalar = double;

int main(int argc, char** argv)
{
    // MPI 并行环境（单进程运行时等价于串行程序）
    par::Environment env(argc, argv);

    try {
        Mesh mesh("tempFile/OpenFOAM_tutorials/pitzDailySteady/constant/polyMesh");

        SparseMatrix<Scalar> A_b(&mesh);

        // 矩阵索引为局部编号。并行时全局单元 0 所在的行只存在于拥有它的进程上，
        // 因此先把全局编号转换为局部编号（串行时两者相同）。
        const long long row = mesh.findLocalCell(0);
        if (row >= 0) {
            A_b.setValue(row, mesh.findLocalCell(0, true), 99);
            A_b.setValue(row, mesh.findLocalCell(1, true), 99);
        }

        for (int i = 0; i < 5; ++i) {
            const long long col = mesh.findLocalCell(i, true);
            const Scalar value = (row >= 0 && col >= 0) ? A_b.at(row, col) : 0.0;
            // 只有一个进程拥有该行，求和即得到该元素
            std::cout << par::allReduceSum(value) << " ";
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
