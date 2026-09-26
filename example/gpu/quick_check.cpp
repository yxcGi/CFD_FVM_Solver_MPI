#include "Parallel/Parallel.h"
#include <iostream>

#include "Field.hpp"
#include "Geometry/Mesh.h"
#include "SIMPLE.hpp"
#include "Vector.hpp"

using Scalar = double;

int main(int argc, char** argv)
{
    // MPI 并行环境（单进程运行时等价于串行程序）
    par::Environment env(argc, argv);

    // GPU 快速一致性检查：三维方腔，只算 5 步（tools/verify_gpu.sh 使用）
    Mesh mesh("tempFile/OpenFOAM_tutorials/cavity3D_4/constant/polyMesh");

    Field<Vector<Scalar>> U("U", &mesh);
    Field<Scalar> p("p", &mesh);

    U.setValue(Vector<Scalar>(0, 0, 0));
    p.setValue(0.0);

    // U.setBoundaryCondition("movingWall", 1, 0, Vector<Scalar>(4, 0, 0)); // Re = 400
    // U.setBoundaryCondition("movingWall", 1, 0, Vector<Scalar>(10, 0, 0)); // Re = 1000
    U.setBoundaryCondition("movingWall", 1, 0, Vector<Scalar>(16, 0, 0)); // Re = 1600
    U.setBoundaryCondition("fixedWalls", 1, 0, Vector<Scalar>(0, 0, 0));

    p.setBoundaryCondition("movingWall", 0, 1, 0.0);
    p.setBoundaryCondition("fixedWalls", 0, 1, 0.0);

    FaceField<Scalar> rho("rho", &mesh);
    rho.setValue(1.0);

    FaceField<Scalar> nu("nu", &mesh);
    nu.setValue(0.01);

    algorithm::simple::SIMPLE::Options options;
    options.maxOuterIterations = 5;
    options.alphaU = 0.7;
    options.alphaP = 0.3;
    options.convergenceTolerance = 1e-8;
    options.nNonOrthogonalCorrectors = 3;
    options.divScheme = fvm::DivType::MUSCL;
    options.useParallel = true;

    algorithm::simple::SIMPLE solver(U, p, rho, nu, options);
    solver.solve();

    U.writeToFile("U_quick_check.dat");
    p.writeToFile("p_quick_check.dat");

    return 0;
}
