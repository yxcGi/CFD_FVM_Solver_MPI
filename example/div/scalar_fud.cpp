#include "Field.hpp"
#include "Geometry/Mesh.h"
// #include "Laplacian.hpp"
#include "Solver.hpp"
#include "SparseMatrix.hpp"
#include "Div.hpp"


int main() {
    Mesh mesh("/home/gczl/Desktop/yxc/code/CFD_FVM_Solver/tempFile/OpenFOAM_tutorials/cavity/constant/polyMesh");
    Field<Scalar> T("T", &mesh);

    T.setValue(0);

    T.setBoundaryCondition("movingWall", 0, 1, 0);
    T.setBoundaryCondition("leftWalls", 1, 0, 100);
    T.setBoundaryCondition("bottomWalls", 1, 0, 0);
    T.setBoundaryCondition("rightWalls", 0, 1, 0);
    T.cellToFace();

    FaceField<Scalar> rho("rho", &mesh);
    rho.setValue(1);
    FaceField<Vector<Scalar>> Uf("Uf", &mesh);
    Uf.setValue(Vector<Scalar>(0.1, 0.1, 0));

    SparseMatrix<Scalar> A(&mesh);
    for (int i = 0; i < 100; ++i)
    {
        fvm::Div(A, rho, T, Uf, fvm::DivType::FUD);

        Solver<Scalar> solver(A, Solver<Scalar>::Method::Jacobi, 100000);
        solver.init(T.getCellField().getData());
        solver.setTolerance(1e-6);
        solver.solve();
        A.clear();
        T.cellToFace();
    }

    T.writeToFile("T.dat");
}
