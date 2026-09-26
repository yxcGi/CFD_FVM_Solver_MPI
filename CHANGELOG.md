# Changelog

## v0.3.0 - GPU (CUDA) parallel linear solver

### Added

- CUDA backend for the Jacobi linear solver (`src/core/Gpu/`): MPI + GPU, one GPU
  per process (device = node-local rank % GPUs per node). The matrix is uploaded
  once per solve; iterations stay on the GPU and only halo values go through MPI.
- Results are bitwise identical to the CPU version (same per-row summation order,
  shared row kernels in `JacobiRow.h`, `-fmad=false`, IEEE division/sqrt).
- CMake option `CFD_USE_CUDA` (ON by default; CPU-only build when nvcc is not found).
  Runtime falls back to the CPU when no GPU is present or `CFD_USE_GPU=0`.
- CMake option `CFD_GPU_EMULATE` (testing only): host emulation of the GPU backend.
- `tools/compare_gpu.sh` to compare GPU runs with the CPU reference.
- `par::HaloExchange::startPacked()/finishPacked()` and `par::nodeLocalRank()`.

## v0.2.0 - MPI parallel solver

### Added

- MPI domain-decomposition parallelism: every example runs with `mpirun -np N`.
- Built-in recursive coordinate bisection (RCB) decomposition; no external partitioner needed.
- `par::` communication layer: MPI environment, halo exchange (non-blocking,
  overlapped with computation), ordered global reductions, gather to rank 0.
- Results and output files are bitwise identical to the serial program for any
  number of processes.
- `tools/compare_parallel.sh` to check outputs across process counts.
- CMake option `CFD_USE_MPI` (ON by default; falls back to a single-process build
  when MPI is not found).

### Changed

- Example `main()` functions construct `par::Environment` first.
- `Mesh::getCellNumber()` returns the number of cells owned by the process;
  cell field storage uses `Mesh::getLocalCellNumber()` (owned + ghost cells).
- `SIMPLE::Options::pressureReferenceCell` is a global cell index.
- `example/div/scalar_fud.cpp` uses the relative mesh path like the other examples.
- `example/matrix/sparse_matrix_access_test.cpp` maps global cell indices to local ones.
- Compile with `-ffp-contract=off` so floating-point results do not depend on FMA contraction.

## v0.1.0 - First runnable FVM solver release

### Added

- OpenFOAM-style `polyMesh` reading.
- Mesh topology and geometry calculation.
- Scalar, vector, and tensor field abstractions.
- Finite volume discretization operators:
  - `fvm::Laplacian`
  - `fvm::Div`
  - `fvm::Source`
- CSR sparse matrix assembly.
- Jacobi linear solver with scalar and vector unknown support.
- SIMPLE algorithm for steady incompressible flow.
- Rhie-Chow interpolation for collocated grids.
- Non-orthogonal correction for diffusion terms.
- Example cases:
  - 2D lid-driven cavity
  - 2D backward-facing step
  - 3D lid-driven cavity
  - Laplacian tests
  - convection tests
  - source term tests
  - sparse matrix tests

### Known limitations

- Mainly supports steady problems.
- Linear solver is still primarily Jacobi-based.
- Turbulence models are not implemented.
- Boundary conditions are still being extended.
- Parallelism is currently shared-memory based only.
