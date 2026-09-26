#!/usr/bin/env bash
#
# GPU 快速一致性验证：终端只输出 true 或 false
#
# 用法（在项目根目录、有 NVIDIA 显卡和 CUDA 的 Linux 机器上执行）：
#   tools/verify_gpu.sh
#
# 流程：编译 gpu_quick_check（三维方腔，只算 5 步）→ 用 CPU 跑一次（CFD_USE_GPU=0）
#       → 用 GPU 跑一次 →（有 MPI 时）用 2 个进程在 GPU 上再跑一次 → 逐字节比较输出文件。
# 全部一致输出 true；否则（或没有用上 GPU、编译/运行失败）输出 false。
# 详细过程写在 build/verify_gpu/verify.log，出现 false 时可查看原因。

set -u

root=$(cd "$(dirname "$0")/.." && pwd)
build="$root/build/verify_gpu"
log="$build/verify.log"
mkdir -p "$build"
: > "$log"

fail() {
    echo "FAILED: $1" >> "$log"
    echo false
    exit 1
}

# 1. 编译（只编译需要的目标）
cmake_args=(-S "$root" -B "$build/cmake" -DCMAKE_BUILD_TYPE=Release -DCFD_USE_CUDA=ON)
if command -v nvidia-smi > /dev/null 2>&1; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .')
    if [[ "$cap" =~ ^[0-9]+$ ]]; then
        cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$cap")
    fi
fi
cmake "${cmake_args[@]}" >> "$log" 2>&1 || fail "cmake configure"
grep -q "CUDA found" "$log" || fail "nvcc not found (CUDA toolkit missing)"
cmake --build "$build/cmake" --target gpu_quick_check -j >> "$log" 2>&1 || fail "build"
exe="$root/bin/gpu_quick_check"

# 2. 运行：run <目录名> <CFD_USE_GPU> <进程数（0 表示不用 mpirun）>
run() {
    local dir="$build/$1"
    rm -rf -- "$dir"
    mkdir -p "$dir"
    ln -s "$root/tempFile" "$dir/tempFile"
    echo "== run $1" >> "$log"
    if [ "$3" = 0 ]; then
        (cd "$dir" && CFD_USE_GPU=$2 "$exe" > out.txt 2>&1)
    else
        (cd "$dir" && CFD_USE_GPU=$2 ${MPIRUN:-mpirun} -np "$3" "$exe" > out.txt 2>&1)
    fi
    local rc=$?
    grep -E "^\[GPU\]" "$dir/out.txt" >> "$log"
    [ $rc -eq 0 ] || fail "run $1 (see $dir/out.txt)"
}

same() {
    for f in U_quick_check.dat p_quick_check.dat; do
        cmp -s "$build/cpu/$f" "$build/$1/$f" || fail "$f differs between cpu and $1"
    done
    echo "$1: identical" >> "$log"
}

run cpu 0 0
run gpu 1 0
grep -q -E "^\[GPU\]" "$build/gpu/out.txt" || fail "no GPU was used (no CUDA device found)"
same gpu

if grep -q "MPI found" "$log" && command -v "${MPIRUN:-mpirun}" > /dev/null 2>&1; then
    run gpu_np2 1 2
    same gpu_np2
fi

echo true
