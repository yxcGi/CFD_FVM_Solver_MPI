#!/usr/bin/env bash
#
# GPU 结果一致性检查
#
# 用法（在项目根目录执行，需用 -DCFD_USE_CUDA=ON 编译且机器上有 GPU）：
#   tools/compare_gpu.sh <算例名> [进程数列表...]
#
# 例如：
#   tools/compare_gpu.sh simple_cavity3d_unstructured 1 2
#
# 脚本先用 CFD_USE_GPU=0 以单进程在 CPU 上运行算例作为参考，
# 再对给定的各进程数在 GPU 上运行（每个进程一块 GPU，GPU 少于进程数时共用），
# 逐字节比较所有输出的 .dat 文件和屏幕日志。
# 每次运行在 build/compare_gpu/<算例名>/ 下进行。
#
# 可通过环境变量 MPIRUN 指定启动命令（默认 "mpirun"）。

set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <example> [np ...]" >&2
    exit 2
fi

example=$1
shift
nps=("$@")
if [ ${#nps[@]} -eq 0 ]; then
    nps=(1)
fi

root=$(cd "$(dirname "$0")/.." && pwd)
exe="$root/bin/$example"
mpirun_cmd=${MPIRUN:-mpirun}

if [ ! -x "$exe" ]; then
    echo "error: $exe not found, build the project first" >&2
    exit 2
fi

workdir="$root/build/compare_gpu/$example"
rm -rf "$workdir"

# 去掉与进程数 / 设备有关的行（分解信息、计时、GPU 设备信息）
filter_log() {
    grep -v -E "Mesh decomposed|^[0-9.]+s$|^\[GPU\]" "$1"
}

# run_case <目录名> <进程数> <CFD_USE_GPU>
run_case() {
    local name=$1 np=$2 useGpu=$3
    local dir="$workdir/$name"
    mkdir -p "$dir"
    ln -s "$root/tempFile" "$dir/tempFile"
    local start end
    start=$(date +%s.%N)
    # 本机运行时 mpirun 会把环境变量传给各进程（多节点时 OpenMPI 需加 -x CFD_USE_GPU）
    (cd "$dir" && CFD_USE_GPU=$useGpu $mpirun_cmd -np "$np" "$exe" > log.txt 2>&1)
    local rc=$?
    end=$(date +%s.%N)
    printf "%-8s np=%-3s exit=%s  time=%.2fs\n" "$name" "$np" "$rc" "$(echo "$end - $start" | bc)"
    grep -E "^\[GPU\]" "$dir/log.txt" | sed 's/^/    /'
    return $rc
}

run_case cpu 1 0 || { echo "CPU reference run failed, see $workdir/cpu/log.txt"; exit 1; }

status=0
for np in "${nps[@]}"; do
    name="gpu_np$np"
    run_case "$name" "$np" 1 || { echo "  run failed, see $workdir/$name/log.txt"; status=1; continue; }
    if ! grep -q -E "^\[GPU\]" "$workdir/$name/log.txt"; then
        echo "  warning: no GPU was used (built without CUDA or no device found)"
    fi
    for ref in "$workdir"/cpu/*.dat; do
        [ -e "$ref" ] || continue
        file=$(basename "$ref")
        if cmp -s "$ref" "$workdir/$name/$file"; then
            echo "  $file: identical"
        else
            echo "  $file: DIFFERENT"
            status=1
        fi
    done
    if diff <(filter_log "$workdir/cpu/log.txt") <(filter_log "$workdir/$name/log.txt") > /dev/null; then
        echo "  log: identical"
    else
        echo "  log: DIFFERENT"
        status=1
    fi
done

exit $status
