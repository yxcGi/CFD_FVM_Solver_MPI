#!/usr/bin/env bash
#
# 并行结果一致性检查
#
# 用法（在项目根目录执行）：
#   tools/compare_parallel.sh <算例名> [进程数列表...]
#
# 例如：
#   tools/compare_parallel.sh simple_pitz_daily_steady 2 4 8
#
# 脚本先以单进程运行算例（单进程结果与原串行程序逐位一致），
# 再以给定的各进程数运行，逐字节比较所有输出的 .dat 文件和屏幕日志。
# 每次运行在 build/compare/<算例名>/np<N>/ 下进行。
#
# 可通过环境变量 MPIRUN 指定启动命令（默认 "mpirun"），例如：
#   MPIRUN="mpirun --oversubscribe" tools/compare_parallel.sh laplacian_scalar_dirichlet 2 3 4

set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <example> [np ...]" >&2
    exit 2
fi

example=$1
shift
nps=("$@")
if [ ${#nps[@]} -eq 0 ]; then
    nps=(2 4)
fi

root=$(cd "$(dirname "$0")/.." && pwd)
exe="$root/bin/$example"
mpirun_cmd=${MPIRUN:-mpirun}

if [ ! -x "$exe" ]; then
    echo "error: $exe not found, build the project first" >&2
    exit 2
fi

workdir="$root/build/compare/$example"
rm -rf "$workdir"

# 去掉与进程数有关的行（分解信息、计时）
filter_log() {
    grep -v -E "Mesh decomposed|^[0-9.]+s$" "$1"
}

run_case() {
    local np=$1
    local dir="$workdir/np$np"
    mkdir -p "$dir"
    ln -s "$root/tempFile" "$dir/tempFile"
    local start end
    start=$(date +%s.%N)
    (cd "$dir" && $mpirun_cmd -np "$np" "$exe" > log.txt 2>&1)
    local rc=$?
    end=$(date +%s.%N)
    printf "np=%-3s exit=%s  time=%.2fs\n" "$np" "$rc" "$(echo "$end - $start" | bc)"
    return $rc
}

run_case 1 || { echo "reference run failed, see $workdir/np1/log.txt"; exit 1; }

status=0
for np in "${nps[@]}"; do
    run_case "$np" || { echo "  run failed, see $workdir/np$np/log.txt"; status=1; continue; }
    for ref in "$workdir"/np1/*.dat; do
        [ -e "$ref" ] || continue
        name=$(basename "$ref")
        if cmp -s "$ref" "$workdir/np$np/$name"; then
            echo "  $name: identical"
        else
            echo "  $name: DIFFERENT"
            status=1
        fi
    done
    if diff <(filter_log "$workdir/np1/log.txt") <(filter_log "$workdir/np$np/log.txt") > /dev/null; then
        echo "  log: identical"
    else
        echo "  log: DIFFERENT"
        status=1
    fi
done

exit $status
