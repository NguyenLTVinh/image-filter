#!/bin/bash

set -e

TEST_DIR="tests"
BUILD_DIR="build"
BATCH_TEST_DIR="${TEST_DIR}/test_25"
BATCH_IMAGES_DIR="${BATCH_TEST_DIR}/images"
BATCH_KERNEL_FILE="${BATCH_TEST_DIR}/in.txt"

BASE_SRC="variants/variant1-serial.c"
BASE_BIN="$BUILD_DIR/serial"

TARGET_SRC="$1"
TARGET_BIN="$BUILD_DIR/parallel"

if [ -z "$TARGET_SRC" ]; then
    echo "⚠️  No target source provided."
    exit 1
fi

IS_BATCH=0
if [[ "$TARGET_SRC" == *batch* ]]; then
    IS_BATCH=1
fi

mkdir -p "$BUILD_DIR"

echo "🔧 Compiling baseline ${BASE_SRC} ..."
set +e
nvcc -O3 -arch=sm_75 -Xcompiler -Wno-unused-result "$BASE_SRC" utilities.c -o "$BASE_BIN" --fmad=false
base_compile_exit=$?
set -e

if [ $base_compile_exit -ne 0 ] || [ ! -f "$BASE_BIN" ]; then
    echo "❌ Baseline compilation failed."
    exit 1
fi
echo "✅ Baseline compiled."

echo "🔧 Compiling target ${TARGET_SRC} ..."
set +e
nvcc -O3 -arch=sm_75 -Xcompiler -Wno-unused-result "$TARGET_SRC" utilities.c -o "$TARGET_BIN" --fmad=false
target_compile_exit=$?
set -e
if [ $target_compile_exit -ne 0 ] || [ ! -f "$TARGET_BIN" ]; then
    echo "❌ Target compilation failed."
    exit 1
else
    echo "✅ Target compiled."
fi

echo

run_correctness() {
    local test_id=$1
    local input_image="$TEST_DIR/test_${test_id}/in.ppm"
    local input_kernel="$TEST_DIR/test_${test_id}/in.txt"
    local ref_output="$TEST_DIR/test_${test_id}/out.ppm"
    local tmp_output="$BUILD_DIR/out_test${test_id}.ppm"
    local log_file="$BUILD_DIR/log_test${test_id}.txt"

    if [ ! -f "$input_image" ] || [ ! -f "$input_kernel" ] || [ ! -f "$ref_output" ]; then
        echo "⚠️  Missing input or reference output for test_${test_id}"
        return
    fi

    local timeout_val=30

    echo "🚀 Running test_${test_id} (timeout=${timeout_val}s) ..."

    set +e
    timeout "${timeout_val}s" "$TARGET_BIN" "$input_image" "$input_kernel" "$tmp_output" >"$log_file" 2>&1
    local exit_code=$?
    set -e

    local elapsed
    elapsed=$(grep "Total computation time:" "$log_file" | awk '{print $4}' || echo "nan")

    if [ $exit_code -eq 124 ]; then
        echo "⏱️  test_${test_id} TIMEOUT after ${timeout_val}s"
        rm -f "$tmp_output" "$log_file"
        return
    elif [ $exit_code -ne 0 ]; then
        echo "💥 test_${test_id} RUNTIME ERROR (exit code $exit_code)"
        echo "🔍 See $log_file for details"
        echo
        rm -f "$tmp_output" "$log_file"
        return
    fi

    echo "🔍 Comparing output with reference..."
    if diff -q "$tmp_output" "$ref_output" > /dev/null; then
        echo "✅ UNIT TEST PASSED: test_${test_id} (t=${elapsed}s)"
    else
        echo "❌ UNIT TEST FAILED: test_${test_id}"
    fi

    rm -f "$tmp_output" "$log_file"
    echo
}

run_speedup() {
    local test_id=$1
    local input_image="$TEST_DIR/test_${test_id}/in.ppm"
    local input_kernel="$TEST_DIR/test_${test_id}/in.txt"
    local ref_output="$TEST_DIR/test_${test_id}/out.ppm"
    local tmp_output_target="$BUILD_DIR/out_test${test_id}_target.ppm"
    local tmp_output_base="$BUILD_DIR/out_test${test_id}_baseline.ppm"
    local log_target="$BUILD_DIR/log_test${test_id}_target.txt"
    local log_base="$BUILD_DIR/log_test${test_id}_baseline.txt"

    if [ ! -f "$input_image" ] || [ ! -f "$input_kernel" ] || [ ! -f "$ref_output" ]; then
        echo "⚠️  Missing input or reference output for test_${test_id}"
        return
    fi

    local timeout_val=120
    local min_speedup=${MIN_SPEEDUP:-1.05}

    echo "🚀 Running speedup test_${test_id} (timeout=${timeout_val}s, min speedup=${min_speedup}x) ..."

    set +e
    timeout "${timeout_val}s" "$TARGET_BIN" "$input_image" "$input_kernel" "$tmp_output_target" >"$log_target" 2>&1
    local exit_target=$?
    timeout "${timeout_val}s" "$BASE_BIN" "$input_image" "$input_kernel" "$tmp_output_base" >"$log_base" 2>&1
    local exit_base=$?
    set -e

    local target_elapsed base_elapsed
    target_elapsed=$(grep "Total computation time:" "$log_target" | awk '{print $4}' || echo "nan")
    base_elapsed=$(grep "Total computation time:" "$log_base" | awk '{print $4}' || echo "nan")

    if [ $exit_target -eq 124 ] || [ $exit_base -eq 124 ]; then
        echo "⏱️  test_${test_id} TIMEOUT after ${timeout_val}s"
        rm -f "$tmp_output_target" "$tmp_output_base" "$log_target" "$log_base"
        return
    fi

    if [ $exit_target -ne 0 ]; then
        echo "💥 test_${test_id} RUNTIME ERROR (target exit $exit_target)"
        echo "🔍 See $log_target for details"
        rm -f "$tmp_output_target" "$tmp_output_base" "$log_target" "$log_base"
        return
    fi

    if [ $exit_base -ne 0 ]; then
        echo "💥 test_${test_id} RUNTIME ERROR (baseline exit $exit_base)"
        echo "🔍 See $log_base for details"
        rm -f "$tmp_output_target" "$tmp_output_base" "$log_target" "$log_base"
        return
    fi

    if ! diff -q "$tmp_output_target" "$ref_output" > /dev/null; then
        echo "❌ UNIT TEST FAILED: test_${test_id}"
        echo "❌ SPEEDUP FAIL: incorrect output"
        rm -f "$tmp_output_target" "$tmp_output_base" "$log_target" "$log_base"
        echo
        return
    fi

    local speedup
    speedup=$(awk -v b="$base_elapsed" -v t="$target_elapsed" 'BEGIN{if(t==0||t!=t||b!=b){print 0}else{print b/t}}')
    if awk -v s="$speedup" -v m="$min_speedup" 'BEGIN{exit (s < m ? 0 : 1)}'; then
        echo "❌ SPEEDUP FAIL: ${target_elapsed}s ${speedup}x (threshold=${min_speedup}x)"
    else
        echo "✅ SPEEDUP PASS: ${target_elapsed}s ${speedup}x (threshold=${min_speedup}x)"
    fi

    rm -f "$tmp_output_target" "$tmp_output_base" "$log_target" "$log_base"
    echo
}

prepare_subset_dir() {
    local size=$1
    local subset_dir="$BUILD_DIR/batch_subset_${size}"
    rm -rf "$subset_dir"
    mkdir -p "$subset_dir"

    if [ ! -d "$BATCH_IMAGES_DIR" ]; then
        echo "⚠️  Missing input image directory: $BATCH_IMAGES_DIR. Skipping batch size ${size}."
        rm -rf "$subset_dir"
        return 0
    fi

    mapfile -t imgs < <(find "$BATCH_IMAGES_DIR" -maxdepth 1 -type f -name '*.ppm' | sort | head -n "$size")

    local count=${#imgs[@]}
    if [ $count -lt $size ]; then
        echo "⚠️  Not enough images for batch size ${size} (found $count). Skipping."
        rm -rf "$subset_dir"
        return 0
    fi

    for f in "${imgs[@]}"; do
        cp "$f" "$subset_dir/" 2>/dev/null || true
    done

    echo "$subset_dir"
}

run_batch_dir_test() {
    local size=$1
    local kernel_file="$BATCH_KERNEL_FILE"
    if [ ! -f "$kernel_file" ]; then
        echo "⚠️  Missing batch kernel file: $kernel_file. Skipping batch size ${size}."
        return
    fi
    local subset_dir
    subset_dir=$(prepare_subset_dir "$size") || return
    if [ -z "$subset_dir" ] || [ ! -d "$subset_dir" ]; then
        return
    fi

    local out_dir_serial="${BUILD_DIR}/batch${size}_serial"
    local out_dir_target="${BUILD_DIR}/batch${size}_target"
    local log_serial="${BUILD_DIR}/batch${size}_serial.log"
    local log_target="${BUILD_DIR}/batch${size}_target.log"

    rm -rf "$out_dir_serial" "$out_dir_target"
    mkdir -p "$out_dir_serial" "$out_dir_target"

    echo "🚀 Running batch size ${size}..."

    set +e
    "$BASE_BIN" "$subset_dir" "$kernel_file" "$out_dir_serial" >"$log_serial" 2>&1
    local exit_base=$?
    "$TARGET_BIN" "$subset_dir" "$kernel_file" "$out_dir_target" >"$log_target" 2>&1
    local exit_target=$?
    set -e

    if [ $exit_base -ne 0 ]; then
        echo "💥 Baseline FAILED (exit $exit_base)"
        cat "$log_serial"
        return
    fi
    if [ $exit_target -ne 0 ]; then
        echo "💥 Target FAILED (exit $exit_target)"
        cat "$log_target"
        return
    fi

    local base_t target_t
    base_t=$(grep "Total computation time:" "$log_serial" | awk '{print $4}' || echo "nan")
    target_t=$(grep "Total computation time:" "$log_target" | awk '{print $4}' || echo "nan")

    echo "   Verifying correctness..."
    if diff -r -q "$out_dir_serial" "$out_dir_target" > /dev/null; then
        echo "✅ OUTPUT MATCHES"
    else
        echo "❌ OUTPUT MISMATCH"
        return
    fi

    local speedup
    speedup=$(awk -v b="$base_t" -v t="$target_t" 'BEGIN{if(t==0||t!=t||b!=b){print 0}else{print b/t}}')
    echo "   Baseline time: ${base_t}s"
    echo "   Target time:   ${target_t}s"
    echo "   Speedup:       ${speedup}x"
    echo
}

echo "====================================================================="
echo "Running correctness tests (0-14)"
echo "====================================================================="
for i in {0..14}; do
    test_dir="$TEST_DIR/test_$i"
    if [ -d "$test_dir" ]; then
        run_correctness "$i"
    else
        echo "⚠️  Skipping missing $test_dir"
    fi
done

echo "====================================================================="
echo "Running speedup tests (15-24)"
echo "====================================================================="
for i in {15..24}; do
    test_dir="$TEST_DIR/test_$i"
    if [ -d "$test_dir" ]; then
        run_speedup "$i"
    else
        echo "⚠️  Skipping missing $test_dir"
    fi
done

if [ $IS_BATCH -eq 1 ]; then
    echo "====================================================================="
    echo "Running batch-size tests (4, 8, 16, 32, 64, 128, 256)"
    echo "====================================================================="
    for sz in 4 8 16 32 64 128 256; do
        run_batch_dir_test "$sz"
    done
fi

echo "All tests complete."