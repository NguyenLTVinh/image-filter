#!/bin/bash

set -e

TEST_DIR="tests"
BUILD_DIR="build"
BASE_SRC="filter-serial.c"
BASE_BIN="$BUILD_DIR/serial"
TARGET_SRC="$1"
TARGET_BIN="$BUILD_DIR/parallel"

if [ -z "$TARGET_SRC" ]; then
    echo "⚠️  No target source provided."
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "🔧 Compiling baseline ${BASE_SRC} ..."
set +e
nvcc -O3 -arch=sm_75 "$BASE_SRC" -o "$BASE_BIN"
base_compile_exit=$?
set -e

if [ $base_compile_exit -ne 0 ] || [ ! -f "$BASE_BIN" ]; then
    echo "❌ Baseline compilation failed."
    exit 1
fi
echo "✅ Baseline compiled."

echo "🔧 Compiling target ${TARGET_SRC} ..."
set +e
nvcc -O3 -arch=sm_75 "$TARGET_SRC" -o "$TARGET_BIN"
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

    local timeout_val=60
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

echo "All tests complete."