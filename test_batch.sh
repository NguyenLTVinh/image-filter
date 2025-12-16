#!/bin/bash

set -e

# Configuration
TEST_BATCHES_DIR="test_batches"
INPUT_IMAGES_DIR="${TEST_BATCHES_DIR}/indir"
BUILD_DIR="build"

BASE_SRC="filter-serial.c"
BASE_BIN="$BUILD_DIR/serial"

TARGET_SRC="${1:-filter-batch-optimized.c}"
TARGET_BIN="$BUILD_DIR/parallel"

# Create build directory
mkdir -p "$BUILD_DIR"

# Compilation
echo "🔧 Compiling baseline ${BASE_SRC} ..."
set +e
gcc -O3 -fopenmp "$BASE_SRC" utilities.c -o "$BASE_BIN" -lm -lpthread
base_compile_exit=$?
set -e

if [ $base_compile_exit -ne 0 ] || [ ! -f "$BASE_BIN" ]; then
    echo "❌ Baseline compilation failed."
    exit 1
fi
echo "✅ Baseline compiled."

echo "🔧 Compiling target ${TARGET_SRC} ..."
set +e
gcc -O3 -fopenmp "$TARGET_SRC" utilities.c -o "$TARGET_BIN" -lm -lpthread
target_compile_exit=$?
set -e
if [ $target_compile_exit -ne 0 ] || [ ! -f "$TARGET_BIN" ]; then
    echo "❌ Target compilation failed."
    exit 1
else
    echo "✅ Target compiled."
fi

echo

# Function to run comparison
run_batch_test() {
    local test_dir="$1"
    local test_id=$(basename "$test_dir")
    local input_kernel="${test_dir}/in.txt"
    
    local out_dir_serial="${BUILD_DIR}/${test_id}_serial"
    local out_dir_target="${BUILD_DIR}/${test_id}_target"
    
    local log_serial="${BUILD_DIR}/${test_id}_serial.log"
    local log_target="${BUILD_DIR}/${test_id}_target.log"

    if [ ! -f "$input_kernel" ]; then
        echo "⚠️  Missing kernel file for ${test_id}"
        return
    fi

    # Clean previous outputs
    rm -rf "$out_dir_serial" "$out_dir_target"
    mkdir -p "$out_dir_serial" "$out_dir_target"

    echo "🚀 Running ${test_id}..."

    # Run Baseline
    # echo "   Running baseline..."
    set +e
    "$BASE_BIN" "$INPUT_IMAGES_DIR" "$input_kernel" "$out_dir_serial" > "$log_serial" 2>&1
    local exit_serial=$?
    set -e
    
    if [ $exit_serial -ne 0 ]; then
        echo "💥 Baseline FAILED (exit code $exit_serial)"
        cat "$log_serial"
        return
    fi
    
    local time_serial
    time_serial=$(grep "Total computation time:" "$log_serial" | awk '{print $4}')

    # Run Target
    # echo "   Running target..."
    set +e
    "$TARGET_BIN" "$INPUT_IMAGES_DIR" "$input_kernel" "$out_dir_target" > "$log_target" 2>&1
    local exit_target=$?
    set -e

    if [ $exit_target -ne 0 ]; then
        echo "💥 Target FAILED (exit code $exit_target)"
        cat "$log_target"
        return
    fi

    local time_target
    time_target=$(grep "Total computation time:" "$log_target" | awk '{print $4}')

    # Correctness Check
    echo "   Verifying correctness..."
    if diff -r -q "$out_dir_serial" "$out_dir_target" > /dev/null; then
        echo "✅ OUTPUT MATCHES"
    else
        echo "❌ OUTPUT MISMATCH"
        # Optional: Show diff details
        # diff -r "$out_dir_serial" "$out_dir_target" | head -n 20
        return
    fi

    # Speedup Calculation
    local speedup
    speedup=$(awk -v b="$time_serial" -v t="$time_target" 'BEGIN {if(t==0){print 0}else{print b/t}}')
    
    echo "   Baseline time: ${time_serial}s"
    echo "   Target time:   ${time_target}s"
    echo "   Speedup:       ${speedup}x"
    echo
}

# Main Loop
echo "====================================================================="
echo "Running Batch Tests"
echo "====================================================================="

# Find all test directories and sort them
for test_dir in $(ls -d ${TEST_BATCHES_DIR}/test_* | sort -V); do
    if [ -d "$test_dir" ]; then
        run_batch_test "$test_dir"
    fi
done

echo "All batch tests complete."
