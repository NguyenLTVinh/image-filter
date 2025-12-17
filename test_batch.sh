#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <target_program.c>"
    echo "Example: $0 filter_omp_v1.c"
    exit 1
fi

TEST_BATCHES_DIR="test_batches"
INPUT_IMAGES_DIR="${TEST_BATCHES_DIR}/indir"
BUILD_DIR="build"
BASE_SRC="filter-serial.c"
BASE_BIN="$BUILD_DIR/serial"
TARGET_SRC="$1"
TARGET_BIN="$BUILD_DIR/parallel"
OUTPUT_FILE="thread_scaling_results.txt"

THREAD_COUNTS=(2 4 8 16 32 64 128 256 288)

if [ ! -f "$TARGET_SRC" ]; then
    echo "❌ Error: Target source file '$TARGET_SRC' not found."
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "====================================================================" > "$OUTPUT_FILE"
echo "Thread Scaling Test Results" >> "$OUTPUT_FILE"
echo "Target Program: $TARGET_SRC" >> "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "====================================================================" >> "$OUTPUT_FILE"
echo >> "$OUTPUT_FILE"

echo "🔧 Compiling baseline ${BASE_SRC} ..."
set +e
gcc -O3 -fopenmp "$BASE_SRC" utilities.c -o "$BASE_BIN" -lm -lpthread
base_compile_exit=$?
set -e

if [ $base_compile_exit -ne 0 ] || [ ! -f "$BASE_BIN" ]; then
    echo "❌ Baseline compilation failed."
    echo "❌ Baseline compilation failed." >> "$OUTPUT_FILE"
    exit 1
fi
echo "✅ Baseline compiled."
echo "✅ Baseline compiled." >> "$OUTPUT_FILE"

echo "🔧 Compiling target ${TARGET_SRC} ..."
set +e
gcc -O3 -fopenmp "$TARGET_SRC" utilities.c -o "$TARGET_BIN" -lm -lpthread
target_compile_exit=$?
set -e

if [ $target_compile_exit -ne 0 ] || [ ! -f "$TARGET_BIN" ]; then
    echo "❌ Target compilation failed."
    echo "❌ Target compilation failed." >> "$OUTPUT_FILE"
    exit 1
else
    echo "✅ Target compiled."
    echo "✅ Target compiled." >> "$OUTPUT_FILE"
fi
echo
echo >> "$OUTPUT_FILE"

run_batch_test() {
    local test_dir="$1"
    local num_threads="$2"
    local test_id=$(basename "$test_dir")
    local input_kernel="${test_dir}/in.txt"
    local out_dir_serial="${BUILD_DIR}/${test_id}_serial"
    local out_dir_target="${BUILD_DIR}/${test_id}_target_${num_threads}"
    local log_serial="${BUILD_DIR}/${test_id}_serial.log"
    local log_target="${BUILD_DIR}/${test_id}_target_${num_threads}.log"

    if [ ! -f "$input_kernel" ]; then
        echo "⚠️  Missing kernel file for ${test_id}"
        echo "⚠️  Missing kernel file for ${test_id}" >> "$OUTPUT_FILE"
        return
    fi

    rm -rf "$out_dir_target"
    mkdir -p "$out_dir_target"
    
    if [ ! -d "$out_dir_serial" ]; then
        rm -rf "$out_dir_serial"
        mkdir -p "$out_dir_serial"
        
        echo "   Running baseline..."
        set +e
        "$BASE_BIN" "$INPUT_IMAGES_DIR" "$input_kernel" "$out_dir_serial" > "$log_serial" 2>&1
        local exit_serial=$?
        set -e
        
        if [ $exit_serial -ne 0 ]; then
            echo "💥 Baseline FAILED (exit code $exit_serial)"
            echo "💥 Baseline FAILED (exit code $exit_serial)" >> "$OUTPUT_FILE"
            cat "$log_serial"
            return
        fi
    fi
    
    local time_serial
    time_serial=$(grep "Total computation time:" "$log_serial" | awk '{print $4}')

    export OMP_NUM_THREADS=$num_threads
    echo "   Running target with $num_threads threads..."
    set +e
    "$TARGET_BIN" "$INPUT_IMAGES_DIR" "$input_kernel" "$out_dir_target" > "$log_target" 2>&1
    local exit_target=$?
    set -e

    if [ $exit_target -ne 0 ]; then
        echo "💥 Target FAILED (exit code $exit_target)"
        echo "💥 Target FAILED (exit code $exit_target)" >> "$OUTPUT_FILE"
        cat "$log_target"
        rm -rf "$out_dir_target"
        return
    fi

    local time_target
    time_target=$(grep "Total computation time:" "$log_target" | awk '{print $4}')

    echo "   Verifying correctness..."
    local correct=0
    if diff -r -q "$out_dir_serial" "$out_dir_target" > /dev/null; then
        echo "✅ OUTPUT MATCHES"
        echo "✅ OUTPUT MATCHES" >> "$OUTPUT_FILE"
        correct=1
    else
        echo "❌ OUTPUT MISMATCH"
        echo "❌ OUTPUT MISMATCH" >> "$OUTPUT_FILE"
    fi

    local speedup
    speedup=$(awk -v b="$time_serial" -v t="$time_target" 'BEGIN {if(t==0){print "0.000"}else{printf "%.3f", b/t}}')
    
    local time_serial_fmt
    time_serial_fmt=$(printf "%.3f" "$time_serial")
    
    local time_target_fmt
    time_target_fmt=$(printf "%.3f" "$time_target")
    
    echo "   Baseline time: ${time_serial_fmt}s"
    echo "   Target time:   ${time_target_fmt}s"
    echo "   Speedup:       ${speedup}x"
    echo
    
    echo "   Threads: $num_threads" >> "$OUTPUT_FILE"
    echo "   Baseline time: ${time_serial_fmt}s" >> "$OUTPUT_FILE"
    echo "   Target time:   ${time_target_fmt}s" >> "$OUTPUT_FILE"
    echo "   Speedup:       ${speedup}x" >> "$OUTPUT_FILE"
    echo >> "$OUTPUT_FILE"
    
    echo "   Cleaning up output directory..."
    rm -rf "$out_dir_target"
    
    if [ "$correct" -eq 1 ]; then
        return 0
    else
        return 1
    fi
}

cleanup_test() {
    local test_id="$1"
    local out_dir_serial="${BUILD_DIR}/${test_id}_serial"
    echo "   Cleaning up ${test_id} serial output..."
    rm -rf "$out_dir_serial"
}

echo "====================================================================="
echo "Running Batch Tests with Different Thread Counts"
echo "Target Program: $TARGET_SRC"
echo "====================================================================="
echo "====================================================================" >> "$OUTPUT_FILE"
echo "Running Batch Tests with Different Thread Counts" >> "$OUTPUT_FILE"
echo "====================================================================" >> "$OUTPUT_FILE"
echo >> "$OUTPUT_FILE"

for test_dir in $(ls -d ${TEST_BATCHES_DIR}/test_* | sort -V); do
    if [ -d "$test_dir" ]; then
        test_num=$(basename "$test_dir" | sed 's/test_//')
        test_id=$(basename "$test_dir")
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🧪 TEST $(basename "$test_dir")"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$OUTPUT_FILE"
        echo "🧪 TEST $(basename "$test_dir")" >> "$OUTPUT_FILE"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$OUTPUT_FILE"
        echo >> "$OUTPUT_FILE"
        
        for threads in "${THREAD_COUNTS[@]}"; do
            echo "🚀 Running with $threads threads..."
            run_batch_test "$test_dir" "$threads"
        done
        
        cleanup_test "$test_id"
        
        echo
        echo >> "$OUTPUT_FILE"
    fi
done

echo "All batch tests complete."
echo "Results saved to: $OUTPUT_FILE"
echo >> "$OUTPUT_FILE"
echo "All batch tests complete." >> "$OUTPUT_FILE"