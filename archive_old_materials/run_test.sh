#!/bin/bash
output_file="raw_result.csv"
summary_file="new_data.csv"
rm -f "$output_file" "$summary_file"

test_params=("Ops" "QKV" "RoPE" "Cache" "QK_T" "AttnV" "UpGate" "Down" "SiLU")

echo "timestamp,status,module,level,time" > "$output_file"

for test in "${test_params[@]}"; do
    for level in {1..13}; do
        echo "Testing: -test $test -level $level -logN=16 -hidDim=4096 -expDim=16384 -seqLen=512 -parallel=true 2>&1"
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        output=$(go run . -test=$test -level=$level  2>&1)
        exit_status=$?
        sanitized_output=$(echo "$output" | grep -oP 'Consumed\s+\K[0-9.]+(?=\s+seconds)' | tr '\n' ' ' | sed 's/[[:space:]]*$//' || echo "N/A")
        echo "$timestamp,$exit_status,$test,$level,$sanitized_output" >> "$output_file"
        echo "----------------------------------"
    done
done

# test="Softmax"
# for level in {1..16}; do
#     echo "Testing: -test $test -level $level"
#     timestamp=$(date '+%Y-%m-%d %H:%M:%S')
#     output=$(go run . -test=$test -level=16 -btpLevel=$level -logN=16 -hidDim=4096 -expDim=16384 -seqLen=512 2>&1)
#     exit_status=$?
#     sanitized_output=$(echo "$output" | grep -oP 'Consumed\s+\K[0-9.]+(?=\s+seconds)' | tr '\n' ' ' | sed 's/[[:space:]]*$//' || echo "N/A")
#     echo "$timestamp,$exit_status,Softmax,$level,$sanitized_output" >> "$output_file"
#     echo "----------------------------------"
# done

# test="Norm"
# for level in {1..16}; do
#     echo "Testing: -test $test -level $level"
#     timestamp=$(date '+%Y-%m-%d %H:%M:%S')
#     output=$(go run . -test=$test -level=$level -logN=16 -hidDim=4096 -expDim=16384 -seqLen=512 2>&1)
#     exit_status=$?
#     sanitized_output=$(echo "$output" | grep -oP 'Consumed\s+\K[0-9.]+(?=\s+seconds)' | tr '\n' ' ' | sed 's/[[:space:]]*$//' || echo "N/A")
#     echo "$timestamp,$exit_status,SqrtNt,$level,$sanitized_output" >> "$output_file"
    
#     output=$(go run . -test=$test -level=16 -btpLevel=$level -logN=16 -hidDim=4096 -expDim=16384 -seqLen=512 2>&1)
#     exit_status=$?
#     sanitized_output=$(echo "$output" | grep -oP 'Consumed\s+\K[0-9.]+(?=\s+seconds)' | tr '\n' ' ' | sed 's/[[:space:]]*$//' || echo "N/A")
#     echo "$timestamp,$exit_status,SqrtGold,$level,$sanitized_output" >> "$output_file"
#     echo "----------------------------------"
# done

echo "Tests completed! Results saved to $output_file."

# test_params=("QKV" "RoPE" "Cache" "QK_T" "AttnV" "UpGate" "Down" "SiLU" "Softmax" "CtMult" "SqrtNt" "SqrtGold")

# {
#     echo -n "module"
#     for i in {1..16}; do
#         printf "\t%.2f" "$i"
#     done
#     echo
# } > "$summary_file"

# for test in "${test_params[@]}"; do
#     times=$(awk -F',' -v mod="$test" '
#         $3 ~ mod {
#             gsub(/"/, "", $2)
#             gsub(/"/, "", $5)
#             if ($2 == "0") {
#                 n = split($5, arr, /[ \t]+/)
#                 if (mod == "Softmax" || mod == "SqrtGold") {
#                     print arr[n]
#                 } else {
#                     print arr[1]
#                 }
#             } else {
#                 print "0.00"
#             }
#         }
#     ' "$output_file")

#     echo -n "$test" >> "$summary_file"
#     i=1
#     while read -r t; do
#         if [[ -z "$t" || "$t" == "N/A" ]]; then
#             printf "\t0.00" >> "$summary_file"
#         else
#             printf "\t%.2f" "$t" >> "$summary_file"
#         fi
#         ((i++))
#     done <<< "$times"

#     echo >> "$summary_file"
# done

# echo "Summary table written to $summary_file."

# python3 bootstrap.py --prune=1 --file=new_data.csv
