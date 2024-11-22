#!/bin/bash

# Output file for results
results_file="results.txt"

# Clear or create the results file
> "$results_file"

# Loop through files in the output directory
for file in output/*; do
    echo "$file" >> "$results_file"
    grep "\"name" "$file" >> "$results_file"
    grep "\"model" "$file" >> "$results_file"
    grep "\"dataset" "$file" >> "$results_file"
    grep "\"learning_rate" "$file" >> "$results_file"
    grep ^Parameters "$file" >> "$results_file"
    grep MAE "$file" | tail -n 1 >> "$results_file"
    echo "=================" >> "$results_file"
done

# Feed results.txt into the Python script
python csv_ify_results.py < "$results_file"

echo "CSV file has been created from results.txt"
