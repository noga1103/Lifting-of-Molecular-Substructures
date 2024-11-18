for file in output_bak/*; do
    grep model $file
    grep learning_rate $file
    grep Parameters $file
    grep MAE $file | tail -n 1
    echo "================="
done
