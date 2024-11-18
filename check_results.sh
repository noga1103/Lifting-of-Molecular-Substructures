for file in output/*; do
    echo $file
    grep \"name $file
    grep \"model $file
    grep \"learning_rate $file
    grep ^Parameters $file
    grep MAE $file | tail -n 1
    echo "================="
done
