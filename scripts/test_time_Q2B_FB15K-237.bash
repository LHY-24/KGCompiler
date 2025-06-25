mkdir test_time_Q2B_FB15K-237

backends=("hidet" "eager" "inductor")
query_tasks=("1p" "2p" "3p" "4p" "5p")
batchsizes=(1 10)

for task in "${query_tasks[@]}"
do
    for bs in ${batchsizes[@]}
    do
        for compile_backend in "${backends[@]}"
        do
            echo $task $bs $compile_backend

            CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
            --data_path data/FB15k-237-betae -n 128 -b 1024 -d 400 -g 24 \
            -lr 0.0001 --max_steps 450001 --cpu_num 10 --valid_steps 15000 \
            --geo box  -boxm "(none,0.02)" --tasks $task --test_batch_size $bs \
            --compile "$compile_backend" > test_time_Q2B_FB15K-237/$task-$bs-$compile_backend.txt
        done
    done
done