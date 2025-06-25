mkdir test_time_GQE_FB15K-237

backends=("hidet" "eager" "inductor")
query_tasks=("1p" "1p2p" "2i" "2u" "1p.2p.3p.2i.3i.ip.pi.2u.up")
batchsizes=(1 10 100)

for task in "${query_tasks[@]}"
do
    for bs in ${batchsizes[@]}
    do
        for compile_backend in "${backends[@]}"
        do
            echo $task $bs $compile_backend

            CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
            --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
            -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
            --tasks $task --test_batch_size $bs \
            --compile "$compile_backend" > test_time_GQE_FB15K-237/$task-$bs-$compile_backend.txt
        done
    done
done