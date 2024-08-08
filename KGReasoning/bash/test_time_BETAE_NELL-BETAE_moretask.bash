mkdir test_time_BETAE_NELL-BETAE_moretask

backends=("hidet" "eager")
query_tasks=("1p" "2i" "ip" "pi" "2in" "inp" "pin" "pni" "2u" "up" "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up")
batchsizes=(1 10)

for task in "${query_tasks[@]}"
do
    for bs in ${batchsizes[@]}
    do
        for compile_backend in "${backends[@]}"
        do
            echo $task $bs $compile_backend

            CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
            --data_path data/NELL-betae -n 128 -b 512 -d 400 -g 60 \
            -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 \
            -betam "(1600,2)" --tasks $task --test_batch_size $bs \
            --compile "$compile_backend" > test_time_BETAE_NELL-BETAE_moretask/$task-$bs-$compile_backend.txt
        done
    done
done