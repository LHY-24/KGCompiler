mkdir test_BETAE_all_tasks_1

backends=("eager" "inductor" "hidet")
# datasets=(FB15k-237-betae FB15k-betae NELL-betae)
datasets=(FB15k-237-betae)
query_tasks=("1p" "2p" "3p" "2i" "3i" "ip" "pi" "2u" "up" "2in" "3in" "inp" "pin" "pni" "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up")
batchsizes=(1)
cpu=(10)

for ds in ${datasets[@]}
do
    for task in "${query_tasks[@]}"
    do
        for bs in ${batchsizes[@]}
        do
            for compile_backend in "${backends[@]}"
            do
                echo $ds $task $bs $compile_backend

                CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
                --data_path data/$ds -n 128 -b 512 -d 400 -g 60 \
                -lr 0.0001 --max_steps 450001 --cpu_num $cpu --geo beta --valid_steps 15000 \
                -betam "(1600,2)" --tasks $task --test_batch_size $bs \
                --compile "$compile_backend" > test_BETAE_all_tasks_1/$ds-$task-$bs-$compile_backend.txt
            done
        done
    done
done