backends=("inductor" "eager")
datasets=(FB15k-237-betae FB15k-betae NELL-betae)
query_tasks=("1p" "2p" "3p" "2i" "3i" "ip" "pi" "2u" "up" "2in" "3in" "inp" "pin" "pni" "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up")
# geos=("vec", "box", "beta")
geos=("vec" "box" "beta")
batchsizes=(1 2 4 8 16)
cpu=(10)
timer_output=Timer/$(date "+%Y-%m-%d-%H:%M:%S")
PY_EXE=/home/hongyu2021/anaconda3/envs/smore/bin/python
MAIN_PY=/home/hongyu2021/KG-Compilation/KGReasoning/main.py

mkdir $timer_output

for geo in ${geos[@]}
do
    
    for ds in ${datasets[@]}
    do
        for task in "${query_tasks[@]}"
        do
            for bs in ${batchsizes[@]}
            do
                for compile_backend in "${backends[@]}"
                do
                    # 创建文件夹
                    exact_output_dir=${timer_output}

                    for d in ${geo} ${ds} ${task} ${bs} ${compile_backend}
                    do
                        exact_output_dir=$exact_output_dir/$d
                        if [ ! -d $exact_output_dir ];then
                            mkdir $exact_output_dir
                        fi
                    done
                    echo "mkdir:" $exact_output_dir


                    # 后台执行推理脚本 && 记录pid
                    CUDA_VISIBLE_DEVICES=0 $PY_EXE $MAIN_PY --cuda --do_test \
                    --data_path data/$ds -n 128 -b 512 -d 400 -g 60 \
                    -lr 0.0001 --max_steps 450001 --cpu_num $cpu --geo $geo --valid_steps 15000 \
                    -betam "(1600,2)" --tasks $task --test_batch_size $bs \
                    --compile "$compile_backend" --timer_output $timer_output &
                    do_test_pid=$!

                    echo "evaluate process pid: " $do_test_pid

                    # 后台执行显存测量脚本 && 记录pid
                    $PY_EXE ./nvidia_smi.py  -o $exact_output_dir/nv_mem.txt -p $do_test_pid &
                    nv_mem_meas_pid=$!

                    # 等待推理结束，杀死显存测量进程
                    wait $do_test_pid
                    kill -9 $nv_mem_meas_pid
                done
            done
        done
    done
done