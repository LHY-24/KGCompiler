set +e
backends=("hidet" "eager" "inductor")
for compile_backend in "${backends[@]}"
do
    # FB15k-237
    echo 1
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
    --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --compile "$compile_backend" > time_count/vec-FB15k-237-betae-max_steps-cpu_num-1p.2p.3p.2i.3i.ip.pi.2u.up-$compile_backend.txt

    # CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    #   --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
    #   --cpu_num 1 --geo vec  \
    #   --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"
    echo 2
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 \
    -betam "(1600,2)" --compile "$compile_backend" > time_count/beta-FB15k-237-betae-alltask-$compile_backend.txt

    echo 3
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
    -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --compile "$compile_backend" > time_count/box-FB15k-237-betae-1p.2p.3p.2i.3i.ip.pi.2u.up-1-$compile_backend.txt


    echo 4
    # FB15k
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-betae -n 128 -b 512 -d 800 -g 24 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
    --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --compile "$compile_backend" > time_count/vec-FB15k-betae-1p.2p.3p.2i.3i.ip.pi.2u.up-$compile_backend.txt

    echo 5
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-betae -n 128 -b 512 -d 400 -g 60 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 \
    -betam "(1600,2)" --compile "$compile_backend" > time_count/beta-FB15k-betae-alltask-$compile_backend.txt

    echo 6
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-betae -n 128 -b 512 -d 400 -g 24 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
    -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --compile "$compile_backend" > time_count/box-FB15k-betae-1p.2p.3p.2i.3i.ip.pi.2u.up-$compile_backend.txt

    echo 7
    # NELL
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/NELL-betae -n 128 -b 512 -d 800 -g 24 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
    --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --compile "$compile_backend" > time_count/vec-NELL-betae-1p.2p.3p.2i.3i.ip.pi.2u.up-$compile_backend.txt

    echo 8
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/NELL-betae -n 128 -b 512 -d 400 -g 60 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 \
    -betam "(1600,2)" --compile "$compile_backend" > time_count/beta-NELL-betae-alltask-$compile_backend.txt

    echo 9
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/NELL-betae -n 128 -b 512 -d 400 -g 24 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
    -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --compile "$compile_backend" > time_count/box-NELL-betae-1p.2p.3p.2i.3i.ip.pi.2u.up-$compile_backend.txt


    echo 9
    ## Evaluation
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
    -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
    -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --checkpoint_path $CKPT_PATH --compile "$compile_backend" > time_count/box-FB15k-237-betae-1p.2p.3p.2i.3i.ip.pi.2u.up-2-$compile_backend.txt

    echo 10
    ## Q2B
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
    --geo box  -boxm "(none,0.02)" --tasks "1p" --compile "$compile_backend" > time_count/box-FB15k-237-betae-1p-$compile_backend.txt

    echo 11
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
    --geo box  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --compile "$compile_backend" > time_count/box-FB15k-237-betae-1p.2p.3p.2i.3i.ip.pi.2u.up-3-$compile_backend.txt

    echo 12
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
    --geo box  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" \
    --test_batch_size 10 --compile "$compile_backend" > time_count/box-FB15k-237-betae-1p.2p.3p.2i.3i.ip.pi.2u.up-batchsize10-$compile_backend.txt

    echo 13
    # GQE
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
    --geo vec --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --compile "$compile_backend" > time_count/vec-FB15k-237-betae-1p.2p.3p.2i.3i.ip.pi.2u.up-$compile_backend.txt

    echo 14
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-betae -n 128 -b 512 -d 800 -g 24 \
    --geo vec --tasks "1p" \
    --test_batch_size 1 --compile "$compile_backend" > time_count/vec-FB15k-betae-1p-$compile_backend.txt

    echo 15
    # BETAE
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
    --geo beta -betam "(1600,2)" --compile "$compile_backend" > time_count/beta-FB15k-237-betae-$compile_backend.txt

    echo 16
    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
    --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
    --geo beta -betam "(1600,2)" \
    --test_batch_size 10 --compile "$compile_backend" > time_count/beta-FB15k-237-betae-batchsize10-$compile_backend.txt
done

set -e