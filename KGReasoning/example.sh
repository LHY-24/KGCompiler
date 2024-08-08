
# FB15k-237
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

# CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
#   --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
#   --cpu_num 1 --geo vec  \
#   --tasks "1p"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 \
  -betam "(1600,2)"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"


# FB15k
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-betae -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-betae -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 \
  -betam "(1600,2)"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-betae -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"


# NELL
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/NELL-betae -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/NELL-betae -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 \
  -betam "(1600,2)"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/NELL-betae -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"


## Evaluation
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --checkpoint_path $CKPT_PATH


## Q2B
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p" 

# -n 128 -b 512 -d 400 --cpu_num 1 --tasks "1p"   eager: 99909.28 ms   inductor: 101665.20 ms   hidet: 97865.99 ms


CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  --geo box  -boxm "(none,0.02)" --tasks "pi"

TORCH_COMPILE_DEBUG=1 CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  --geo box  -boxm "(none,0.02)" --tasks "1p"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  --geo box  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  --geo box  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" \
  --test_batch_size 10


# GQE
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 1024 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 2000001 --cpu_num 6 --valid_steps 15000 \
  --geo vec \
  --tasks "1p" 
# eager: 43084.94 ms
# inductor: 42964.54 ms / 40249.14 ms

# -n, -d, -cpu_num
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 1024 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 10 --valid_steps 15000 \
  --test_batch_size 10 \
  --geo vec \
  --tasks "1p" \
  --compile "inductor"

# task: 1p
# --cpu_num=6   -n 128  --test_batch_size 1   eager: inductor: 39832.19 ms
# --cpu_num=10  -n 128  --test_batch_size 1   inductor: 42039.83 ms (eager: 44735.90 ms)
# --cpu_num=6   -n 1024 --test_batch_size 1   inductor: 40609.68 ms (eager: 43111.63 ms)
# --cpu_num=6   -n 1024 --test_batch_size 10  inductor: 28378.10 ms (eager: 20641.07 ms)
# --cpu_num=1   -n 1024 --test_batch_size 10  eager: 58325.31 ms  inductor: 61222.72 ms hidet: 60077.38 ms
# --cpu_num=10  -n 1024 --test_batch_size 10  eager: 21031.73 ms  inductor: 21460.79 ms hidet: 20694.09 ms

# task: 3p
# --cpu_num=10  -n 1024 --test_batch_size 10  eager: 13413.66 ms  inductor: 15565.48 ms hidet: 13610.83 ms

# task: 5p
# --cpu_num=10  -n 1024 --test_batch_size 10  eager: 344.13 ms  inductor: 402.53 ms hidet: 346.26 ms


# CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
#   --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
#   -lr 0.0001 --max_steps 450001 --cpu_num 1 --valid_steps 15000 \
#   --geo vec \
#   --tasks "1p" 
# eager: 96999.75 ms
# inductor: 100419.55 ms

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
  --geo vec \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-betae -n 128 -b 512 -d 800 -g 24 \
  --geo vec \
  --tasks "ip" \
  --test_batch_size 1 \
  --compile hidet


# BETAE
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --valid_steps 15000 \
  --geo beta -betam "(1600,2)" \
  --tasks "1p"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
  --geo beta -betam "(1600,2)"

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
  --geo beta -betam "(1600,2)" \
  --test_batch_size 10

