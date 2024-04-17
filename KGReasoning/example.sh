
# FB15k-237
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

# CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
#   --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
#   --cpu_num 1 --geo vec  \
#   --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

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
  --geo box  -boxm "(none,0.02)" --tasks "1p"
# inductor-1  [evaluate]-1 times per count: 320269.2 ms
# inductor-2  [evaluate]-1 times per count: 257673.1 ms
# inductor-3  [evaluate]-1 times per count: 230937.9 ms
# [evaluate]-1 times per count: 110186.3 ms
# [evaluate]-2 times per count: 105806.2 ms
# [evaluate]-3 times per count: 102878.4 ms
# inductor-4   [evaluate]-1 times per count: 299353.9 ms
# inductor-5   [evaluate]-1 times per count: 199687.6 ms

# eager-1     [evaluate]-1 times per count: 264392.0 ms
# eager-2     [evaluate]-1 times per count: 120601.9 ms
# eager-3     [evaluate]-1 times per count:  90068.4 ms
# eager-4     [evaluate]-1 times per count:  90496.3 ms
# [evaluate]-1 times per count: 89555.2 ms
# [evaluate]-2 times per count: 106329.9 ms
# [evaluate]-3 times per count: 169766.9 ms
# eager-5     [evaluate]-1 times per count: 300117.8 ms
# eager-6     [evaluate]-1 times per count: 297786.5 ms

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  --geo box  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"
# Inductor 
# [evaluate]-1 times per count: 146977.4 ms
# [evaluate]-2 times per count: 144994.5 ms
# [evaluate]-3 times per count: 149926.7 ms

# [time_counter1]-1 times per count:  149876.1 ms
# [evaluate]-1 times per count:       149890.8 ms
# [time_counter1]-2 times per count: 152607.7 ms
# [evaluate]-2 times per count: 152624.6 ms
# [time_counter1]-3 times per count: 158090.1 ms
# [evaluate]-3 times per count: 158107.6 ms

# GQE
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
  --geo vec --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"

# BETAE
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
  --geo beta -betam "(1600,2)"

