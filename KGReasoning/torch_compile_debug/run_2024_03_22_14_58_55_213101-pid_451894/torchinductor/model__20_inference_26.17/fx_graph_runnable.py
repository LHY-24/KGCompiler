
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config


torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.2.1
# torch cuda version: 12.1
# torch git version: 6c8c5ad5eaf47a62fafbb4a2747198cbffbf1ff0


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Wed_Nov_22_10:17:15_PST_2023 
# Cuda compilation tools, release 12.3, V12.3.107 
# Build cuda_12.3.r12.3/compiler.33567101_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 3090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        sub = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        abs_1 = torch.ops.aten.abs.default(sub);  sub = None
        sub_1 = torch.ops.aten.sub.Tensor(abs_1, arg3_1)
        relu = torch.ops.aten.relu.default(sub_1);  sub_1 = None
        minimum = torch.ops.aten.minimum.default(abs_1, arg3_1);  abs_1 = arg3_1 = None
        abs_2 = torch.ops.aten.abs.default(relu);  relu = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(abs_2, 1);  abs_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, [-1]);  pow_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sum_1, 1.0);  sum_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(arg0_1, pow_2);  arg0_1 = pow_2 = None
        abs_3 = torch.ops.aten.abs.default(minimum);  minimum = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(abs_3, 1);  abs_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_3, [-1]);  pow_3 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(sum_2, 1.0);  sum_2 = None
        mul = torch.ops.aten.mul.Tensor(pow_4, 0.02);  pow_4 = None
        sub_3 = torch.ops.aten.sub.Tensor(sub_2, mul);  sub_2 = mul = None
        return (sub_3,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1,), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 23208000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 14505, 400), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 1, 400), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, 1, 400), is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
