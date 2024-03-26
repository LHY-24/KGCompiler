
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



# torch version: 2.2.1+cu121
# torch cuda version: 12.1
# torch git version: 6c8c5ad5eaf47a62fafbb4a2747198cbffbf1ff0


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 3090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
        sub = torch.ops.aten.sub.Tensor(arg3_1, arg5_1);  arg3_1 = arg5_1 = None
        abs_1 = torch.ops.aten.abs.default(sub);  sub = None
        sub_1 = torch.ops.aten.sub.Tensor(abs_1, arg6_1)
        relu = torch.ops.aten.relu.default(sub_1);  sub_1 = None
        minimum = torch.ops.aten.minimum.default(abs_1, arg6_1);  abs_1 = arg6_1 = None
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
    reader.symint(14505)  # arg1_1
    reader.symint(400)  # arg2_1
    buf1 = reader.storage(None, 23208000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 1, s0, s1), is_leaf=True)  # arg3_1
    reader.symint(2)  # arg4_1
    buf2 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, s2, 1, s1), is_leaf=True)  # arg5_1
    buf3 = reader.storage(None, 3200, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, s2, 1, s1), is_leaf=True)  # arg6_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
