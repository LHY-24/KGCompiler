
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
        self.register_buffer('_tensor_constant0', tensor([0], device='cuda:0').cuda())
        self.register_buffer('_tensor_constant1', tensor([]))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        slice_1 = torch.ops.aten.slice.Tensor(arg4_1, 0, 0, 9223372036854775807)
        select = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
        index = torch.ops.aten.index.Tensor(arg0_1, [select]);  select = None
        full_default = torch.ops.aten.full.default([1, 400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_2 = torch.ops.aten.slice.Tensor(arg4_1, 0, 0, 9223372036854775807)
        select_1 = torch.ops.aten.select.int(slice_2, 1, 1);  slice_2 = None
        index_1 = torch.ops.aten.index.Tensor(arg1_1, [select_1]);  arg1_1 = select_1 = None
        slice_3 = torch.ops.aten.slice.Tensor(arg4_1, 0, 0, 9223372036854775807);  arg4_1 = None
        select_2 = torch.ops.aten.select.int(slice_3, 1, 1);  slice_3 = None
        index_2 = torch.ops.aten.index.Tensor(arg2_1, [select_2]);  arg2_1 = select_2 = None
        add = torch.ops.aten.add.Tensor(index, index_1);  index = index_1 = None
        clone = torch.ops.aten.clone.default(add);  add = None
        unsqueeze = torch.ops.aten.unsqueeze.default(clone, 1);  clone = None
        clone_1 = torch.ops.aten.clone.default(index_2);  index_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(clone_1, 1);  clone_1 = None
        _tensor_constant0 = self._tensor_constant0
        full_default_1 = torch.ops.aten.full.default([1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_3 = torch.ops.aten.index.Tensor(arg5_1, [full_default_1]);  arg5_1 = full_default_1 = None
        view = torch.ops.aten.view.default(index_3, [-1]);  index_3 = None
        index_4 = torch.ops.aten.index.Tensor(arg0_1, [view]);  arg0_1 = view = None
        view_1 = torch.ops.aten.view.default(index_4, [1, 14505, -1]);  index_4 = None
        sub = torch.ops.aten.sub.Tensor(view_1, unsqueeze);  view_1 = unsqueeze = None
        abs_1 = torch.ops.aten.abs.default(sub);  sub = None
        sub_1 = torch.ops.aten.sub.Tensor(abs_1, unsqueeze_1)
        relu = torch.ops.aten.relu.default(sub_1);  sub_1 = None
        minimum = torch.ops.aten.minimum.default(abs_1, unsqueeze_1);  abs_1 = unsqueeze_1 = None
        abs_2 = torch.ops.aten.abs.default(relu);  relu = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(abs_2, 1);  abs_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, [-1]);  pow_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sum_1, 1.0);  sum_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(arg3_1, pow_2);  arg3_1 = pow_2 = None
        abs_3 = torch.ops.aten.abs.default(minimum);  minimum = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(abs_3, 1);  abs_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_3, [-1]);  pow_3 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(sum_2, 1.0);  sum_2 = None
        mul = torch.ops.aten.mul.Tensor(pow_4, 0.02);  pow_4 = None
        sub_3 = torch.ops.aten.sub.Tensor(sub_2, mul);  sub_2 = mul = None
        clone_2 = torch.ops.aten.clone.default(sub_3);  sub_3 = None
        return (clone_2,)
        
def load_args(reader):
    buf0 = reader.storage(None, 23208000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (14505, 400), requires_grad=True, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 758400, device=device(type='cuda', index=0))
    reader.tensor(buf1, (474, 400), requires_grad=True, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 758400, device=device(type='cuda', index=0))
    reader.tensor(buf2, (474, 400), requires_grad=True, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf4, (1, 2), dtype=torch.int64, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 116040, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf5, (1, 14505), dtype=torch.int64, is_leaf=True)  # arg5_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
