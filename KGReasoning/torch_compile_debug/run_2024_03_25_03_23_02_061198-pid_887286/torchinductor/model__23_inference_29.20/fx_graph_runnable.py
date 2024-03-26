
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1):
        slice_1 = torch.ops.aten.slice.Tensor(arg12_1, 0, 0, 9223372036854775807)
        select = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
        index = torch.ops.aten.index.Tensor(arg0_1, [select]);  select = None
        full_default = torch.ops.aten.full.default([1, 400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_2 = torch.ops.aten.slice.Tensor(arg12_1, 0, 0, 9223372036854775807)
        select_1 = torch.ops.aten.select.int(slice_2, 1, 1);  slice_2 = None
        index_1 = torch.ops.aten.index.Tensor(arg1_1, [select_1]);  select_1 = None
        slice_3 = torch.ops.aten.slice.Tensor(arg12_1, 0, 0, 9223372036854775807)
        select_2 = torch.ops.aten.select.int(slice_3, 1, 1);  slice_3 = None
        index_2 = torch.ops.aten.index.Tensor(arg2_1, [select_2]);  select_2 = None
        add = torch.ops.aten.add.Tensor(index, index_1);  index = index_1 = None
        slice_4 = torch.ops.aten.slice.Tensor(arg12_1, 0, 0, 9223372036854775807)
        select_3 = torch.ops.aten.select.int(slice_4, 1, 2);  slice_4 = None
        index_3 = torch.ops.aten.index.Tensor(arg0_1, [select_3]);  arg0_1 = select_3 = None
        full_default_1 = torch.ops.aten.full.default([1, 400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_5 = torch.ops.aten.slice.Tensor(arg12_1, 0, 0, 9223372036854775807)
        select_4 = torch.ops.aten.select.int(slice_5, 1, 3);  slice_5 = None
        index_4 = torch.ops.aten.index.Tensor(arg1_1, [select_4]);  arg1_1 = select_4 = None
        slice_6 = torch.ops.aten.slice.Tensor(arg12_1, 0, 0, 9223372036854775807);  arg12_1 = None
        select_5 = torch.ops.aten.select.int(slice_6, 1, 3);  slice_6 = None
        index_5 = torch.ops.aten.index.Tensor(arg2_1, [select_5]);  arg2_1 = select_5 = None
        add_2 = torch.ops.aten.add.Tensor(index_3, index_4);  index_3 = index_4 = None
        cat = torch.ops.aten.cat.default([add, add_2]);  add = add_2 = None
        view = torch.ops.aten.view.default(cat, [2, 1, 400]);  cat = None
        view_1 = torch.ops.aten.view.default(view, [2, 400])
        permute = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm = torch.ops.aten.addmm.default(arg4_1, view_1, permute);  arg4_1 = view_1 = permute = None
        view_2 = torch.ops.aten.view.default(addmm, [2, 1, 400]);  addmm = None
        relu = torch.ops.aten.relu.default(view_2);  view_2 = None
        view_3 = torch.ops.aten.view.default(relu, [2, 400]);  relu = None
        permute_1 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg6_1, view_3, permute_1);  arg6_1 = view_3 = permute_1 = None
        view_4 = torch.ops.aten.view.default(addmm_1, [2, 1, 400]);  addmm_1 = None
        amax = torch.ops.aten.amax.default(view_4, [0], True)
        sub = torch.ops.aten.sub.Tensor(view_4, amax);  view_4 = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [0], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul = torch.ops.aten.mul.Tensor(div, view);  div = view = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul, [0]);  mul = None
        cat_1 = torch.ops.aten.cat.default([index_2, index_5]);  index_2 = index_5 = None
        view_5 = torch.ops.aten.view.default(cat_1, [2, 1, 400]);  cat_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [2, 400])
        permute_2 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg8_1, view_6, permute_2);  arg8_1 = view_6 = permute_2 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [2, 1, 400]);  addmm_2 = None
        relu_1 = torch.ops.aten.relu.default(view_7);  view_7 = None
        mean = torch.ops.aten.mean.dim(relu_1, [0]);  relu_1 = None
        permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg10_1, mean, permute_3);  arg10_1 = mean = permute_3 = None
        sigmoid = torch.ops.aten.sigmoid.default(addmm_3);  addmm_3 = None
        min_1 = torch.ops.aten.min.dim(view_5, 0);  view_5 = None
        getitem = min_1[0];  min_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(getitem, sigmoid);  getitem = sigmoid = None
        return (sum_2, mul_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 23208000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (14505, 400), requires_grad=True, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 758400, device=device(type='cuda', index=0))
    reader.tensor(buf1, (474, 400), requires_grad=True, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 758400, device=device(type='cuda', index=0))
    reader.tensor(buf2, (474, 400), requires_grad=True, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 640000, device=device(type='cuda', index=0))
    reader.tensor(buf3, (400, 400), requires_grad=True, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf4, (400,), requires_grad=True, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 640000, device=device(type='cuda', index=0))
    reader.tensor(buf5, (400, 400), requires_grad=True, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf6, (400,), requires_grad=True, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 640000, device=device(type='cuda', index=0))
    reader.tensor(buf7, (400, 400), requires_grad=True, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf8, (400,), requires_grad=True, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 640000, device=device(type='cuda', index=0))
    reader.tensor(buf9, (400, 400), requires_grad=True, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1600, device=device(type='cuda', index=0))
    reader.tensor(buf10, (400,), requires_grad=True, is_leaf=True)  # arg10_1
    reader.symint(4)  # arg11_1
    buf11 = reader.storage(None, 8*s0, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf11, (1, s0), dtype=torch.int64, is_leaf=True)  # arg12_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
