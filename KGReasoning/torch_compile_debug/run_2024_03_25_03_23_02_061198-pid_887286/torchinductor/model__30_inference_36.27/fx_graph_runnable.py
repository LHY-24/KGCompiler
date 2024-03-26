
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

    
    
    def forward(self, arg0_1, arg1_1):
        slice_1 = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807)
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 2);  slice_1 = None
        slice_3 = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807)
        slice_4 = torch.ops.aten.slice.Tensor(slice_3, 1, 5, 6);  slice_3 = None
        cat = torch.ops.aten.cat.default([slice_2, slice_4], 1);  slice_2 = slice_4 = None
        slice_5 = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807)
        slice_6 = torch.ops.aten.slice.Tensor(slice_5, 1, 2, 4);  slice_5 = None
        slice_7 = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807);  arg1_1 = None
        slice_8 = torch.ops.aten.slice.Tensor(slice_7, 1, 5, 6);  slice_7 = None
        cat_1 = torch.ops.aten.cat.default([slice_6, slice_8], 1);  slice_6 = slice_8 = None
        cat_2 = torch.ops.aten.cat.default([cat, cat_1], 1);  cat = cat_1 = None
        view = torch.ops.aten.view.default(cat_2, [2, -1]);  cat_2 = None
        return (view,)
        
def load_args(reader):
    reader.symint(6)  # arg0_1
    buf0 = reader.storage(None, 8*s0, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (1, s0), dtype=torch.int64, is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
