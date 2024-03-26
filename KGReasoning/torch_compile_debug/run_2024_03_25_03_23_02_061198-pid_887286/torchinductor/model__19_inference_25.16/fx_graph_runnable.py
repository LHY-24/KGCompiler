
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        slice_1 = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, 9223372036854775807)
        select = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
        index = torch.ops.aten.index.Tensor(arg0_1, [select]);  arg0_1 = select = None
        full = torch.ops.aten.full.default([1, 400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_2 = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, 9223372036854775807)
        select_1 = torch.ops.aten.select.int(slice_2, 1, 1);  slice_2 = None
        index_1 = torch.ops.aten.index.Tensor(arg1_1, [select_1]);  arg1_1 = select_1 = None
        slice_3 = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, 9223372036854775807);  arg3_1 = None
        select_2 = torch.ops.aten.select.int(slice_3, 1, 1);  slice_3 = None
        index_2 = torch.ops.aten.index.Tensor(arg2_1, [select_2]);  arg2_1 = select_2 = None
        add = torch.ops.aten.add.Tensor(index, index_1);  index = index_1 = None
        add_1 = torch.ops.aten.add.Tensor(full, index_2);  full = index_2 = None
        return (add, add_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 23208000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (14505, 400), requires_grad=True, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 758400, device=device(type='cuda', index=0))
    reader.tensor(buf1, (474, 400), requires_grad=True, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 758400, device=device(type='cuda', index=0))
    reader.tensor(buf2, (474, 400), requires_grad=True, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf3, (1, 2), dtype=torch.int64, is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
