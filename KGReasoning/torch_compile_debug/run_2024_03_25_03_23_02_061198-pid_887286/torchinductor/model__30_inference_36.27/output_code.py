
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_linhongyu/pv/cpvj55rpjls4fmj7c4fwaltalfi43lvktaidoxlupowxzob3rvmp.py
# Source Nodes: [cat_3, queries_1], Original ATen: [aten.cat, aten.view]
# cat_3 => cat_2
# queries_1 => view
triton_poi_fused_cat_view_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp13 = tl.load(in_ptr0 + (5))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 2, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (x0), tmp7 & xmask, other=0.0)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp0 >= tmp5
    tmp12 = tmp11 & tmp4
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tl.where(tmp6, tmp10, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 6, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = (-3) + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp5
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr0 + ((-1) + x0), tmp26 & xmask, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp23 >= tmp5
    tmp31 = tmp23 < tmp3
    tmp32 = tmp30 & tmp20
    tmp33 = tl.where(tmp32, tmp14, tmp15)
    tmp34 = tl.where(tmp25, tmp29, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp20, tmp34, tmp35)
    tmp37 = tl.where(tmp4, tmp19, tmp36)
    tl.store(out_ptr0 + (x0), tmp37, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (1, s0), (s0, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((2, 3), device='cuda', dtype=torch.int64)
        # Source Nodes: [cat_3, queries_1], Original ATen: [aten.cat, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_cat_view_0.run(arg1_1, buf0, 6, grid=grid(6), stream=stream0)
        del arg1_1
        return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 6
    arg1_1 = rand_strided((1, 6), (6, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
