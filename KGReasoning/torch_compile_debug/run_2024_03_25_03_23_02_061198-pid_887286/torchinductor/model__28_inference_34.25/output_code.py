
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


# kernel path: /tmp/torchinductor_linhongyu/h4/ch466eesb2ovo4qmx5swvoumpimimr7rakke2fwl7wmicpfft7qe.py
# Source Nodes: [embedding, embedding_1, offset_embedding_1, r_embedding, r_offset_embedding, zeros_like], Original ATen: [aten.add, aten.index_select, aten.zeros_like]
# embedding => index
# embedding_1 => add
# offset_embedding_1 => add_1
# r_embedding => index_1
# r_offset_embedding => index_2
# zeros_like => full
triton_poi_fused_add_index_select_zeros_like_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_zeros_like_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 400)
    x0 = xindex % 400
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (ks0*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + (ks0*x1)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 + 14505
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 14505)) | ~xmask, "index out of bounds: 0 <= tmp3 < 14505")
    tmp4 = tl.load(in_ptr1 + (x0 + (400*tmp3)), xmask)
    tmp6 = tmp5 + 474
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 474)) | ~xmask, "index out of bounds: 0 <= tmp8 < 474")
    tmp9 = tl.load(in_ptr2 + (x0 + (400*tmp8)), xmask)
    tmp10 = tmp4 + tmp9
    tmp11 = tl.load(in_ptr3 + (x0 + (400*tmp8)), xmask)
    tmp12 = 0.0
    tmp13 = tmp12 + tmp11
    tl.store(out_ptr0 + (x2), tmp10, xmask)
    tl.store(out_ptr1 + (x2), tmp13, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    s0 = arg3_1
    s1 = arg4_1
    assert_size_stride(arg0_1, (14505, 400), (400, 1))
    assert_size_stride(arg1_1, (474, 400), (400, 1))
    assert_size_stride(arg2_1, (474, 400), (400, 1))
    assert_size_stride(arg5_1, (s0, s1), (s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((s0, 400), device='cuda', dtype=torch.float32)
        buf1 = empty((s0, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [embedding, embedding_1, offset_embedding_1, r_embedding, r_offset_embedding, zeros_like], Original ATen: [aten.add, aten.index_select, aten.zeros_like]
        triton_poi_fused_add_index_select_zeros_like_0_xnumel = 400*s0
        stream0 = get_cuda_stream(0)
        triton_poi_fused_add_index_select_zeros_like_0.run(arg5_1, arg0_1, arg1_1, arg2_1, buf0, buf1, s1, triton_poi_fused_add_index_select_zeros_like_0_xnumel, grid=grid(triton_poi_fused_add_index_select_zeros_like_0_xnumel), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg5_1
        return (buf0, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((14505, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((474, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((474, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = 2
    arg4_1 = 2
    arg5_1 = rand_strided((2, 2), (2, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
