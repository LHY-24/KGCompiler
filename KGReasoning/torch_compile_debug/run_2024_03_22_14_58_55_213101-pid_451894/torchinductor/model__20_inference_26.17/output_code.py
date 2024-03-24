
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


# kernel path: /tmp/torchinductor_caohanghang/2f/c2fkxlqjeu5tfkahzhwnvrzoedpa2em7rmyoupsk2g24sebfbflg.py
# Source Nodes: [delta, distance_in, distance_out, logit, mul, norm, norm_1, sub, sub_1, sub_2], Original ATen: [aten.abs, aten.linalg_vector_norm, aten.minimum, aten.mul, aten.relu, aten.sub]
# delta => abs_1
# distance_in => minimum
# distance_out => relu
# logit => sub_3
# mul => mul
# norm => abs_2, pow_2, sum_1
# norm_1 => abs_3, pow_4, sum_2
# sub => sub
# sub_1 => sub_1
# sub_2 => sub_2
triton_per_fused_abs_linalg_vector_norm_minimum_mul_relu_sub_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_linalg_vector_norm_minimum_mul_relu_sub_0', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 14505
    XBLOCK: tl.constexpr = 1
    rnumel = 400
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (400*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + (0))
    tmp19 = tl.broadcast_to(tmp18, [1])
    tmp2 = tmp0 - tmp1
    tmp3 = tl.abs(tmp2)
    tmp5 = tmp3 - tmp4
    tmp6 = triton_helpers.maximum(0, tmp5)
    tmp7 = tl.abs(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = triton_helpers.minimum(tmp3, tmp4)
    tmp13 = tl.abs(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp20 = tmp19 - tmp11
    tmp21 = 0.02
    tmp22 = tmp17 * tmp21
    tmp23 = tmp20 - tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, ), (1, ))
    assert_size_stride(arg1_1, (1, 14505, 400), (5802000, 400, 1))
    assert_size_stride(arg2_1, (1, 1, 400), (400, 400, 1))
    assert_size_stride(arg3_1, (1, 1, 400), (400, 400, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 14505), device='cuda', dtype=torch.float32)
        buf2 = buf0; del buf0  # reuse
        # Source Nodes: [delta, distance_in, distance_out, logit, mul, norm, norm_1, sub, sub_1, sub_2], Original ATen: [aten.abs, aten.linalg_vector_norm, aten.minimum, aten.mul, aten.relu, aten.sub]
        stream0 = get_cuda_stream(0)
        triton_per_fused_abs_linalg_vector_norm_minimum_mul_relu_sub_0.run(buf2, arg1_1, arg2_1, arg3_1, arg0_1, 14505, 400, grid=grid(14505), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 14505, 400), (5802000, 400, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 400), (400, 400, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1, 400), (400, 400, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
