
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


# kernel path: /tmp/torchinductor_linhongyu/qk/cqkcl2q3cgf3golspgetfvbdn522zaxlbtaqskbvqedt4s5bu3yw.py
# Source Nodes: [delta, distance_in, distance_out, mul, negative_logit, norm, norm_1, sub, sub_1, sub_2], Original ATen: [aten.abs, aten.linalg_vector_norm, aten.minimum, aten.mul, aten.relu, aten.sub]
# delta => abs_1
# distance_in => minimum
# distance_out => relu
# mul => mul
# negative_logit => sub_3
# norm => abs_2, pow_2, sum_1
# norm_1 => abs_3, pow_4, sum_2
# sub => sub
# sub_1 => sub_1
# sub_2 => sub_2
triton_red_fused_abs_linalg_vector_norm_minimum_mul_relu_sub_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_linalg_vector_norm_minimum_mul_relu_sub_0', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14505
    rnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp11 = tl.load(in_ptr2 + (1))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 14505
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 14505)) | ~xmask, "index out of bounds: 0 <= tmp3 < 14505")
        tmp4 = tl.load(in_ptr1 + (r1 + (400*tmp3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 + 14505
        tmp8 = tmp6 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp6)
        tl.device_assert((0 <= tmp9) & (tmp9 < 14505), "index out of bounds: 0 <= tmp9 < 14505")
        tmp10 = tl.load(in_ptr1 + (r1 + (400*tmp9)), rmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 + 474
        tmp14 = tmp12 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp12)
        tl.device_assert((0 <= tmp15) & (tmp15 < 474), "index out of bounds: 0 <= tmp15 < 474")
        tmp16 = tl.load(in_ptr3 + (r1 + (400*tmp15)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp10 + tmp16
        tmp18 = tmp4 - tmp17
        tmp19 = tl.abs(tmp18)
        tmp20 = tl.load(in_ptr4 + (r1 + (400*tmp15)), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = triton_helpers.maximum(0, tmp21)
        tmp23 = tl.abs(tmp22)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
        tl.store(out_ptr0 + (r1 + (400*x0)), tmp19, rmask & xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp28 = tl.load(in_ptr2 + (1))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    _tmp37 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp27 = tl.load(out_ptr0 + (r1 + (400*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tmp29 + 474
        tmp31 = tmp29 < 0
        tmp32 = tl.where(tmp31, tmp30, tmp29)
        tl.device_assert((0 <= tmp32) & (tmp32 < 474), "index out of bounds: 0 <= tmp32 < 474")
        tmp33 = tl.load(in_ptr4 + (r1 + (400*tmp32)), rmask, eviction_policy='evict_first', other=0.0)
        tmp34 = triton_helpers.minimum(tmp27, tmp33)
        tmp35 = tl.abs(tmp34)
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp38 = _tmp37 + tmp36
        _tmp37 = tl.where(rmask & xmask, tmp38, _tmp37)
    tmp37 = tl.sum(_tmp37, 1)[:, None]
    tmp39 = tl.load(in_ptr5 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, 1])
    tmp41 = tmp40 - tmp25
    tmp42 = 0.02
    tmp43 = tmp37 * tmp42
    tmp44 = tmp41 - tmp43
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp44, xmask)
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
    assert_size_stride(arg0_1, (14505, 400), (400, 1))
    assert_size_stride(arg1_1, (474, 400), (400, 1))
    assert_size_stride(arg2_1, (474, 400), (400, 1))
    assert_size_stride(arg3_1, (1, ), (1, ))
    assert_size_stride(arg4_1, (1, 2), (2, 1))
    assert_size_stride(arg5_1, (1, 14505), (14505, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 14505, 400), device='cuda', dtype=torch.float32)
        buf1 = empty((1, 14505), device='cuda', dtype=torch.float32)
        buf3 = buf1; del buf1  # reuse
        # Source Nodes: [delta, distance_in, distance_out, mul, negative_logit, norm, norm_1, sub, sub_1, sub_2], Original ATen: [aten.abs, aten.linalg_vector_norm, aten.minimum, aten.mul, aten.relu, aten.sub]
        stream0 = get_cuda_stream(0)
        triton_red_fused_abs_linalg_vector_norm_minimum_mul_relu_sub_0.run(buf3, arg5_1, arg0_1, arg4_1, arg1_1, arg2_1, arg3_1, buf0, 14505, 400, grid=grid(14505), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((14505, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((474, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((474, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 2), (2, 1), device='cuda:0', dtype=torch.int64)
    arg5_1 = rand_strided((1, 14505), (14505, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
