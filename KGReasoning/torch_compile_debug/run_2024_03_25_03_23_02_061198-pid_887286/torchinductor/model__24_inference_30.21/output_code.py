
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


# kernel path: /tmp/torchinductor_linhongyu/gj/cgju4mpyhplpcv23g34kjzihby4l3jz5wuka3c3gd3cq2mg6hvzx.py
# Source Nodes: [stack_2, stack_3], Original ATen: [aten.stack]
# stack_2 => cat_1
# stack_3 => cat
triton_poi_fused_stack_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 400)
    x0 = xindex % 400
    x2 = xindex
    tmp5 = tl.load(in_ptr0 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (3))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp28 = tl.load(in_ptr0 + (5))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp38 = tl.load(in_ptr0 + (0))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp48 = tl.load(in_ptr0 + (2))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
    tmp58 = tl.load(in_ptr0 + (4))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp7 = tmp6 + 474
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 474)) | ~tmp4, "index out of bounds: 0 <= tmp9 < 474")
    tmp10 = tl.load(in_ptr1 + (x0 + (400*tmp9)), tmp4 & xmask, other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 2, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tmp13 & tmp15
    tmp19 = tmp18 + 474
    tmp20 = tmp18 < 0
    tmp21 = tl.where(tmp20, tmp19, tmp18)
    tl.device_assert(((0 <= tmp21) & (tmp21 < 474)) | ~tmp16, "index out of bounds: 0 <= tmp21 < 474")
    tmp22 = tl.load(in_ptr1 + (x0 + (400*tmp21)), tmp16 & xmask, other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp16, tmp22, tmp23)
    tmp25 = tmp0 >= tmp14
    tmp26 = tl.full([1], 3, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp30 = tmp29 + 474
    tmp31 = tmp29 < 0
    tmp32 = tl.where(tmp31, tmp30, tmp29)
    tl.device_assert(((0 <= tmp32) & (tmp32 < 474)) | ~tmp25, "index out of bounds: 0 <= tmp32 < 474")
    tmp33 = tl.load(in_ptr1 + (x0 + (400*tmp32)), tmp25 & xmask, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp25, tmp33, tmp34)
    tmp36 = tl.where(tmp16, tmp24, tmp35)
    tmp37 = tl.where(tmp4, tmp12, tmp36)
    tmp40 = tmp39 + 14505
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tl.device_assert(((0 <= tmp42) & (tmp42 < 14505)) | ~tmp4, "index out of bounds: 0 <= tmp42 < 14505")
    tmp43 = tl.load(in_ptr2 + (x0 + (400*tmp42)), tmp4 & xmask, other=0.0)
    tmp44 = tl.load(in_ptr3 + (x0 + (400*tmp9)), tmp4 & xmask, other=0.0)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp4, tmp45, tmp46)
    tmp50 = tmp49 + 14505
    tmp51 = tmp49 < 0
    tmp52 = tl.where(tmp51, tmp50, tmp49)
    tl.device_assert(((0 <= tmp52) & (tmp52 < 14505)) | ~tmp16, "index out of bounds: 0 <= tmp52 < 14505")
    tmp53 = tl.load(in_ptr2 + (x0 + (400*tmp52)), tmp16 & xmask, other=0.0)
    tmp54 = tl.load(in_ptr3 + (x0 + (400*tmp21)), tmp16 & xmask, other=0.0)
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp16, tmp55, tmp56)
    tmp60 = tmp59 + 14505
    tmp61 = tmp59 < 0
    tmp62 = tl.where(tmp61, tmp60, tmp59)
    tl.device_assert(((0 <= tmp62) & (tmp62 < 14505)) | ~tmp25, "index out of bounds: 0 <= tmp62 < 14505")
    tmp63 = tl.load(in_ptr2 + (x0 + (400*tmp62)), tmp25 & xmask, other=0.0)
    tmp64 = tl.load(in_ptr3 + (x0 + (400*tmp32)), tmp25 & xmask, other=0.0)
    tmp65 = tmp63 + tmp64
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp25, tmp65, tmp66)
    tmp68 = tl.where(tmp16, tmp57, tmp67)
    tmp69 = tl.where(tmp4, tmp47, tmp68)
    tl.store(out_ptr0 + (x2), tmp37, xmask)
    tl.store(out_ptr1 + (x2), tmp69, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_linhongyu/lq/clqfb6x5el66kneoqspbb6zq3ktlpq6nezxjeex27c7ag5e2duxd.py
# Source Nodes: [layer1_act], Original ATen: [aten.relu]
# layer1_act => relu
triton_poi_fused_relu_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 400
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_linhongyu/l5/cl52oi6g2rf2ajv35yr54tcwuuejweffobmya65cv2pzszhuvdk3.py
# Source Nodes: [attention, embedding_10, mul], Original ATen: [aten._softmax, aten.mul, aten.sum]
# attention => amax, div, exp, sub, sum_1
# embedding_10 => sum_2
# mul => mul
triton_poi_fused__softmax_mul_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_mul_sum_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (400 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (800 + x0), xmask)
    tmp14 = tl.load(in_ptr1 + (x0), xmask)
    tmp17 = tl.load(in_ptr1 + (400 + x0), xmask)
    tmp21 = tl.load(in_ptr1 + (800 + x0), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tmp1 - tmp4
    tmp8 = tl.exp(tmp7)
    tmp9 = tmp6 + tmp8
    tmp10 = tmp3 - tmp4
    tmp11 = tl.exp(tmp10)
    tmp12 = tmp9 + tmp11
    tmp13 = tmp6 / tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp8 / tmp12
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tmp11 / tmp12
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tl.store(out_ptr0 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_linhongyu/y4/cy4ngbhs5xv47k6ni73z6raaow5c22npnfhzkarrd5wjkhwnx6pk.py
# Source Nodes: [layer1_act_1, layer1_mean], Original ATen: [aten.mean, aten.relu]
# layer1_act_1 => relu_1
# layer1_mean => mean
triton_poi_fused_mean_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr0 + (400 + x0), xmask)
    tmp8 = tl.load(in_ptr0 + (800 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp5 = tmp4 + tmp1
    tmp6 = triton_helpers.maximum(0, tmp5)
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8 + tmp1
    tmp10 = triton_helpers.maximum(0, tmp9)
    tmp11 = tmp7 + tmp10
    tmp12 = 3.0
    tmp13 = tmp11 / tmp12
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_linhongyu/fn/cfnyyflgge3ajcunrul6ldsbwwlaym5ssjjm64xxb2lxtxxjb4ln.py
# Source Nodes: [gate, min_1, offset_embedding_9], Original ATen: [aten.min, aten.mul, aten.sigmoid]
# gate => sigmoid
# min_1 => min_1
# offset_embedding_9 => mul_1
triton_poi_fused_min_mul_sigmoid_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_min_mul_sigmoid_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (400 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (800 + x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp4 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
    args.clear()
    s0 = arg11_1
    assert_size_stride(arg0_1, (14505, 400), (400, 1))
    assert_size_stride(arg1_1, (474, 400), (400, 1))
    assert_size_stride(arg2_1, (474, 400), (400, 1))
    assert_size_stride(arg3_1, (400, 400), (400, 1))
    assert_size_stride(arg4_1, (400, ), (1, ))
    assert_size_stride(arg5_1, (400, 400), (400, 1))
    assert_size_stride(arg6_1, (400, ), (1, ))
    assert_size_stride(arg7_1, (400, 400), (400, 1))
    assert_size_stride(arg8_1, (400, ), (1, ))
    assert_size_stride(arg9_1, (400, 400), (400, 1))
    assert_size_stride(arg10_1, (400, ), (1, ))
    assert_size_stride(arg12_1, (1, s0), (s0, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((3, 400), device='cuda', dtype=torch.float32)
        buf1 = empty((3, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [stack_2, stack_3], Original ATen: [aten.stack]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_stack_0.run(arg12_1, arg2_1, arg0_1, arg1_1, buf0, buf1, 1200, grid=grid(1200), stream=stream0)
        del arg0_1
        del arg12_1
        del arg1_1
        del arg2_1
        buf2 = empty((3, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1, (3, 400), (400, 1), 0), reinterpret_tensor(arg3_1, (400, 400), (1, 400), 0), out=buf2)
        del arg3_1
        buf3 = reinterpret_tensor(buf2, (3, 1, 400), (400, 400, 1), 0); del buf2  # reuse
        # Source Nodes: [layer1_act], Original ATen: [aten.relu]
        triton_poi_fused_relu_1.run(buf3, arg4_1, 1200, grid=grid(1200), stream=stream0)
        del arg4_1
        buf4 = empty((3, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__self___center_net_layer2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg6_1, reinterpret_tensor(buf3, (3, 400), (400, 1), 0), reinterpret_tensor(arg5_1, (400, 400), (1, 400), 0), alpha=1, beta=1, out=buf4)
        del arg5_1
        del arg6_1
        del buf3
        buf5 = empty((1, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention, embedding_10, mul], Original ATen: [aten._softmax, aten.mul, aten.sum]
        triton_poi_fused__softmax_mul_sum_2.run(buf4, buf1, buf5, 400, grid=grid(400), stream=stream0)
        del buf1
        buf6 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf0, (3, 400), (400, 1), 0), reinterpret_tensor(arg7_1, (400, 400), (1, 400), 0), out=buf6)
        del arg7_1
        buf7 = empty((1, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer1_act_1, layer1_mean], Original ATen: [aten.mean, aten.relu]
        triton_poi_fused_mean_relu_3.run(buf6, arg8_1, buf7, 400, grid=grid(400), stream=stream0)
        del arg8_1
        del buf6
        buf8 = empty((1, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [layer1_act_1, layer1_mean], Original ATen: [aten.mean, aten.relu]
        extern_kernels.mm(buf7, reinterpret_tensor(arg9_1, (400, 400), (1, 400), 0), out=buf8)
        del arg9_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Source Nodes: [gate, min_1, offset_embedding_9], Original ATen: [aten.min, aten.mul, aten.sigmoid]
        triton_poi_fused_min_mul_sigmoid_4.run(buf9, buf0, arg10_1, 400, grid=grid(400), stream=stream0)
        del arg10_1
        return (buf5, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((14505, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((474, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((474, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((400, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((400, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((400, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((400, 400), (400, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = 6
    arg12_1 = rand_strided((1, 6), (6, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
