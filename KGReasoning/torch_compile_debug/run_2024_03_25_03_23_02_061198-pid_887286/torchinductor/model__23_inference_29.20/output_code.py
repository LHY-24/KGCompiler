
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


# kernel path: /tmp/torchinductor_linhongyu/b3/cb36p34akyja26j7klgzr4ztepuqvqycqdp5oy6fk2juuzwmzaau.py
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
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 400)
    x0 = xindex % 400
    x2 = xindex
    tmp5 = tl.load(in_ptr0 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp23 = tl.load(in_ptr0 + (2))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp29 = tl.load(in_ptr0 + (3))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp7 = tmp6 + 14505
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 14505)) | ~tmp4, "index out of bounds: 0 <= tmp9 < 14505")
    tmp10 = tl.load(in_ptr1 + (x0 + (400*tmp9)), tmp4 & xmask, other=0.0)
    tmp13 = tmp12 + 474
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert(((0 <= tmp15) & (tmp15 < 474)) | ~tmp4, "index out of bounds: 0 <= tmp15 < 474")
    tmp16 = tl.load(in_ptr2 + (x0 + (400*tmp15)), tmp4 & xmask, other=0.0)
    tmp17 = tmp10 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 2, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp25 = tmp24 + 14505
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 14505)) | ~tmp20, "index out of bounds: 0 <= tmp27 < 14505")
    tmp28 = tl.load(in_ptr1 + (x0 + (400*tmp27)), tmp20 & xmask, other=0.0)
    tmp31 = tmp30 + 474
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 474)) | ~tmp20, "index out of bounds: 0 <= tmp33 < 474")
    tmp34 = tl.load(in_ptr2 + (x0 + (400*tmp33)), tmp20 & xmask, other=0.0)
    tmp35 = tmp28 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp20, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp19, tmp37)
    tmp39 = tl.load(in_ptr3 + (x0 + (400*tmp15)), tmp4 & xmask, other=0.0)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp4, tmp39, tmp40)
    tmp42 = tl.load(in_ptr3 + (x0 + (400*tmp33)), tmp20 & xmask, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp20, tmp42, tmp43)
    tmp45 = tl.where(tmp4, tmp41, tmp44)
    tl.store(out_ptr0 + (x2), tmp38, xmask)
    tl.store(out_ptr1 + (x2), tmp45, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_linhongyu/kc/ckcvaxbuypkapy7wlxh76kd4hlao27nh6fsmsqusnfz2ql3goigj.py
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
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 800
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


# kernel path: /tmp/torchinductor_linhongyu/zs/czstzlzkmfagt2krx7irago5yypexr323442ey4ee2fnutydofmw.py
# Source Nodes: [attention, embedding_7, mul], Original ATen: [aten._softmax, aten.mul, aten.sum]
# attention => amax, div, exp, sub, sum_1
# embedding_7 => sum_2
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
    tmp9 = tl.load(in_ptr1 + (x0), xmask)
    tmp12 = tl.load(in_ptr1 + (400 + x0), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = tmp0 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp5 = tmp1 - tmp2
    tmp6 = tl.exp(tmp5)
    tmp7 = tmp4 + tmp6
    tmp8 = tmp4 / tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 / tmp7
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_linhongyu/cb/ccbqvke5iykt4nkqkjqobcoadme2w7begay7nfau2h35xogqsb6a.py
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
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp5 = tmp4 + tmp1
    tmp6 = triton_helpers.maximum(0, tmp5)
    tmp7 = tmp3 + tmp6
    tmp8 = 2.0
    tmp9 = tmp7 / tmp8
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_linhongyu/p6/cp65w3wm3azrlg4nrdmzly257gotdkrvryfz6xvqmrfjvs2txgrl.py
# Source Nodes: [gate, min_1, offset_embedding_6], Original ATen: [aten.min, aten.mul, aten.sigmoid]
# gate => sigmoid
# min_1 => min_1
# offset_embedding_6 => mul_1
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_min_mul_sigmoid_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp4 = tl.load(in_ptr0 + (1))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp36 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp37 = tl.load(in_ptr2 + (x0), xmask)
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp6 = tmp5 + 474
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 474)) | ~tmp3, "index out of bounds: 0 <= tmp8 < 474")
    tmp9 = tl.load(in_ptr1 + (x0 + (400*tmp8)), tmp3 & xmask, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp3, tmp9, tmp10)
    tmp12 = tmp0 >= tmp2
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp17 = tmp16 + 474
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 474)) | ~tmp12, "index out of bounds: 0 <= tmp19 < 474")
    tmp20 = tl.load(in_ptr1 + (x0 + (400*tmp19)), tmp12 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp12, tmp20, tmp21)
    tmp23 = tl.where(tmp3, tmp11, tmp22)
    tmp24 = tmp2 >= tmp0
    tmp25 = tmp2 < tmp2
    tl.device_assert(((0 <= tmp8) & (tmp8 < 474)) | ~tmp25, "index out of bounds: 0 <= tmp8 < 474")
    tmp26 = tl.load(in_ptr1 + (x0 + (400*tmp8)), tmp25 & xmask, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = tmp2 >= tmp2
    tmp30 = tmp2 < tmp13
    tl.device_assert(((0 <= tmp19) & (tmp19 < 474)) | ~tmp29, "index out of bounds: 0 <= tmp19 < 474")
    tmp31 = tl.load(in_ptr1 + (x0 + (400*tmp19)), tmp29 & xmask, other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp29, tmp31, tmp32)
    tmp34 = tl.where(tmp25, tmp28, tmp33)
    tmp35 = triton_helpers.minimum(tmp23, tmp34)
    tmp38 = tmp36 + tmp37
    tmp39 = tl.sigmoid(tmp38)
    tmp40 = tmp35 * tmp39
    tl.store(in_out_ptr0 + (x0), tmp40, xmask)
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
        buf0 = empty((2, 400), device='cuda', dtype=torch.float32)
        buf5 = empty((2, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [stack_2, stack_3], Original ATen: [aten.stack]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_stack_0.run(arg12_1, arg0_1, arg1_1, arg2_1, buf0, buf5, 800, grid=grid(800), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty((2, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf0, (2, 400), (400, 1), 0), reinterpret_tensor(arg3_1, (400, 400), (1, 400), 0), out=buf1)
        del arg3_1
        buf2 = reinterpret_tensor(buf1, (2, 1, 400), (400, 400, 1), 0); del buf1  # reuse
        # Source Nodes: [layer1_act], Original ATen: [aten.relu]
        triton_poi_fused_relu_1.run(buf2, arg4_1, 800, grid=grid(800), stream=stream0)
        del arg4_1
        buf3 = empty((2, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__self___center_net_layer2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg6_1, reinterpret_tensor(buf2, (2, 400), (400, 1), 0), reinterpret_tensor(arg5_1, (400, 400), (1, 400), 0), alpha=1, beta=1, out=buf3)
        del arg5_1
        del arg6_1
        del buf2
        buf4 = empty((1, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention, embedding_7, mul], Original ATen: [aten._softmax, aten.mul, aten.sum]
        triton_poi_fused__softmax_mul_sum_2.run(buf3, buf0, buf4, 400, grid=grid(400), stream=stream0)
        del buf0
        buf6 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf5, (2, 400), (400, 1), 0), reinterpret_tensor(arg7_1, (400, 400), (1, 400), 0), out=buf6)
        del arg7_1
        del buf5
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
        # Source Nodes: [gate, min_1, offset_embedding_6], Original ATen: [aten.min, aten.mul, aten.sigmoid]
        triton_poi_fused_min_mul_sigmoid_4.run(buf9, arg12_1, arg2_1, arg10_1, 400, grid=grid(400), stream=stream0)
        del arg10_1
        del arg12_1
        del arg2_1
        return (buf4, buf9, )


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
    arg11_1 = 4
    arg12_1 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
