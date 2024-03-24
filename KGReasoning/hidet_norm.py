from hidet.ir import primitives as prim
from hidet.ir.compute import reduce
from hidet.ir.expr import cast
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.utils import compute, input_like, normalize_dim
from typing import (
    List, Tuple, Optional, Union, Any, Sequence, TYPE_CHECKING
)
from hidet.graph import ops
import torch
import hidet

class KgNormTask(Task):
    """
        x [3, 4, 5] dim = 0 -> y[4, 5]
        # compute primitive
out = compute(
    name='hint_name',
    shape=[n1, n2, ..., nk],
    fcompute=lambda i1, i2, ..., ik: f(i1, i2, ..., ik)
)

# semantics
for i1 in range(n1):
  for i2 in range(n2):
    ...
      for ik in range(nk):
        out[i1, i2, ..., ik] = f(i1, i2, ..., ik)

# reduce primitive
out = reduce(
    name='hint_name',
    shape=[n1, n2, ..., nk],
    fcompute=lambda i1, i2, ..., ik: f(i1, i2, ..., ik)
    reduce_type='sum' | 'max' | 'min' | 'avg'
)

# semantics
values = []
for i1 in range(n1):
  for i2 in range(n2):
    ...
      for ik in range(nk):
        values.append(f(i1, i2, ..., ik))
out = reduce_type(values)

out[j][k] = sum_{i}(abs(in[i][j][k]))

out = 

for i:
    for j:
        
    """

    def __init__(self, x: TensorNode, p: float, dim: int, eps: float):
        x_shape = x.const_shape
        y_shape = x_shape
        dtype = x.type.dtype
        reduce_shape = []
        other_shape = []
        for idx, size in enumerate(x_shape):
            if idx == dim:
                reduce_shape.append(size)
            else:
                other_shape.append(size)

        def sum_compute(*indices):
            def sum_reduce(*reduction_axis):
                x_indices = []
                p = 0
                q = 0
                for i in range(len(x.shape)):
                    if i != dim:
                        x_indices.append(indices[p])
                        p += 1
                    else:
                        x_indices.append(reduction_axis[q])
                        q += 1
                assert p == len(indices) and q == len(reduction_axis)
                # Force fp32 reduction for accuracy
                return prim.abs(x[x_indices])

            return reduce(shape=reduce_shape, fcompute=sum_reduce, reduce_type='sum')

        y = compute(name='y', shape=other_shape, fcompute=sum_compute)

        """
        [I, J, K] reduce j
        for i in I
            for k in K
                values = []
                for j in J:
                    values.append(abs(x[i, j, k]))
                out[i][k] = sum(values)
        """

        # ops.reduce()
        # p_norm = compute(name='p_norm', shape=other_shape, fcompute=lambda *indices: prim.pow(sum_[indices], 1.0 / p))

        # def y_compute(*indices):
        #     norm_indices = [index for i, index in enumerate(indices) if i != dim]
        #     return cast(x[indices] / prim.max(p_norm[norm_indices], eps), dtype)


        # y = reduce()
        super().__init__(name='kg_norm', inputs=[x], outputs=[y], attributes={'p': p, 'dim': dim})


class KgNormOp(Operator):
    def __init__(self, x: Tensor, p: float, dim: int, eps: float):
        super().__init__(
            inputs=[x], attributes={'p': p, 'dim': dim, 'eps': eps}, task=KgNormTask(input_like(x, 'x'), p, dim, eps)
        )

'''
input: Any,
    p: float | str | None = "fro",
    dim: Any | None = None,
    keepdim: bool = False,
    out: Any | None = None,
    dtype: Any | None = None
'''

def kg_norm(input, p: Optional[Union[float, str]] = 1, dim=None, keepdim=False, out=None, dtype=None):
    """LP norm.

    Parameters
    ----------
    x: Tensor
        The data to be normalized.
    p: float
        The exponent value in the norm formulation.
    dim: int
        The dimension to reduce.
    eps: float
        Small value to avoid division by zero.

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    # Normalize dim
    dim = normalize_dim(dim, rank=len(input.shape))

    eps=1e-12
    return KgNormOp(input, p, dim, eps).outputs[0]

def test_norm():
    dim = 0
    x = torch.rand([3, 4, 5])
    print(x)
    y = torch.norm(x, p = 1, dim=dim)
    print("torch: ", y.shape)
    print(y)
    print("*"*99)

    x_hidet = hidet.from_torch(x)
    y_hidet = kg_norm(x_hidet, p = 1, dim=dim)
    print("hidet: ", y_hidet.shape)
    print(y_hidet)

test_norm()