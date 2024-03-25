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
        same as torch.norm
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
                _p = 0
                q = 0
                for i in range(len(x.shape)):
                    if i != dim:
                        x_indices.append(indices[_p])
                        _p += 1
                    else:
                        x_indices.append(reduction_axis[q])
                        q += 1
                assert _p == len(indices) and q == len(reduction_axis)

                return prim.pow(prim.abs(x[x_indices]), p)
            y_sum = reduce(shape=reduce_shape, fcompute=sum_reduce, reduce_type='sum')
            if p == 0 or p == 1:
                return y_sum
            return prim.pow(y_sum, 1/p)

        y = compute(name='y', shape=other_shape, fcompute=sum_compute)

        """
        [I, J, K] reduce j
        for i in I
            for k in K
                values = []
                for j in J:
                    values.append(abs(x[i, j, k]) ** p)
                y_sum = sum(values)
                out[i][k] = y_sum ** (1/p)
        """
        super().__init__(name='kg_norm', inputs=[x], outputs=[y], attributes={'p': p, 'dim': dim})


class KgNormOp(Operator):
    def __init__(self, x: Tensor, p: float, dim: int, eps: float):
        super().__init__(
            inputs=[x], attributes={'p': p, 'dim': dim, 'eps': eps}, task=KgNormTask(input_like(x, 'x'), p, dim, eps)
        )


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
    p = 1
    dim = 0
    x = torch.rand([3, 4, 5])
    print("x: ", x.shape)
    print(x)
    print("*"*99)
    y = torch.norm(x, p = p, dim=dim)
    print("torch: ", y.shape)
    print(y)
    print("*"*99)

    x_hidet = hidet.from_torch(x)
    y_hidet = kg_norm(x_hidet, p = p, dim=dim)
    print("hidet: ", y_hidet.shape)
    print(y_hidet)

if __name__ == "__main__":
    test_norm()