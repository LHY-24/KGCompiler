import torch
import hidet
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.utils import compute, input_like, reduce
from hidet.ir import primitives as prim
from hidet.ir import expr
import math

class GammaTask(Task):
    def __init__(self, x: TensorNode):
        g = 7
        p = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ]
        def compute_1(x):
            return math.pi / (prim.sin(math.pi * x) * compute_2(1 - x))

        def compute_2(x):
            x = x - 1
            a_reduce = p[0] + p[1] / (x + 1) + p[2] / (x + 2) + p[3] / (x + 3) + p[4] / (x + 4) + p[5] / (x + 5) + p[6] / (x + 6) + p[7] / (x + 7) + p[8] / (x + 8)
            t = x + g + 0.5
            return prim.sqrt(2 * math.pi) * prim.pow(t, (x + 0.5)) * prim.exp(-t) * a_reduce

        def y_compute(*indices):
            x_i = x[indices]
            return expr.if_then_else(
                cond=expr.less_than(x_i, 0.5),
                then_expr=compute_1(x_i),
                else_expr=compute_2(x_i)
            )
        y = compute(
            name = "y",
            shape = x.const_shape,
            fcompute = y_compute
        )

        super().__init__(name='gamma', inputs=[x], outputs=[y], attributes={})
    

class GammaOp(Operator):
    def __init__(self, x: Tensor):
        super().__init__(
            inputs=[x], 
            attributes={}, 
            task=GammaTask(input_like(x, 'x'))
        )


def gamma(input):
    return GammaOp(input).outputs[0]

def lgamma(input):
    return ops.log(ops.abs(gamma(input)))

def test_gamma():
    x = 1.2
    y_math = math.gamma(x)
    print("*"*20, "math", "*"*20)
    print(y_math)
    x_hidet = hidet.asarray([x])
    y_hidet = gamma(x_hidet)
    print("*"*20, "hidet", "*"*20)
    print(y_hidet)


from hidet import ops
def test_lgamma():
    x_torch = torch.asarray([1.2, 2, 3.4])
    y_tensor = torch.lgamma(x_torch)
    print("*"*20, "torch", "*"*20)
    print(y_tensor)
    x_hidet = hidet.from_torch(x_torch)
    y_hidet = ops.log(ops.abs(gamma(x_hidet)))
    print("*"*20, "hidet", "*"*20)
    print(y_hidet)

if __name__ == "__main__":
    # test_gamma()
    test_lgamma()