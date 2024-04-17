import torch
import hidet
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.utils import compute, input_like, reduce
from hidet.ir import primitives as prim
from hidet.ir import expr
import math

class DigammaTask(Task):
    def __init__(self, x: TensorNode):
        datatype = x.type.dtype
        # x > 0
        def compute_fun(x):
            return expr.if_then_else(
                cond = expr.less_than(x, 8),
                then_expr = compute_func_1(x),
                else_expr = compute_func_2(x)
            )

        # 0 < x < 8
        def compute_func_1(x):
            # k = [1/x, 1/(x+1), ...]
            k = compute(
                name = "k",
                shape = [8 - x // 1], 
                fcompute = lambda i : 1 / (x + i)
            )
            # sum_k = 1 / (x) + 1 / (x + 1) + ...
            sum_k = reduce(
                name = "sum_k",
                shape = k.shape,
                fcompute = lambda i : k[i],
                reduce_type = "sum"
            )

            # while x < 8: x += 1
            x_1 = 8 + x % 1
            x_1 = expr.cast(x_1, datatype)
            xx = 1 / (x_1 * x_1)
            ret = -sum_k + prim.log(x_1) - 0.5 / x_1 - xx * (1 / 12 - xx * (1 / 120 + xx / 252))
            return ret
        
        # x >= 8
        def compute_func_2(x):
            xx = 1 / (x * x)
            ret = prim.log(expr.cast(x, datatype)) - 0.5 / x - xx * (1 / 12 - xx * (1 / 120 + xx / 252))
            return ret

        def x_lessthan_0_then_part(x):
            x_1 = 1 - x
            return compute_fun(x_1) - math.pi / prim.tan(math.pi * x)
            
        def x_lessthan_0_else_part(x):
            return compute_fun(x)


        def digamma_compute(*indices):
            x_i = x[indices]
            return expr.if_then_else(
                cond=expr.equal(x_i, 0),
                then_expr=expr.cast(-99999999999, datatype),
                else_expr=expr.if_then_else(
                    cond=expr.less_than(x_i, 0),
                    then_expr=x_lessthan_0_then_part(x_i),
                    else_expr=x_lessthan_0_else_part(x_i)
                )
            )
        
        y = compute(
            name = "y",
            shape = x.const_shape,
            fcompute = digamma_compute
        )

        super().__init__(name='digamma', inputs=[x], outputs=[y], attributes={})
    

class DigammaOp(Operator):
    def __init__(self, x: Tensor):
        super().__init__(
            inputs=[x], 
            attributes={}, 
            task=DigammaTask(input_like(x, 'x'))
        )


def digamma(input):
    return DigammaOp(input).outputs[0]

def test_digamma():
    x = torch.Tensor([7, 8, 9]).double()
    y = torch.digamma(x)
    print("*"*20, "torch", "*"*20)
    print(y)
    x_hidet = hidet.from_torch(x)
    y_hidet = digamma(x_hidet)
    print("*"*20, "hidet", "*"*20)
    print(y_hidet)


if __name__ == "__main__":
    test_digamma()