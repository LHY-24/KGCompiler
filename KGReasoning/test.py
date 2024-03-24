import torch
import hidet

from hidet.graph import ops

# print graph
from torch._dynamo import optimize
from torch._inductor import graph
class CustomGraphLowering(graph.GraphLowering):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        *args,
        **kwargs,
    ):
        super().__init__(gm, *args, **kwargs)
        print("*"*30 + "Print Fx Graph" + "*"*30)
        gm.graph.print_tabular()
graph.GraphLowering = CustomGraphLowering

def test_take(input, dim, index):
    a = hidet.from_torch(input)
    b = hidet.from_torch(index)
    c = ops.transform.take(a, b, axis=dim)
    return c

def test_index_select():
    input = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    index = torch.as_tensor([0], dtype=torch.int32)
    dim = 1
    b = torch.index_select(input, dim, index)
    print(b)

    print(test_take(input, dim, index))

def test_concat():
    x = torch.rand([3, 4, 5])
    y = torch.rand([0])
    print(torch.cat([x, y]))

    x = hidet.randn([3, 4, 5])
    y = hidet.randn([0])
    print(ops.concat([x, y], axis = 0))

def my_decorator(func):
    def warper():
        print("call function ")
        func()


def test_norm():
    dim = 0
    x = torch.rand([3, 4, 5])
    print(x)
    y = torch.norm(x, p = 1, dim=dim)
    print("torch: ", y.shape)
    print(y)
    print("*"*99)

    x_hidet = hidet.from_torch(x)
    y_hidet = ops.normalize.lp_norm(x_hidet, p = 1, dim=dim)
    print("hidet: ", y_hidet.shape)
    print(y_hidet)

test_norm()


