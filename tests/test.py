import scipy.special
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


def test_broadcast_tensor():
    x = torch.arange(3).view(1, 1, 3)
    y = torch.arange(2).view(2, 1)
    a, b = torch.broadcast_tensors(x, y)
    print("*"*20, "torch", "*"*20)
    print("x: ", x)
    print("y: ", y)
    print("a: ", a)
    print("b: ", b)

    x_hidet = hidet.from_torch(x)
    y_hidet = hidet.from_torch(y)
    a_hidet = ops.broadcast(x_hidet, [2, 3])
    b_hidet = ops.broadcast(y_hidet, [2, 3])
    print("*"*20, "hidet", "*"*20)
    print("a: ", a_hidet)
    print("b: ", b_hidet)

def test_broadcast_shape():
    # print(torch.broadcast_shapes([1,3], [2, 1]))
    x = torch.arange(3).view(1, 1, 3)
    y = torch.arange(2).view(2, 1)
    s = get_broadcast_shape(x, y)
    print(s)

from hidet import Tensor
def get_broadcast_shape(x: Tensor, y: Tensor):
    broadcase_shape = []
    x_shape = x.shape
    y_shape = y.shape
    dim = max(len(x_shape), len(y_shape))
    for i in reversed(range(dim)):
        x_shape_i = 1 if i > len(x_shape)-1 else x_shape[i]
        y_shape_i = 1 if i > len(y_shape)-1 else y_shape[i]
        assert x_shape_i == 1 or y_shape_i == 1
        broadcase_shape.append(x_shape_i if y_shape_i == 1 else y_shape_i)
    return list(reversed(broadcase_shape))
        
import scipy
import math

def test_torch_digamma():
    def digamma_1(x):
        if x == 0:
            return float('inf')
        elif x < 0:
            return digamma(1 - x) - math.pi / math.tan(math.pi * x)
        else:
            result = 0
            while x < 8:
                result -= 1 / x
                x += 1
            print("result = ", result)
            xx = 1 / (x * x)
            result += math.log(x) - 0.5 / x - xx * (1 / 12 - xx * (1 / 120 + xx / 252))
            return result
    
    # 0 < x < 8
    def compute_2(x):
        k = []
        for i in range(8 - x // 1):
            k.append(1 / (x + i))
        
        sum_k = sum(k)
        print("sum_k: ", sum_k)

        x_1 = 8 + x % 1
        xx = 1 / (x_1 * x_1)
        ret = -sum_k + math.log(x_1) - 0.5 / x_1 - xx * (1 / 12 - xx * (1 / 120 + xx / 252))
        return ret

    # x > 8
    def compute_3(x):
        xx = 1 / (x * x)
        ret = math.log(x) - 0.5 / x - xx * (1 / 12 - xx * (1 / 120 + xx / 252))
        return ret

    def compute_1(x):
        if x < 8:
            return compute_2(x)
        else:
            return compute_3(x)
            

    def digamma(x):
        # Handle special cases
        if x == 0:
            return float('-inf')
        elif x < 0:
            x_1 = 1 - x
            return compute_1(x_1) - math.pi / math.tan(math.pi * x)
        else:
            return compute_1(x)


    print(torch.digamma(torch.Tensor([2])))
    print(digamma(2))


def test_tan():
    # test_torch_digamma()
    x = torch.randn([2, 2]).cuda()
    
    x_hidet = hidet.from_torch(x)
    x_hidet = x_hidet.astype("f32")
    y = ops.tan(x_hidet)
    print(y)
    print(torch.tan(x))

def test_hidet_ir_util_broadcastshape():
    from hidet.ir.utils.broadcast_utils import broadcast_shapes
    s1 = [1, 2, 1]
    s2 = [1, 1, 4]
    s3 = [3, 1, 1]
    bs = broadcast_shapes([s1, s2, s3])
    print(bs)

def test_torch_Tensor_lgamma():
    def gamma(x, num_points=10000):
        # 定义积分函数
        def integrand(t):
            return t**(x-1) * math.exp(-t)

        # 梯形法则数值积分
        integral = 0
        dt = 50 / num_points  # 选择合适的积分步长
        for i in range(num_points):
            integral += integrand(i * dt + dt / 2) * dt

        return integral

    def lanczos_gamma(x):
        # Lanczos近似参数
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
        # if x < 0.5:
        #     return math.pi / (math.sin(math.pi * x) * lanczos_gamma(1 - x))
        # else:
        #     x -= 1
        #     a = p[0]
        #     for i in range(1, g + 2):
        #         a += p[i] / (x + i)
        #     t = x + g + 0.5
        #     return math.sqrt(2 * math.pi) * math.pow(t, (x + 0.5)) * math.exp(-t) * a

        # x < 0.5
        def compute_1(x):
            return math.pi / (math.sin(math.pi * x) * compute_2(1 - x))
        
        # x >= 0.5
        def compute_2(x):
            x = x - 1
            a = p[0]
            for i in range(1, g + 2):
                a += p[i] / (x + i)
            t = x + g + 0.5
            return math.sqrt(2 * math.pi) * math.pow(t, (x + 0.5)) * math.exp(-t) * a
        
        return compute_1(x) if x < 0.5 else compute_2(x)

    
    # x = torch.Tensor([-1.2])
    # print(x.lgamma())

    print(math.gamma(1.2))
    print(lanczos_gamma(1.2))


def test_reshape():
    def reshape(x, *shape):
        return ops.reshape(x, shape)

    x = torch.randn([2, 2, 2])
    y = x.reshape([2, -1])
    print(y)
    x_hidet = hidet.from_torch(x)
    y_hidet = ops.reshape(x_hidet, [2, -1])
    print(reshape(x_hidet, [2, -1]))

def main():
    test_reshape()

if __name__ == "__main__":
    main()

