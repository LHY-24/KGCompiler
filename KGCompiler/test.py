from flagir.Context import Context
from flagir.Region import Region
from flagir.Operation import *
from flagir.Type import F32Type, TensorType

if __name__ == "__main__":
  with Context() as ctx:
    with Region("top", ctx) as region:
      name1 = "x"
      name2 = "y"
      shape = [10]
      f32 = F32Type()
      tensorType = TensorType(shape, f32)
      input1 = InputOperation(ctx, name1, tensorType)
      input2 = InputOperation(ctx, name2, tensorType)
      add = MapOperation(ctx, "z", input1, input2, mapper="add")
      mul = MapOperation(ctx, "t", input1, input2, mapper="mul")
      max = MapOperation(ctx, "w", add, mul, mapper="max")
      output = OutputOperation(ctx, max)
  region.dump()
