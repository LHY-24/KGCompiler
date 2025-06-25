class Context:
  def __init__(self):
    self.indent = 0
    self.curRegion = None

  def setCurRegion(self, region):
    self.curRegion = region


class Region:
  def __init__(self, name, ctx):
    self.name = name
    self.indent = ctx.indent
    self.ops = []

  def addOps(self, op):
    self.ops.append(op)

  def __enter__(self):
    ctx.indent += 2
    ctx.curRegion = self
    return self

  def __exit__(self, exc_type, exc_value, tb):
    ctx.indent -= 2
  
  def dump(self):
    print("{{")
    for op in self.ops:
      for i in range(op.indent):
        print(" ", end = '')
      op.dump()
    print("}}")

class InputOperation:
  def __init__(self, ctx, name, type):
    self.name = name
    self.indent = ctx.indent
    self.shape = type.shape
    self.dtype = type.dtype
    ctx.curRegion.addOps(self)

  def getShapeStr(self):
    return ', '.join(str(ele) for ele in self.shape)
  
  def dump(self):
    shape = self.getShapeStr()
    print(f"INPUT {self.name}<{shape}>: Tensor.{self.dtype.name}")

class MapOperation:
  def __init__(self, ctx, name, *args, **kwargs):
    self.indent = ctx.indent
    self.name = name
    self.argList = args
    self.argDict = kwargs
    self.shape = self.getResShape()
    self.dtype = self.getResDtype()
    ctx.curRegion.addOps(self)

  def getResShape(self):
    # TODO: handle different cases for the result shape inference.
    return self.argList[0].shape
  
  def getResDtype(self):
    # TODO: handle different cases for the result shape inference.
    return self.argList[0].dtype

  def getArgShapeStr(self, i):
    shape = self.argList[i].shape
    return ', '.join(str(ele) for ele in shape)
  
  def getResShapeStr(self):
    return ', '.join(str(ele) for ele in self.shape)
  
  def getAttrStr(self):
    return self.argDict['mapper']
  
  def getArgsNameListStr(self):
    nameList = ''
    for arg in self.argList:
      nameList += arg.name
      nameList += " "
    return nameList


  def dump(self):
    resultShape = self.getResShapeStr()
    resultDtype = self.dtype.name
    resultSize = 1
    argSize = len(self.argList)
    argsDtype = self.argList[0].dtype.name
    argsShape = self.getArgShapeStr(0)
    argsNameList = self.getArgsNameListStr()
    mapperName = self.getAttrStr()
    print(f"{self.name}<{resultShape}>: {resultDtype} = map.i{argSize}o{resultSize}.{argsDtype}<{argsShape}> {argsNameList} !mapper={mapperName}")

class OutputOperation:
  def __init__(self, ctx, arg):
    self.indent = ctx.indent
    self.name = arg.name
    ctx.curRegion.addOps(self)
  
  def dump(self):
    print(f"OUTPUT {self.name}")

class Type:
  def __init__(self, name):
    self.name = name

class F32Type(Type):
  def __init__(self):
    super().__init__("f32")

class TensorType(Type):
  def __init__(self, shape, dtype):
    super().__init__("Tensor")
    self.shape = shape
    self.dtype = dtype

class Operation:
  def __init__(self, name, inputs, output):
    self.name = name
    self.inputs = inputs
    self.output = output

if __name__ == "__main__":
  ctx = Context()
  region = Region("top", ctx)
  with region:
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
