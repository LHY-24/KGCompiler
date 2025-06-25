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
