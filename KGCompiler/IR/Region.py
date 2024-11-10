class Region:
  def __init__(self, name, ctx):
    self.name = name
    self.indent = ctx.indent
    self.ops = []
    self.ctx = ctx

  def addOps(self, op):
    self.ops.append(op)

  def __enter__(self):
    self.ctx.indent += 2
    self.ctx.curRegion = self
    return self

  def __exit__(self, exc_type, exc_value, tb):
    self.ctx.indent -= 2
  
  def dump(self):
    print("{{")
    for op in self.ops:
      for i in range(op.indent):
        print(" ", end = '')
      op.dump()
    print("}}")