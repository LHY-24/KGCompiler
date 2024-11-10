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