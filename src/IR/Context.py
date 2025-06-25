class Context:
  def __init__(self):
    self.indent = 0
    self.curRegion = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass

  def setCurRegion(self, region):
    self.curRegion = region