import torch
from typing import List

def DynamoCompiler(gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
  print("Custom Compiler from FX Graph to FlagIR:")
  print("-------------------------------------------------------------------")
  gm.graph.print_tabular()
  print("-------------------------------------------------------------------")
  # Initialize the MLIR context.
  with Context():
    module = Importer(gm, inputs)
    # module = Lowering(module)
  return gm.forward

def Importer(gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
  # Initialize the symbol table.
  symbolTable = {}
  # Create a region.
  with RegionBlock():
    arguments = []
    for arg in inputs:
      shapeList = list(arg.shape)
      tensorArg = TensorConst.get(shapeList)
    with FunctionBlock():
      for node in gm.graph.nodes:
        CodeGen(node, symbolTable, arguments)

def CodeGen(node, symbolTable, argsList):
  if node.op == "placeholder" :
    # Bind the placeholder with args into symbol table.
    pass
  if node.op == "call_function" :
    # Parse a call_function operation.
    if node.target.__name__ == "add":
      # Get operands from symbol table.
      input1 = symbolTable.get(str(node._args[0]))
      input2 = symbolTable.get(str(node._args[1]))
      # Generate add operation.
      
      op = AddOpGen(input1, input2)
      # Register into the symbol table.
      symbolTable[str(node.name)] = op
  if node.op == "output" :
    # Generating return operation.
    # ret = symbolTable.get(str(node._args[0][0]))
    outputList = []
    op = OutOp(outputList)
    # Register into the symbol table.
    # symbolTable["output"] = op


class Context():
  def __init__(self):
    pass
     
  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_value, tb):
    pass

class RegionBlock():
  def __init__(self):
    pass
     
  def __enter__(self):
    print("{{")

  def __exit__(self, exc_type, exc_value, tb):
    print("}}")

class FunctionBlock():
  def __init__(self):
    pass
     
  def __enter__(self):
    print("  fn = {")

  def __exit__(self, exc_type, exc_value, tb):
    print("  }")

class TensorConst():
  def __init__(self):
    pass

  def get(type):
    print("  x0 = tensor.const.f32<10>")

class AddOpGen():
  def __init__(self, lhs, rhs):
    operands = []
    operands.append(lhs)
    operands.append(rhs)
    results = []
    emptyRes = 0
    results.append(emptyRes)
    # BuildMapOp(operands, results, "add")
    print("    v0<10>:f32 = map.i2o1.f32<10> x0 x1 !mapper=add")

class OutOp():
  def __init__(self, operands):
    print("    OUT v0")


class Context:
  """
  1 - Symbol Table Management: The Context class keeps track of all the symbols 
      defined in the program and their associated attributes (such as data type,
      scope, and memory location). This information is stored in a symbol table,
      which is a data structure that maps symbol names to their corresponding
      attributes.

  2 - Scoping: The Context class maintains the current scope of the program 
      being compiled. Scoping rules define the visibility and accessibility of 
      symbols within a program, and the Context class enforces these rules by 
      ensuring that symbols are only accessible within their appropriate scopes.

  3 - Type Checking: The Context class is responsible for checking the 
      compatibility of types between different symbols and operations in the 
      program. This ensures that operations are performed on the correct types 
      of data and can detect errors such as type mismatches.

  4 - Code Generation: The Context class can generate code that reflects the 
      information stored in the symbol table and the current scope. This code 
      can be used to produce the final output of the compiler, such as an 
      executable program or an intermediate language like LLVM.
  """
  def __init__(self):
    self.symbol_table = {}  # symbol table mapping names to attributes
    self.scope_stack = []   # stack of nested scopes
    self.code = []          # list of generated code instructions

  def enter_scope(self):
    self.scope_stack.append({})  # add new scope to the stack

  def exit_scope(self):
    self.scope_stack.pop()      # remove current scope from the stack

  def add_symbol(self, name, attrs):
    self.symbol_table[name] = attrs  # add symbol to the current scope

  def get_symbol(self, name):
    for scope in reversed(self.scope_stack):  # search scopes from inner to outer
      if name in scope:
        return scope[name]
    raise Exception(f"Symbol '{name}' not found")  # symbol not found in any scope

  def generate_code(self, instr):
    self.code.append(instr)  # add instruction to the code list

  def check_type(self, type1, type2):
    # check if types are compatible, raise exception if not
    if type1 != type2:
      raise Exception(f"Type mismatch: {type1} and {type2}")

  def generate_var(self, name, type):
    # generate code to allocate memory for a variable
    self.generate_code(f"{name} = allocate_memory({type})")

  def generate_assign(self, target, value):
    # generate code to assign a value to a variable
    self.generate_code(f"{target} = {value}")

  def generate_op(self, op, operands, result):
    # generate code for an operation with one or more operands
    self.generate_code(f"{result} = {op}({', '.join(operands)})")

  def generate_call(self, func, args, result):
    # generate code for a function call with arguments
    self.generate_code(f"{result} = {func}({', '.join(args)})")


class Region:
    def __init__(self, name):
        self.name = name

class Operation:
    def __init__(self, name, inputs, output):
        self.name = name
        self.inputs = inputs
        self.output = output

class Type:
    def __init__(self, name):
        self.name = name

class Value:
    def __init__(self, name, type):
        self.name = name
        self.type = type


# Symbol Table
# Value number 

class Context:
  def __init__(self):
    self.symbol_table = {}
    self.value = {}

  def add_symbol(self, name, data_type, scope):
    if name in self.symbol_table:
      raise Exception(f"Symbol '{name}' already defined in current scope")
    self.symbol_table[name] = {'data_type': data_type, 'scope': scope}

  def get_symbol(self, name):
      return self.symbol_table.get(name, None)

  def generate_unique_id(self, prefix):
      # Generate a unique identifier with the given prefix
      # (e.g. "temp_1", "temp_2", etc.)
      # The implementation of this method is dependent on the compiler's needs
      pass

class Region:
  def __init__(self, name):
    self.name = name
    self.values = []

  def add_value(self, value):
    self.values.append(value)
  
  def dump():
    pass

class Value:
  def __init__(self, name, op):
    self.name = name
    self.op = op

  def dump():
    pass

class Operation:
  def __init__(self, name, inputs, output):
    self.name = name
    self.inputs = inputs
    self.output = output

class Type:
  def __init__(self, base, shape):
    self.base = base
    self.shape = shape

class Attribute:
  def __init__(self, name, attr):
    self.name = name
    self.attr = attr
