# from buddy import compiler
from flagir import compiler
import torch
import torch._dynamo as dynamo
# import logging
# torch._dynamo.config.log_level = logging.INFO
# torch._dynamo.config.output_code = True

def foo(x, y):
  return x + y

foo_mlir = dynamo.optimize(compiler.DynamoCompiler)(foo)
in1 = torch.randn(10)
in2 = torch.randn(10)
foo_mlir(in1, in2)
