import torch
from torch.fx import symbolic_trace

import hidet
from hidet import Tensor
from hidet.graph.transforms import register_resolve_rule, ResolveRule
from hidet.graph.ops.arithmetic import PowOp
from hidet.graph.frontend.torch.dynamo_backends import hidet_backend

from typing import Optional, List

from typing import List, Callable, Sequence, Union
import logging
import torch
import hidet.option
from hidet.ir.type import DataType
from hidet.ir.expr import SymbolVar
from hidet.graph.flow_graph import FlowGraph
from hidet.ir import dtypes
from interpreter import Interpreter
from utils import serialize_output, symbol_like_torch, deserialize_output, resolve_save_dir_multigraph
from dynamo_config import dynamo_config


logger = logging.getLogger(__name__)


def custom_hidet_backend(graph_module, example_inputs):

    assert isinstance(graph_module, torch.fx.GraphModule)

    logger.info('received a subgraph with %d nodes to optimize', len(graph_module.graph.nodes))
    logger.debug('graph: %s', graph_module.graph)

    if dynamo_config['print_input_graph']:
        graph_module.print_readable()
        print('---')
        graph_module.graph.print_tabular()

    # get the interpreter for the subgraph
    interpreter: Interpreter = hidet.frontend.from_torch(graph_module)

    # prepare dummy and symbolic inputs for correctness and flow graph construction
    inputs: List[Union[Tensor, SymbolVar, int, bool, float]] = []  # for flow graph construction
    for example_input in example_inputs:
        if isinstance(example_input, torch.Tensor):
            symbolic_input = symbol_like_torch(example_input)
            inputs.append(symbolic_input)
        elif isinstance(example_input, (int, bool, float)):
            inputs.append(example_input)
        elif isinstance(example_input, torch.SymInt):
            from torch.fx.experimental.symbolic_shapes import SymNode

            node: SymNode = example_input.node
            try:
                inputs.append(node.pytype(example_input))
            except RuntimeError:
                # is a symbolic scalar input
                pytype2dtype = {int: dtypes.int32, float: dtypes.float32, bool: dtypes.boolean}
                inputs.append(hidet.symbol_var(name=str(example_input), dtype=pytype2dtype[node.pytype]))
        else:
            raise ValueError(f'hidet_backend: unexpected example input {example_input}, type {type(example_input)}')

    logger.info('hidet:   inputs: ')
    for arg in inputs:
        if isinstance(arg, hidet.Tensor):
            logger.info('hidet:   %s', arg.signature())
        else:
            logger.info('hidet:   %s', arg)

    # symbolic run to get flow graph
    output = interpreter(*inputs)
    output_format, output_tensors = serialize_output(output)
    input_tensors = [x for x in inputs if isinstance(x, hidet.Tensor)]
    flow_graph: FlowGraph = hidet.trace_from(output_tensors, inputs=input_tensors)
    
    return flow_graph

        
class Module1(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, a):
        
		return torch.pow(a, 3)

torch_model = Module1().cuda()
print(type(torch_model))

x = torch.Tensor([1,2,3]).float().cuda()	
symbolic_traced: torch.fx.GraphModule = symbolic_trace(torch_model)
# print(symbolic_traced.graph, end="\n\n")

hidet_model = custom_hidet_backend(symbolic_traced, x)
print(hidet_model, end="\n\n")
#   original:     <function hidet_backend.<locals>.wrapper>
#   now:          <class 'hidet.graph.flow_graph.FlowGraph'>    


# class MyModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin = torch.nn.Linear(10, 10)		# error: (100,10)???

#     def forward(self, x):
#         return torch.nn.functional.relu(self.lin(x))

# model = MyModule().cuda()
# x = torch.randn(10, 10).cuda()		# error: (10,100)???
# symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)
# print(symbolic_traced.graph, end="\n\n")
# hidet_model = hidet_backend(symbolic_traced, x)
# # print(type(hidet_model))		# <class 'hidet.graph.flow_graph.FlowGraph'>

# a = hidet.symbol(shape=[2, 3], device='cuda')
# b = hidet.ops.pow(a, hidet.asarray(3, device='cuda'))
# graph = hidet.trace_from(b, inputs=[a])
# print('Original graph:')
# print(type(graph))				# <class 'hidet.graph.flow_graph.FlowGraph'>
# print(graph, end="\n\n")


print('Optimized graph without resolving Pow:')
graph = hidet_model
graph_opt = hidet.graph.optimize(graph)
print(graph_opt, end="\n\n")
# print(type(graph_opt))


@register_resolve_rule(PowOp)
class PowResolveRule(ResolveRule):
    def resolve(self, op: PowOp) -> Optional[List[Tensor]]:
        a: Tensor = op.inputs[0]  # get the base tensor
        b: Tensor = op.inputs[1]  # get the exponent tensor
        if not b.is_symbolic() and len(b.shape) == 0 and int(b) == 3:
            # if the exponent is a constant integer 3, resolve the operator to a * a * a
            return [a * a * a]
        # otherwise, return None to indicate that the operator cannot be resolved and the original operator will be kept
        return None


# Optimize the original graph again
# the Pow operator will be resolved to a * a * a
# after that, the two multiplications will be fused into one operator
print('Optimized graph after resolving Pow:')
graph_opt_new = hidet.graph.optimize(graph)
print(graph_opt_new, end="\n\n")
