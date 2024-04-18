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
from utils import serialize_output, deserialize_output, resolve_save_dir_multigraph
from dynamo_config import dynamo_config


def custom_hidet_backend(graph_module, example_inputs):
    from hidet import Tensor
    from interpreter import Interpreter
    from utils import symbol_like_torch

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
