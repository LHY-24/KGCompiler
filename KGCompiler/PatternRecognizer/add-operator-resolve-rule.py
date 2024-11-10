"""

Add Operator Resolve Rule
=========================

This is a tutorial introduces the `operator resolving mechanism` and how to add a new operator resolve rule. An operator
resolve rule is used to resolve an operator to other operators. Usually, we would resolve a more generic operator to
more specific and efficient operators. The operator resolving rules allow us to reuse existing highly-optimized
operators to implement a new operator, while organizing the operators in a more modular way.


Operator Resolving
------------------

The core idea of the **operator resolving** is to resolve a generic operator to more specific and high-optimized
operators. When we define a new operator, we can also attach an operator resolve rule to it. The rule defines how to
resolve the operator to other operators with the same semantics. After the operator is resolved, the original operator
will be replaced by the resolved operators. This process is transparent to the user and is done automatically by a pass
when we optimize a flow graph.

There are typical two scenarios that we need to resolve an operator to other operators:

- **Resolve a generic operator to specialized variants**: We can provide a generic operator and lots of its specialized
  variants. When optimizing the model, we can resolve the generic operator to the most suitable specialized operator.
  For example, in KGCompiler, we provided a generic :py:func:`~hidet.ops.matmul` operator with the same semantics as
  the numpy equivalent :py:data:`numpy.matmul`. This operator is a generic operator and is scheduled automatically by
  our auto-scheduler, thus it is not very efficient. But we also provided a lot of specialized variants of the operators
  such as highly-optimized :py:func:`~hidet.ops.batch_matmul` that only accepts :math:`A=[B, M, K]` and
  :math:`B=[B, K, N]`. During the operator resolving, we first reshape and broadcast the input tensors to align the
  input shapes with the specialized operator, then use the specialized operator to compute the result, and finally
  reshape the output tensor to get the correct output shape.


Add Operator Resolve Rule
-------------------------

To add a resolve rule to an operator, we need to

#. define a subclass of :class:`~hidet.graph.transforms.resolve_variant.ResolveRule` and then
#. register the rule by decorating it with :func:`~hidet.graph.transforms.resolve_variant.register_resolve_rule`.

In the following example, we resolve the :func:`~hidet.ops.pow` operator to normal multiplications if the exponent
is a constant integer 3.

Before we start, let's have a look at the original behavior when there is no such resolve rule.
"""
import hidet

a = hidet.symbol(shape=[2, 3], device='cuda')
b = hidet.ops.pow(a, hidet.asarray(3, device='cuda'))
graph = hidet.trace_from(b, inputs=[a])
print('Original graph:')
print(graph)

print('Optimized graph without resolving Pow:')
graph_opt = hidet.graph.optimize(graph)
print(graph_opt)

# %%
# The original graph contains a :func:`~hidet.ops.pow` operator, and the optimized graph is the same as the
# original graph. Now let's add the resolve rule and see what happens.

from typing import Optional, List
from hidet import Tensor
from hidet.graph.ops.arithmetic import PowOp
from hidet.graph.transforms import register_resolve_rule, ResolveRule


@register_resolve_rule(PowOp)
class PowResolveRule(ResolveRule):
    def resolve(self, op: PowOp) -> Optional[List[Tensor]]:
        a: Tensor = op.inputs[0]  # get the base tensor
        b: Tensor = op.inputs[1]  # get the exponent tensor
        if not b.is_symbolic() and len(b.shape) == 0 and int(b) == 3:
            # if the exponent is a constant integer 3, resolve the operator to a * a * a
            return [a * a * a]
        # otherwise, return None to indicate that the operator cannot be resolved
        # and the original operator will be kept
        return None


# optimize the original graph again
# the Pow operator will be resolved to a * a * a
# after that, the two multiplications will be fused into one operator
graph_opt_new = hidet.graph.optimize(graph)
print('Optimized graph after resolving Pow:')
print(graph_opt_new)