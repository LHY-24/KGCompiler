from hidet.graph.transforms import registered_rewrite_rules, clear_registered_rewrite_rules
from hidet.graph.ops.matmul import MatmulOp
from hidet.graph.ops.transform import ConcatOp
from hidet.graph.transforms import TensorPattern, SubgraphRewriteRule
from hidet.graph.transforms import op_pattern, register_rewrite_rule
from hidet.utils import same_list

print('Registered rewrite rules:')
for rule in registered_rewrite_rules():
    assert isinstance(rule, SubgraphRewriteRule)
    print(rule.name)

# Registered rewrite rules:
# a + x => x + a
# x - a => x + (-a)
# (x + a) + b => x + (a + b)
# (x + a) * b => x * b + a * b
# (x + a) + (y + b) => (x + y) + (a + b)
# reshape(x) * scale
# reshape(x) + bias
# squeeze(x) * c => squeeze(x * c)
# y1 = cast(x), y2 = cast(x) => y1 = y2 = z = cast(x)
# y1,2,3 = cast(x) => y1 = y2 = y3 = z = cast(x)
# cast(cast(x)) => x
# binaryOp(unaryOp_left(x), unaryOp_right(x)) => compositeOp(x)
# binaryOp(unaryOp(x), x) => compositeOp(x)
# binaryOp(x, unaryOp(x)) => compositeOp(x)
# conv2d(x, w) * scale => conv2d(x, w * scale)
# conv2d(x, w1)|conv2d(x, w2)|conv2d(x, w3) => conv2d(x, w1 + w2 + w3)
# conv2d(x, w1)|conv2d(x, w2) => conv2d(x, w1 + w2)
# 3 branches of matmul(x, branch c) + branch b ==> matmul(x, c) + b followed by split
# matmul(x, c1)|matmul(x, c2)|matmul(x, c3) => matmul(x, concat(c1, c2, c3)) followed by split
# matmul(x, c1)|matmul(x, c2) ==> matmul(x, concat(c1, c2)) followed by split