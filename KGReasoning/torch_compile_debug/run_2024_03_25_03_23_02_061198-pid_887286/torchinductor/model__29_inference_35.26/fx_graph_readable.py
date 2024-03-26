class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1]", arg1_1: "Sym(s0)", arg2_1: "Sym(s1)", arg3_1: "f32[1, 1, s0, s1]", arg4_1: "Sym(s2)", arg5_1: "f32[1, s2, 1, s1]", arg6_1: "f32[1, s2, 1, s1]"):
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:414, code: delta = (entity_embedding - query_center_embedding).abs()
        sub: "f32[1, s2, s0, s1]" = torch.ops.aten.sub.Tensor(arg3_1, arg5_1);  arg3_1 = arg5_1 = None
        abs_1: "f32[1, s2, s0, s1]" = torch.ops.aten.abs.default(sub);  sub = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:415, code: distance_out = F.relu(delta - query_offset_embedding)
        sub_1: "f32[1, s2, s0, s1]" = torch.ops.aten.sub.Tensor(abs_1, arg6_1)
        relu: "f32[1, s2, s0, s1]" = torch.ops.aten.relu.default(sub_1);  sub_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:416, code: distance_in = torch.min(delta, query_offset_embedding)
        minimum: "f32[1, s2, s0, s1]" = torch.ops.aten.minimum.default(abs_1, arg6_1);  abs_1 = arg6_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:417, code: logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        abs_2: "f32[1, s2, s0, s1]" = torch.ops.aten.abs.default(relu);  relu = None
        pow_1: "f32[1, s2, s0, s1]" = torch.ops.aten.pow.Tensor_Scalar(abs_2, 1);  abs_2 = None
        sum_1: "f32[1, s2, s0]" = torch.ops.aten.sum.dim_IntList(pow_1, [-1]);  pow_1 = None
        pow_2: "f32[1, s2, s0]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 1.0);  sum_1 = None
        sub_2: "f32[1, s2, s0]" = torch.ops.aten.sub.Tensor(arg0_1, pow_2);  arg0_1 = pow_2 = None
        abs_3: "f32[1, s2, s0, s1]" = torch.ops.aten.abs.default(minimum);  minimum = None
        pow_3: "f32[1, s2, s0, s1]" = torch.ops.aten.pow.Tensor_Scalar(abs_3, 1);  abs_3 = None
        sum_2: "f32[1, s2, s0]" = torch.ops.aten.sum.dim_IntList(pow_3, [-1]);  pow_3 = None
        pow_4: "f32[1, s2, s0]" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 1.0);  sum_2 = None
        mul: "f32[1, s2, s0]" = torch.ops.aten.mul.Tensor(pow_4, 0.02);  pow_4 = None
        sub_3: "f32[1, s2, s0]" = torch.ops.aten.sub.Tensor(sub_2, mul);  sub_2 = mul = None
        return (sub_3,)
        