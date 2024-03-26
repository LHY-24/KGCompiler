class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[14505, 400]", arg1_1: "f32[474, 400]", arg2_1: "f32[474, 400]", arg3_1: "f32[1]", arg4_1: "i64[1, 2]", arg5_1: "i64[1, 14505]"):
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:216, code: offset_embedding = torch.zeros_like(embedding).cuda()
        full_default: "f32[1, 400]" = torch.ops.aten.full.default([1, 400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:474, code: negative_sample_regular = negative_sample[all_idxs]
        _tensor_constant0: "i64[1]" = self._tensor_constant0
        full_default_1: "i64[1]" = torch.ops.aten.full.default([1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_3: "i64[1, 14505]" = torch.ops.aten.index.Tensor(arg5_1, [full_default_1]);  arg5_1 = full_default_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:476, code: negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
        view: "i64[14505]" = torch.ops.aten.reshape.default(index_3, [-1]);  index_3 = None
        index_4: "f32[14505, 400]" = torch.ops.aten.index.Tensor(arg0_1, [view]);  view = None
        view_1: "f32[1, 14505, 400]" = torch.ops.aten.reshape.default(index_4, [1, 14505, -1]);  index_4 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:214, code: embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
        select: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 0)
        index: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg0_1, [select]);  arg0_1 = select = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:226, code: r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
        select_1: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 1)
        index_1: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg1_1, [select_1]);  arg1_1 = select_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:228, code: embedding += r_embedding
        add: "f32[1, 400]" = torch.ops.aten.add.Tensor(index, index_1);  index = index_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:442, code: all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        unsqueeze: "f32[1, 1, 400]" = torch.ops.aten.unsqueeze.default(add, 1);  add = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:414, code: delta = (entity_embedding - query_center_embedding).abs()
        sub: "f32[1, 14505, 400]" = torch.ops.aten.sub.Tensor(view_1, unsqueeze);  view_1 = unsqueeze = None
        abs_1: "f32[1, 14505, 400]" = torch.ops.aten.abs.default(sub);  sub = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:227, code: r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
        select_2: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 1);  arg4_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:229, code: offset_embedding += self.func(r_offset_embedding)
        index_2: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg2_1, [select_2]);  arg2_1 = select_2 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:443, code: all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        unsqueeze_1: "f32[1, 1, 400]" = torch.ops.aten.unsqueeze.default(index_2, 1);  index_2 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:415, code: distance_out = F.relu(delta - query_offset_embedding)
        sub_1: "f32[1, 14505, 400]" = torch.ops.aten.sub.Tensor(abs_1, unsqueeze_1)
        relu: "f32[1, 14505, 400]" = torch.ops.aten.relu.default(sub_1);  sub_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:417, code: logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        abs_2: "f32[1, 14505, 400]" = torch.ops.aten.abs.default(relu);  relu = None
        sum_1: "f32[1, 14505]" = torch.ops.aten.sum.dim_IntList(abs_2, [-1]);  abs_2 = None
        pow_2: "f32[1, 14505]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 1.0);  sum_1 = None
        sub_2: "f32[1, 14505]" = torch.ops.aten.sub.Tensor(arg3_1, pow_2);  arg3_1 = pow_2 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:416, code: distance_in = torch.min(delta, query_offset_embedding)
        minimum: "f32[1, 14505, 400]" = torch.ops.aten.minimum.default(abs_1, unsqueeze_1);  abs_1 = unsqueeze_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:417, code: logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        abs_3: "f32[1, 14505, 400]" = torch.ops.aten.abs.default(minimum);  minimum = None
        sum_2: "f32[1, 14505]" = torch.ops.aten.sum.dim_IntList(abs_3, [-1]);  abs_3 = None
        pow_4: "f32[1, 14505]" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 1.0);  sum_2 = None
        mul: "f32[1, 14505]" = torch.ops.aten.mul.Tensor(pow_4, 0.02);  pow_4 = None
        sub_3: "f32[1, 14505]" = torch.ops.aten.sub.Tensor(sub_2, mul);  sub_2 = mul = None
        return (sub_3,)
        