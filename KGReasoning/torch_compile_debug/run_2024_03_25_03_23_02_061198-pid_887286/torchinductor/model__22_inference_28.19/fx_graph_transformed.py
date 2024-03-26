class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[14505, 400]", arg1_1: "f32[474, 400]", arg2_1: "f32[474, 400]", arg3_1: "Sym(s0)", arg4_1: "i64[1, s0]"):
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:216, code: offset_embedding = torch.zeros_like(embedding).cuda()
        full_default: "f32[1, 400]" = torch.ops.aten.full.default([1, 400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:214, code: embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
        select: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 0)
        index: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg0_1, [select]);  arg0_1 = select = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:226, code: r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
        select_1: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 1)
        index_1: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg1_1, [select_1]);  select_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:228, code: embedding += r_embedding
        add: "f32[1, 400]" = torch.ops.aten.add.Tensor(index, index_1);  index = index_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:226, code: r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
        select_3: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 2)
        index_3: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg1_1, [select_3]);  select_3 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:228, code: embedding += r_embedding
        add_2: "f32[1, 400]" = torch.ops.aten.add.Tensor(add, index_3);  add = index_3 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:226, code: r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
        select_5: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 3)
        index_5: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg1_1, [select_5]);  arg1_1 = select_5 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:228, code: embedding += r_embedding
        add_4: "f32[1, 400]" = torch.ops.aten.add.Tensor(add_2, index_5);  add_2 = index_5 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:227, code: r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
        select_2: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 1)
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:229, code: offset_embedding += self.func(r_offset_embedding)
        index_2: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg2_1, [select_2]);  select_2 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:227, code: r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
        select_4: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 2)
        index_4: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg2_1, [select_4]);  select_4 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:229, code: offset_embedding += self.func(r_offset_embedding)
        add_3: "f32[1, 400]" = torch.ops.aten.add.Tensor(index_2, index_4);  index_2 = index_4 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:227, code: r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
        select_6: "i64[1]" = torch.ops.aten.select.int(arg4_1, 1, 3);  arg4_1 = None
        index_6: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg2_1, [select_6]);  arg2_1 = select_6 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:229, code: offset_embedding += self.func(r_offset_embedding)
        add_5: "f32[1, 400]" = torch.ops.aten.add.Tensor(add_3, index_6);  add_3 = index_6 = None
        return (add_4, add_5)
        