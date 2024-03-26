class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[14505, 400]", arg1_1: "f32[474, 400]", arg2_1: "f32[474, 400]", arg3_1: "Sym(s0)", arg4_1: "Sym(s1)", arg5_1: "i64[s0, s1]"):
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:214, code: embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
        select: "i64[s0]" = torch.ops.aten.select.int(arg5_1, 1, 0)
        index: "f32[s0, 400]" = torch.ops.aten.index.Tensor(arg0_1, [select]);  arg0_1 = select = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:226, code: r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
        select_1: "i64[s0]" = torch.ops.aten.select.int(arg5_1, 1, 1)
        index_1: "f32[s0, 400]" = torch.ops.aten.index.Tensor(arg1_1, [select_1]);  arg1_1 = select_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:228, code: embedding += r_embedding
        add: "f32[s0, 400]" = torch.ops.aten.add.Tensor(index, index_1);  index = index_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:216, code: offset_embedding = torch.zeros_like(embedding).cuda()
        sym_size_int: "Sym(s0)" = torch.ops.aten.sym_size.int(arg5_1, 0)
        full: "f32[s0, 400]" = torch.ops.aten.full.default([sym_size_int, 400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  sym_size_int = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:227, code: r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
        select_2: "i64[s0]" = torch.ops.aten.select.int(arg5_1, 1, 1);  arg5_1 = None
        index_2: "f32[s0, 400]" = torch.ops.aten.index.Tensor(arg2_1, [select_2]);  arg2_1 = select_2 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:229, code: offset_embedding += self.func(r_offset_embedding)
        add_1: "f32[s0, 400]" = torch.ops.aten.add.Tensor(full, index_2);  full = index_2 = None
        return (add, add_1)
        