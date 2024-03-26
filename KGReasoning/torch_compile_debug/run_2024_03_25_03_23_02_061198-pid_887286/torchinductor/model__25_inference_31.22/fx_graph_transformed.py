class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[14505, 400]", arg1_1: "f32[474, 400]", arg2_1: "f32[474, 400]", arg3_1: "f32[400, 400]", arg4_1: "f32[400]", arg5_1: "f32[400, 400]", arg6_1: "f32[400]", arg7_1: "f32[400, 400]", arg8_1: "f32[400]", arg9_1: "f32[400, 400]", arg10_1: "f32[400]", arg11_1: "Sym(s0)", arg12_1: "i64[1, s0]"):
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:216, code: offset_embedding = torch.zeros_like(embedding).cuda()
        full_default: "f32[1, 400]" = torch.ops.aten.full.default([1, 400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[1, 400]" = torch.ops.aten.full.default([1, 400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:227, code: r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
        select_2: "i64[1]" = torch.ops.aten.select.int(arg12_1, 1, 1)
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:229, code: offset_embedding += self.func(r_offset_embedding)
        index_2: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg2_1, [select_2]);  select_2 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:227, code: r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
        select_5: "i64[1]" = torch.ops.aten.select.int(arg12_1, 1, 3)
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:229, code: offset_embedding += self.func(r_offset_embedding)
        index_5: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg2_1, [select_5]);  select_5 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:239, code: offset_embedding = self.offset_net(torch.stack(offset_embedding_list))
        cat_1: "f32[2, 400]" = torch.ops.aten.cat.default([index_2, index_5]);  index_2 = index_5 = None
        view_5: "f32[2, 1, 400]" = torch.ops.aten.reshape.default(cat_1, [2, 1, 400]);  cat_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:41, code: offset, _ = torch.min(embeddings, dim=0)
        min_1 = torch.ops.aten.min.dim(view_5, 0)
        getitem: "f32[1, 400]" = min_1[0];  min_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:214, code: embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
        select: "i64[1]" = torch.ops.aten.select.int(arg12_1, 1, 0)
        index: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg0_1, [select]);  select = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:226, code: r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
        select_1: "i64[1]" = torch.ops.aten.select.int(arg12_1, 1, 1)
        index_1: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg1_1, [select_1]);  select_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:228, code: embedding += r_embedding
        add: "f32[1, 400]" = torch.ops.aten.add.Tensor(index, index_1);  index = index_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:214, code: embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
        select_3: "i64[1]" = torch.ops.aten.select.int(arg12_1, 1, 2)
        index_3: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg0_1, [select_3]);  arg0_1 = select_3 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:226, code: r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
        select_4: "i64[1]" = torch.ops.aten.select.int(arg12_1, 1, 3)
        index_4: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg1_1, [select_4]);  select_4 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:228, code: embedding += r_embedding
        add_2: "f32[1, 400]" = torch.ops.aten.add.Tensor(index_3, index_4);  index_3 = index_4 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:238, code: embedding = self.center_net(torch.stack(embedding_list))
        cat: "f32[2, 400]" = torch.ops.aten.cat.default([add, add_2]);  add = add_2 = None
        view: "f32[2, 1, 400]" = torch.ops.aten.reshape.default(cat, [2, 1, 400]);  cat = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:57, code: layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, dim)
        view_1: "f32[2, 400]" = torch.ops.aten.reshape.default(view, [2, 400])
        permute: "f32[400, 400]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[2, 400]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        add_tensor_2: "f32[2, 400]" = torch.ops.aten.add.Tensor(mm_default_2, arg4_1);  mm_default_2 = arg4_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:57, code: layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, dim)
        view_2: "f32[2, 1, 400]" = torch.ops.aten.reshape.default(add_tensor_2, [2, 1, 400]);  add_tensor_2 = None
        relu: "f32[2, 1, 400]" = torch.ops.aten.relu.default(view_2);  view_2 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:58, code: attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, dim)
        view_3: "f32[2, 400]" = torch.ops.aten.reshape.default(relu, [2, 400]);  relu = None
        permute_1: "f32[400, 400]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "f32[2, 400]" = torch.ops.aten.addmm.default(arg6_1, view_3, permute_1);  arg6_1 = view_3 = permute_1 = None
        view_4: "f32[2, 1, 400]" = torch.ops.aten.reshape.default(addmm_1, [2, 1, 400]);  addmm_1 = None
        amax: "f32[1, 1, 400]" = torch.ops.aten.amax.default(view_4, [0], True)
        sub: "f32[2, 1, 400]" = torch.ops.aten.sub.Tensor(view_4, amax);  view_4 = amax = None
        exp: "f32[2, 1, 400]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1, 1, 400]" = torch.ops.aten.sum.dim_IntList(exp, [0], True)
        div: "f32[2, 1, 400]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:59, code: embedding = torch.sum(attention * embeddings, dim=0)
        mul: "f32[2, 1, 400]" = torch.ops.aten.mul.Tensor(div, view);  div = view = None
        sum_2: "f32[1, 400]" = torch.ops.aten.sum.dim_IntList(mul, [0]);  mul = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:226, code: r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
        select_6: "i64[1]" = torch.ops.aten.select.int(arg12_1, 1, 4)
        index_6: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg1_1, [select_6]);  arg1_1 = select_6 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:228, code: embedding += r_embedding
        add_4: "f32[1, 400]" = torch.ops.aten.add.Tensor(sum_2, index_6);  sum_2 = index_6 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:38, code: layer1_act = F.relu(self.layer1(embeddings))
        view_6: "f32[2, 400]" = torch.ops.aten.reshape.default(view_5, [2, 400]);  view_5 = None
        permute_2: "f32[400, 400]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[2, 400]" = torch.ops.aten.mm.default(view_6, permute_2);  view_6 = permute_2 = None
        add_tensor_1: "f32[2, 400]" = torch.ops.aten.add.Tensor(mm_default_1, arg8_1);  mm_default_1 = arg8_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:38, code: layer1_act = F.relu(self.layer1(embeddings))
        view_7: "f32[2, 1, 400]" = torch.ops.aten.reshape.default(add_tensor_1, [2, 1, 400]);  add_tensor_1 = None
        relu_1: "f32[2, 1, 400]" = torch.ops.aten.relu.default(view_7);  view_7 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:39, code: layer1_mean = torch.mean(layer1_act, dim=0)
        mean: "f32[1, 400]" = torch.ops.aten.mean.dim(relu_1, [0]);  relu_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:40, code: gate = torch.sigmoid(self.layer2(layer1_mean))
        permute_3: "f32[400, 400]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[1, 400]" = torch.ops.aten.mm.default(mean, permute_3);  mean = permute_3 = None
        add_tensor: "f32[1, 400]" = torch.ops.aten.add.Tensor(mm_default, arg10_1);  mm_default = arg10_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:40, code: gate = torch.sigmoid(self.layer2(layer1_mean))
        sigmoid: "f32[1, 400]" = torch.ops.aten.sigmoid.default(add_tensor);  add_tensor = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:43, code: return offset * gate
        mul_1: "f32[1, 400]" = torch.ops.aten.mul.Tensor(getitem, sigmoid);  getitem = sigmoid = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:227, code: r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
        select_7: "i64[1]" = torch.ops.aten.select.int(arg12_1, 1, 4);  arg12_1 = None
        index_7: "f32[1, 400]" = torch.ops.aten.index.Tensor(arg2_1, [select_7]);  arg2_1 = select_7 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:229, code: offset_embedding += self.func(r_offset_embedding)
        add_5: "f32[1, 400]" = torch.ops.aten.add.Tensor(mul_1, index_7);  mul_1 = index_7 = None
        return (add_4, add_5)
        