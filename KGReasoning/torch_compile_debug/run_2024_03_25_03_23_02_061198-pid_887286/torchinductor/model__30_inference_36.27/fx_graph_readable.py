class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "i64[1, s0]"):
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:403, code: queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        slice_1: "i64[1, s0]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807)
        slice_2: "i64[1, 2]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 2);  slice_1 = None
        slice_3: "i64[1, s0]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807)
        slice_4: "i64[1, 1]" = torch.ops.aten.slice.Tensor(slice_3, 1, 5, 6);  slice_3 = None
        cat: "i64[1, 3]" = torch.ops.aten.cat.default([slice_2, slice_4], 1);  slice_2 = slice_4 = None
        slice_5: "i64[1, s0]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807)
        slice_6: "i64[1, 2]" = torch.ops.aten.slice.Tensor(slice_5, 1, 2, 4);  slice_5 = None
        slice_7: "i64[1, s0]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807);  arg1_1 = None
        slice_8: "i64[1, 1]" = torch.ops.aten.slice.Tensor(slice_7, 1, 5, 6);  slice_7 = None
        cat_1: "i64[1, 3]" = torch.ops.aten.cat.default([slice_6, slice_8], 1);  slice_6 = slice_8 = None
        cat_2: "i64[1, 6]" = torch.ops.aten.cat.default([cat, cat_1], 1);  cat = cat_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:404, code: queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        view: "i64[2, 3]" = torch.ops.aten.view.default(cat_2, [2, -1]);  cat_2 = None
        return (view,)
        