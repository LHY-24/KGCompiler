class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 5]"):
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:401, code: queries = queries[:, :-1] # remove union -1
        slice_1: "i64[1, 5]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 9223372036854775807);  arg0_1 = None
        slice_2: "i64[1, 4]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, -1);  slice_1 = None
        
        # File: /home/linhongyu/KG/KG-Compilation/KGReasoning/models.py:404, code: queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        view: "i64[2, 2]" = torch.ops.aten.reshape.default(slice_2, [2, -1]);  slice_2 = None
        return (view,)
        