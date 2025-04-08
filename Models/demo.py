# """
#     This module contains self-defined GraphBert model.
# """
#
# import torch
# import torch.nn as nn
# from .configuration_gtmgc import GTMGCConfig
#
# from ..modules import ConformerPredictionHead, GraphRegressionHead
# from ..modules import ConformerPredictionOutput, GraphRegressionOutput
# from ..modules import MultiHeadAttention, AddNorm, PositionWiseFFN, Residual
# from ..modules.utils import make_cdist_mask, compute_distance_residual_bias
# from ..modules import AtomEmbedding, NodeEmbedding  # for Ogb embedding ablation
# from transformers import PretrainedConfig, PreTrainedModel
#
#
# class GTMGCBlock(nn.Module):
#     def __init__(self, config: PretrainedConfig = None, encoder: bool = True) -> None:
#         super().__init__()
#         self.config = config
#         self.encoder = encoder
#         assert config is not None, f"config must be specified to build {self.__class__.__name__}"
#         self.use_A_in_attn = getattr(config, "encoder_use_A_in_attn", False) if encoder else getattr(config, "decoder_use_A_in_attn", False)
#         self.use_D_in_attn = getattr(config, "encoder_use_D_in_attn", False) if encoder else getattr(config, "decoder_use_D_in_attn", False)
#         self.multi_attention = MultiHeadAttention(
#             d_q=getattr(config, "d_q", 256),
#             d_k=getattr(config, "d_k", 256),
#             d_v=getattr(config, "d_v", 256),
#             d_model=getattr(config, "d_model", 256),
#             n_head=getattr(config, "n_head", 8),
#             qkv_bias=getattr(config, "qkv_bias", True),
#             attn_drop=getattr(config, "attn_drop", 0.1),
#             use_adjacency=self.use_A_in_attn,
#             use_distance=self.use_D_in_attn,
#         )
#         self.add_norm01 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1), pre_ln=getattr(config, "pre_ln", True))
#         self.position_wise_ffn = PositionWiseFFN(
#             d_in=getattr(config, "d_model", 256),
#             d_hidden=getattr(config, "d_ffn", 1024),
#             d_out=getattr(config, "d_model", 256),
#             dropout=getattr(config, "ffn_drop", 0.1),
#         )
#         self.add_norm02 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1), pre_ln=getattr(config, "pre_ln", True))
#
#     def forward(self, **inputs):
#         # Using kwargs to make getting inputs more flexible
#         X, M = inputs.get("node_embedding"), inputs.get("node_mask")
#         A = inputs.get("adjacency") if self.use_A_in_attn else None
#         D = inputs.get("distance") if self.use_D_in_attn else None
#         attn_out = self.multi_attention(X, X, X, attention_mask=M, adjacency_matrix=A, distance_matrix=D)
#         Y, attn_weight = attn_out["out"], attn_out["attn_weight"]
#         X = self.add_norm01(X, Y)
#         Y = self.position_wise_ffn(X)
#         X = self.add_norm02(X, Y)
#         return {
#             "out": X,
#             "attn_weight": attn_weight,
#         }
#
#
#
# class GTMGCEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__(config, *inputs, **kwargs)
#         assert config is not None, f"config must be specified to build {self.__class__.__name__}"
#         self.embed_style = getattr(config, "embed_style", "atom_tokenized_ids")
#         if self.embed_style == "atom_tokenized_ids":
#             self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 513), getattr(config, "d_embed", 256), padding_idx=0)
#         elif self.embed_style == "atom_type_ids":
#             self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 119), getattr(config, "d_embed", 256), padding_idx=0)
#         elif self.embed_style == "ogb":
#             self.ogb_node_embedding = NodeEmbedding(atom_embedding_dim=getattr(config, "d_embed", 256), attr_reduction="sum")  # for Ogb embedding ablation
#         self.encoder_blocks = nn.ModuleList([GTMGCBlock(config, encoder=True) for _ in range(getattr(config, "n_encode_layers", 12))])
#         self.__init_weights__()
#
#     def forward(self,   a):
#         # Using kwargs to make getting inputs more flexible
#         if self.embed_style == "atom_tokenized_ids":
#             node_input_ids = inputs.get("node_input_ids")
#             node_embedding = self.node_embedding(node_input_ids)
#         elif self.embed_style == "atom_type_ids":
#             node_input_ids = inputs.get("node_type")  # for node type id
#             node_embedding = self.node_embedding(node_input_ids)
#         elif self.embed_style == "ogb":
#             node_embedding = self.ogb_node_embedding(inputs["node_attr"])  # for Ogb embedding ablation 这里输出的应该是n*dim
#         # laplacian positional embedding
#         lap = inputs.get("lap_eigenvectors")
#         node_embedding[:, :, : lap.shape[-1]] = node_embedding[:, :, : lap.shape[-1]] + lap # 对应concat位置编码
#         inputs["node_embedding"] = node_embedding # 这里的embedding就是concat了位置编码的特征
#
#         if self.config.encoder_use_D_in_attn:
#             C = inputs.get("conformer") # C构象
#             D, D_M = torch.cdist(C, C), make_cdist_mask(inputs.get("node_mask"))
#             D = compute_distance_residual_bias(cdist=D, cdist_mask=D_M)  # masked and row-max subtracted distance matrix
#             # D = D * D_M  # for ablation study
#             inputs["distance"] = D
#         attn_weight_dict = {}
#         for i, encoder_block in enumerate(self.encoder_blocks):
#             block_out = encoder_block(**inputs)
#             node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
#             inputs["node_embedding"] = node_embedding
#             attn_weight_dict[f"encoder_block_{i}"] = attn_weight
#         return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}
#
#
# class GTMGCDecoder(GTMGCPretrainedModel):
#     def __init__(self, config: PretrainedConfig = None, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)
#         assert config is not None, f"config must be specified to build {self.__class__.__name__}"
#         self.decoder_blocks = nn.ModuleList([GTMGCBlock(config, encoder=False) for _ in range(getattr(config, "n_decode_layers", 6))])
#         self.__init_weights__()
#
#     def forward(self, **inputs):
#         node_embedding, lap = inputs.get("node_embedding"), inputs.get("lap_eigenvectors")
#         # laplacian positional embedding
#         node_embedding[:, :, : lap.shape[-1]] = node_embedding[:, :, : lap.shape[-1]] + lap
#         inputs["node_embedding"] = node_embedding
#         attn_weight_dict = {}
#         for i, decoder_block in enumerate(self.decoder_blocks):
#             block_out = decoder_block(**inputs)
#             node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
#             inputs["node_embedding"] = node_embedding
#             attn_weight_dict[f"decoder_block_{i}"] = attn_weight
#         return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}
#
#
# class GTMGCForConformerPrediction(GTMGCPretrainedModel):
#     def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)
#         self.encoder = GTMGCEncoder(config)
#         self.decoder = GTMGCDecoder(config)
#         self.conformer_head = ConformerPredictionHead(hidden_X_dim=getattr(config, "d_model", 256))
#         self.__init_weights__()
#
#     def forward(self, **inputs):
#         conformer, node_mask = inputs.get("conformer"), inputs.get("node_mask")# 构象：n*d
#         # encoder forward
#         encoder_out = self.encoder(**inputs)
#         node_embedding, encoder_attn_weight_dict = encoder_out["node_embedding"], encoder_out["attn_weight_dict"]
#         cache_out = self.conformer_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask, compute_loss=True)
#         loss_cache, conformer_cache = cache_out["loss"], cache_out["conformer_hat"]
#         D_cache, D_M = torch.cdist(conformer_cache, conformer_cache), make_cdist_mask(node_mask)
#         D_cache = compute_distance_residual_bias(cdist=D_cache, cdist_mask=D_M)  # masked and row-max subtracted distance matrix
#         # D_cache = D_cache * D_M  # for ablation study
#         inputs["node_embedding"] = node_embedding
#         inputs["distance"] = D_cache
#         decoder_out = self.decoder(**inputs)
#         node_embedding, decoder_attn_weight_dict = decoder_out["node_embedding"], decoder_out["attn_weight_dict"]
#         outputs = self.conformer_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask, compute_loss=True)  # final prediction
#         return ConformerPredictionOutput(
#             loss=(outputs["loss"] + loss_cache) / 2,
#             # loss=outputs["loss"],
#             cdist_mae=outputs["cdist_mae"],
#             cdist_mse=outputs["cdist_mse"],
#             coord_rmsd=outputs["coord_rmsd"],
#             conformer=outputs["conformer"],
#             conformer_hat=outputs["conformer_hat"],
#             # attentions={**encoder_attn_weight_dict, **decoder_attn_weight_dict}
#         )
#
#
# class GTMGCForGraphRegression(GTMGCPretrainedModel):
#     def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)
#         self.encoder = GTMGCEncoder(config)
#         self.decoder = GraphRegressionHead(hidden_X_dim=getattr(config, "d_model", 256))
#         self.__init_weights__()
#
#     def forward(self, **inputs):
#         encoder_out = self.encoder(**inputs)
#         graph_rep = encoder_out["node_embedding"].mean(dim=1)
#         decoder_outputs = self.decoder(hidden_X=graph_rep, labels=inputs.get("labels"))
#         return GraphRegressionOutput(
#             loss=decoder_outputs["loss"],
#             mae=decoder_outputs["mae"],
#             mse=decoder_outputs["mse"],
#             logits=decoder_outputs["logits"],
#             labels=decoder_outputs["labels"],
#         )
#
# ===================================================================================
# import dgl
# import numpy as np
# import torch
# from torch import nn
#
# from Utils.datautils import align_conformer_hat_to_conformer, compute_distance_residual_bias, _compute_loss
# from Utils.module import AddNorm, PositionWiseFFN
# from Utils.random_walk import random_walk_encoding, random_walk_coding
# import torch.nn.functional as F
#
# from Utils.utils import make_cdist_mask
#
#
# class MSRSA(nn.Module):
#     def __init__(self, num_heads: int = 1, dropout: float = 0.0, use_adjacency: bool = True, use_distance: bool = True) -> None:
#         super().__init__()
#         self.num_heads = num_heads
#         self.use_adjacency = use_adjacency
#         self.use_distance = use_distance
#         self.drop_out = nn.Dropout(dropout)
#         self.weight_A, self.weight_D = None, None
#         if self.use_adjacency:
#             self.weight_A = torch.randn(num_heads).view(1, num_heads, 1, 1)
#             self.weight_A = nn.Parameter(self.weight_A, requires_grad=True)
#         if self.use_distance:
#             self.weight_D = torch.randn(num_heads).view(1, num_heads, 1, 1)
#             self.weight_D = nn.Parameter(self.weight_D, requires_grad=True)
#
#     def forward(
#         self,
#         Q: torch.Tensor,
#         K: torch.Tensor,
#         V: torch.Tensor,
#         attention_mask: torch.Tensor = None,
#         adjacency_matrix: torch.Tensor = None,
#         row_subtracted_distance_matrix: torch.Tensor = None,
#     ) -> torch.Tensor:
#         """Compute (multi-head) Self-Attention || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads
#
#         Args:
#             - Q (torch.Tensor): Query, shape: (b, l, d) or (b, h, l, d)
#             - K (torch.Tensor): Key, shape: (b, l, d) or (b, h, l, d)
#             - V (torch.Tensor): Value, shape: (b, l, d) or (b, h, l, d)
#             - attention_mask (torch.Tensor): Attention mask, shape: (b, l) or (b, l, l), 1 for valid, 0 for invalid
#             - adjacency_matrix (torch.Tensor): Adjacency matrix, shape: (b, l, l)
#             - row_subtracted_distance_matrix (torch.Tensor): Subtracted distance matrix (every raw is subtracted by raw-max value), shape: (b, l, l)
#
#         Returns:
#             torch.Tensor: Weighted sum of value, shape: (b, l, d) or (b, h, l, d)
#         """
#         M, A, D_s = attention_mask, adjacency_matrix, row_subtracted_distance_matrix
#         if self.use_adjacency and A is None:
#             raise ValueError(f"Adjacency matrix is not provided when using adjacency matrix in {self.__class__.__name__}")
#         if self.use_distance and D_s is None:
#             raise ValueError(f"Subtracted distance matrix is not provided when using distance matrix in {self.__class__.__name__}")
#         A=A.unsqueeze_(0) if self.use_adjacency else None
#         A = A.unsqueeze(1) if self.use_adjacency else None  # (b, 1, l, l)  will broadcast to (b, h, l, l)
#         D_s = D_s.unsqueeze(0) if self.use_distance else None
#         D_s = D_s.unsqueeze(1) if self.use_distance else None  # (b, 1, l, l)
#         scale = Q.shape[-1] ** 0.5
#         attn_score = Q @ K.mT  # (b, l, l) | (b, h, l, l)
#         B_A = attn_score * (A * self.weight_A) if self.use_adjacency else None  # (b, h, l, l)
#         B_D = attn_score * (D_s * self.weight_D) if self.use_distance else None  # (b, h, l, l)
#         attn_score = attn_score + B_A if B_A is not None else attn_score
#         attn_score = attn_score + B_D if B_D is not None else attn_score
#         attn_score = attn_score / torch.tensor(scale)  # (b, h, l, l) scaled by sqrt(d) after adding residual terms
#         attention_weight = F.softmax(attn_score, dim=-1)  # (b, l, l) | (b, h, l, l)
#         return {
#             "out": self.drop_out(attention_weight) @ V,  # (b, l, d) | (b, h, l, d)
#             "attn_weight": attention_weight.detach(),  # (b, l, l) | (b, h, l, l)
#         }
# class MultiHeadAttention(nn.Module):
#     def __init__(
#         self,
#         d_q: int = None,
#         d_k: int = None,
#         d_v: int = None,
#         d_model: int = None,
#         n_head: int = 1,
#         qkv_bias: bool = False,
#         attn_drop: float = 0,
#         use_adjacency: bool = False,
#         use_distance: bool = False,
#     ) -> None:
#         super().__init__()
#         self.num_heads = n_head
#         self.hidden_dims = d_model
#         self.msrsa = MSRSA(num_heads=n_head, dropout=attn_drop, use_adjacency=use_adjacency, use_distance=use_distance)
#         assert d_q is not None and d_k is not None and d_v is not None and d_model is not None, "Please specify the dimensions of Q, K, V and d_model"
#         assert d_model % n_head == 0, "d_model must be divisible by n_head"
#         # q: q_dims, k: k_dims, v: v_dims, d: hidden_dims, h: num_heads, d_i: dims of each head
#         self.W_q = nn.Linear(d_q, d_model, bias=qkv_bias)  # (q, h*d_i=d)
#         self.W_k = nn.Linear(d_k, d_model, bias=qkv_bias)  # (k, h*d_i=d)
#         self.W_v = nn.Linear(d_v, d_model, bias=qkv_bias)  # (v, h*d_i=d)
#         self.W_o = nn.Linear(d_model, d_model, bias=qkv_bias)  # (h*d_i=d, d)
#
#     def forward(
#         self,
#         queries: torch.Tensor,
#         keys: torch.Tensor,
#         values: torch.Tensor,
#         attention_mask: torch.Tensor = None,
#         adjacency_matrix: torch.Tensor = None,
#         distance_matrix: torch.Tensor = None,
#     ):
#         """Compute multi-head msrsa || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads
#
#         Args:
#             - queries (torch.Tensor): Query, shape: (b, l, q)
#             - keys (torch.Tensor): Key, shape: (b, l, k)
#             - values (torch.Tensor): Value, shape: (b, l, v)
#             - attention_mask (torch.Tensor, optional): Attention mask, shape: (b, l) or (b, l, l). Defaults to None.
#             - adjacency_matrix (torch.Tensor, optional): Adjacency matrix, shape: (b, l, l). Defaults to None.
#             - distance_matrix (torch.Tensor, optional): Distance matrix, shape: (b, l, l). Defaults to None.
#         Returns:
#             torch.Tensor: Output after multi-head msrsa pooling with shape (b, l, d)
#         """
#         # b: batch_size, h:num_heads, l: seq_len, d: d_hidden
#         b, h, l, d_i = queries.shape[0], self.num_heads, queries.shape[1], self.hidden_dims // self.num_heads
#         Q, K, V = self.W_q(queries), self.W_k(keys), self.W_v(values)  # (b, l, h*d_i=d)
#         Q, K, V = [M.view(b, l, h, d_i).permute(0, 2, 1, 3) for M in (Q, K, V)]  # (b, h, l, d_i)
#         attn_out = self.msrsa(Q, K, V, attention_mask, adjacency_matrix, distance_matrix)
#         out, attn_weight = attn_out["out"], attn_out["attn_weight"]
#         out = out.permute(0, 2, 1, 3).contiguous().view(b, l, h * d_i)  # (b, l, h*d_i=d)
#         # return self.W_o(out)  # (b, l, d)
#         return {
#             "out": self.W_o(out),  # (b, l, d)
#             "attn_weight": attn_weight,  # (b, l, l) | (b, h, l, l)
#         }
#
#
# class PredictForConformer(nn.Module):  # 在这个模型执行梯度下降，所以要传入label
#     def __init__(self,config):
#         super(PredictForConformer, self).__init__()
#         self.config=config  #这个是参数
#         self.encoder = GTMGCEncoder(config)
#         self.decoder = GTMGCDecoder(config)
#         self.conformer_head = ConformerPredictionHead(256)
#         self.__init_weights__()
#
#     def forward(self,**inputs):# 传入的conformer已经是b,n,d,这个作为label
#         # adj=g.adjacency_matrix().to_dense()
#         # conformer_hat=self.constructformer(input)     # 这里返回预测坐标b,n,3
#         # 对齐 conformer_hat 到 conformer
#         # conformer_hat = align_conformer_hat_to_conformer(conformer_hat, conformer)
#         # # 计算损失
#         # loss = _compute_loss(conformer, conformer_hat, torch.tensor(g.num_nodes()))
#         # # 反向传播计算梯度
#         conformer, node_mask = inputs.get("conformer"), inputs.get("node_mask")  # 构象：n*d
#         # encoder forward
#         encoder_out = self.encoder(**inputs)
#         node_embedding, encoder_attn_weight_dict = encoder_out["node_embedding"], encoder_out["attn_weight_dict"]
#         cache_out = self.conformer_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask,
#                                         compute_loss=True)
#         loss_cache, conformer_cache = cache_out["loss"], cache_out["conformer_hat"]
#         D_cache, D_M = torch.cdist(conformer_cache, conformer_cache), make_cdist_mask(node_mask)
#         D_cache = compute_distance_residual_bias(cdist=D_cache,
#                                                  cdist_mask=D_M)  # masked and row-max subtracted distance matrix
#         # D_cache = D_cache * D_M  # for ablation study
#         inputs["node_embedding"] = node_embedding
#         inputs["distance"] = D_cache
#         decoder_out = self.decoder(**inputs)
#         node_embedding, decoder_attn_weight_dict = decoder_out["node_embedding"], decoder_out["attn_weight_dict"]
#         outputs = self.conformer_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask,
#                                       compute_loss=True)  # final prediction
#         return [
#             loss=(outputs["loss"] + loss_cache) / 2,
#             # loss=outputs["loss"],
#             cdist_mae=outputs["cdist_mae"],
#             cdist_mse=outputs["cdist_mse"],
#             coord_rmsd=outputs["coord_rmsd"],
#             conformer=outputs["conformer"],
#             conformer_hat=outputs["conformer_hat"],
#             # attentions={**encoder_attn_weight_dict, **decoder_attn_weight_dict}
#         ]
#
# class ConstructFormer(nn.Module):# 送入的是图g，返回的是预测的坐标
#     def __init__(self,config):
#         super(ConstructFormer, self).__init__()
#         self.config=config
#         self.n = nn.Linear(9, 256, bias=True)
#         self.encoder=GTMGCEncoder(config)
#         self.decoder = GTMGCDecoder(config)
#         self.conformer_head = ConformerPredictionHead(hidden_X_dim=getattr(config, "d_model", 256))
#     def forward(self,input):
#         g=input["graph"]
#         node_embedding = self.n(g.ndata['feat']).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))   # 这里的embedding是n*d
#         position_encodding = random_walk_coding(g)
#         # node_embedding[:, :, : position_encodding.shape[-1]] = node_embedding[:, :, : position_encodding.shape[-1]] + position_encodding
#         node_embedding[:, :position_encodding.shape[-1]] += position_encodding  # 拿到位置编码后的节点特征
#
#         node_embedding=node_embedding.unsqueeze[0] # b,l,d
#         adj = g.adjacency_matrix().to_dense()   # 邻接矩阵 # b,l,l
#         adj=adj.unsqueeze_(0)
#
#         return conformer_hat
# class GTMGCEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__(config, *inputs, **kwargs)
#         assert config is not None, f"config must be specified to build {self.__class__.__name__}"
#         self.embed_style = getattr(config, "embed_style", "atom_tokenized_ids")
#         if self.embed_style == "atom_tokenized_ids":
#             self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 513), getattr(config, "d_embed", 256), padding_idx=0)
#         elif self.embed_style == "atom_type_ids":
#             self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 119), getattr(config, "d_embed", 256), padding_idx=0)
#         elif self.embed_style == "ogb":
#             self.ogb_node_embedding = NodeEmbedding(atom_embedding_dim=getattr(config, "d_embed", 256), attr_reduction="sum")  # for Ogb embedding ablation
#         self.encoder_blocks = nn.ModuleList([GTMGCBlock(config, encoder=True) for _ in range(getattr(config, "n_encode_layers", 12))])
#         self.__init_weights__()
#
#     def forward(self, **inputs):
#         # Using kwargs to make getting inputs more flexible
#         if self.embed_style == "atom_tokenized_ids":
#             node_input_ids = inputs.get("node_input_ids")
#             node_embedding = self.node_embedding(node_input_ids)
#         elif self.embed_style == "atom_type_ids":
#             node_input_ids = inputs.get("node_type")  # for node type id
#             node_embedding = self.node_embedding(node_input_ids)
#         elif self.embed_style == "ogb":
#             node_embedding = self.ogb_node_embedding(inputs["node_attr"])  # for Ogb embedding ablation 这里输出的应该是n*dim
#         # laplacian positional embedding
#         lap = inputs.get("lap_eigenvectors")
#         node_embedding[:, :, : lap.shape[-1]] = node_embedding[:, :, : lap.shape[-1]] + lap # 对应concat位置编码
#         inputs["node_embedding"] = node_embedding # 这里的embedding就是concat了位置编码的特征
#
#         if self.config.encoder_use_D_in_attn:
#             C = inputs.get("conformer") # C构象
#             D, D_M = torch.cdist(C, C), make_cdist_mask(inputs.get("node_mask"))
#             D = compute_distance_residual_bias(cdist=D, cdist_mask=D_M)  # masked and row-max subtracted distance matrix
#             # D = D * D_M  # for ablation study
#             inputs["distance"] = D
#         attn_weight_dict = {}
#         for i, encoder_block in enumerate(self.encoder_blocks):
#             block_out = encoder_block(**inputs)
#             node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
#             inputs["node_embedding"] = node_embedding
#             attn_weight_dict[f"encoder_block_{i}"] = attn_weight
#         return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}
#
# class GTMGCDecoder(nn.Module):
#     def __init__(self, config, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)
#         assert config is not None, f"config must be specified to build {self.__class__.__name__}"
#         self.decoder_blocks = nn.ModuleList([GTMGCBlock(config, encoder=False) for _ in range(getattr(config, "n_decode_layers", 6))])
#         self.__init_weights__()
#
#     def forward(self, **inputs):
#         node_embedding, lap = inputs.get("node_embedding"), inputs.get("lap_eigenvectors")
#         # laplacian positional embedding
#         node_embedding[:, :, : lap.shape[-1]] = node_embedding[:, :, : lap.shape[-1]] + lap
#         inputs["node_embedding"] = node_embedding
#         attn_weight_dict = {}
#         for i, decoder_block in enumerate(self.decoder_blocks):
#             block_out = decoder_block(**inputs)
#             node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
#             inputs["node_embedding"] = node_embedding
#             attn_weight_dict[f"decoder_block_{i}"] = attn_weight
#         return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}
# class ConformerPredictionHead(nn.Module):
#     def __init__(self, hidden_X_dim: int = 256) -> None:
#         super().__init__()
#         self.head = nn.Sequential(nn.Linear(hidden_X_dim, hidden_X_dim * 3), nn.GELU(), nn.Linear(hidden_X_dim * 3, 3))
#         # self.criterion = MultiTaskLearnableWeightLoss(n_task=2)
#
#     def forward(self, conformer: torch.Tensor, hidden_X: torch.Tensor, padding_mask: torch.Tensor = None, compute_loss: bool = True) -> torch.Tensor:
#         # get conformer_hat
#         conformer_hat = self.head(hidden_X)
#         # mask padding atoms
#         conformer_hat = mask_hidden_state(conformer_hat, padding_mask)
#         # align conformer_hat to conformer
#         conformer_hat = align_conformer_hat_to_conformer(conformer_hat, conformer)
#         conformer_hat = mask_hidden_state(conformer_hat, padding_mask)
#         if not compute_loss:
#             return conformer_hat
#         return self._compute_loss(conformer, conformer_hat, padding_mask)
#
#     def _compute_loss(self, conformer: torch.Tensor, conformer_hat: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
#         # Convenient for design loss function in the future.
#         cdist, cdist_hat = torch.cdist(conformer, conformer), torch.cdist(conformer_hat, conformer_hat)
#         c_dist_mask = make_cdist_mask(padding_mask)
#         cdist, cdist_hat = cdist * c_dist_mask, cdist_hat * c_dist_mask
#         cdist_mae = self._compute_cdist_mae(cdist, cdist_hat, c_dist_mask)
#         cdist_mse = self._compute_cdist_mse(cdist, cdist_hat, c_dist_mask)
#         coord_rmsd = self._compute_conformer_rmsd(conformer, conformer_hat, padding_mask)
#         # compute learnable weighted loss
#         # loss = self.criterion([cdist_mae, cdist_rmse])
#         loss = cdist_mae
#         return {
#             "loss": loss,
#             "cdist_mae": cdist_mae.detach(),
#             "cdist_mse": cdist_mse.detach(),
#             "coord_rmsd": coord_rmsd.detach(),
#             "conformer": conformer.detach(),
#             "conformer_hat": conformer_hat.detach(),
#         }
#
#     @staticmethod
#     def _compute_cdist_mae(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
#         """Compute mean absolute error of conformer and conformer_hat.
#
#         Args:
#             - masked_cdist (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer.
#             - masked_cdist_hat (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer_hat.
#             - cdist_mask (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the mask of the pairwise distance matrix.
#
#         Returns:
#             torch.Tensor: The mean absolute error of conformer and conformer_hat.
#         """
#         mae = F.l1_loss(masked_cdist, masked_cdist_hat, reduction="sum") / cdist_mask.sum()  # exclude padding atoms
#         return mae
#
#     @staticmethod
#     def _compute_cdist_mse(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
#         """Compute root mean squared error of conformer and conformer_hat.
#
#         Args:
#             - masked_cdist (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer.
#             - masked_cdist_hat (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer_hat.
#             - cdist_mask (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the mask of the pairwise distance matrix.
#
#         Returns:
#             torch.Tensor: The root mean squared error of conformer and conformer_hat.
#         """
#         mse = F.mse_loss(masked_cdist, masked_cdist_hat, reduction="sum") / cdist_mask.sum()  # exclude padding atoms
#         return mse
#
#     @staticmethod
#     def _compute_conformer_rmsd(masked_conformer: torch.Tensor, masked_conformer_hat: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
#         """Compute root mean squared deviation of conformer and conformer_hat.
#
#         Args:
#             - masked_conformer (torch.Tensor): A torch tensor of shape (b, l, 3), which denotes the coordinate of the conformer.
#             - masked_conformer_hat (torch.Tensor): A torch tensor of shape (b, l, 3), which denotes the coordinate of the conformer_hat.
#             - padding_mask (torch.Tensor): A torch tensor of shape (b, l), which denotes the mask of the conformer.
#
#         Returns:
#             torch.Tensor: The root mean squared deviation of conformer and conformer_hat.
#         """
#         R, R_h, M = masked_conformer, masked_conformer_hat, padding_mask
#         delta = (R - R_h).to(torch.float32)
#         point_2_norm = torch.norm(delta, p=2, dim=-1)
#         MSD = torch.sum(point_2_norm**2, dim=-1) / torch.sum(M, dim=-1)
#         RMSD = torch.sqrt(MSD)
#         return RMSD.mean()
#
#
# class Attention(nn.Module):
#     def __init__(self):
#         super(Attention,self).__init__()
#         self.q = nn.Linear(256, 256, bias=True)
#         self.k = nn.Linear(256, 256, bias=True)
#         self.v = nn.Linear(256, 256, bias=True)
#         self.attn = MultiHeadAttention(d_q=256, d_k=256, d_v=256, d_model=256, n_head=4, qkv_bias=True,
#                                        use_adjacency=True, use_distance=False)
#
#     def forward(self,node_embedding,adj):
#         if node_embedding.dim() == 2:  # 维度为 3 时，进行 unsqueeze
#             q = self.q(node_embedding.unsqueeze(0))  # (1, batch_size, num_nodes, 256)
#             k = self.k(node_embedding.unsqueeze(0))
#             v = self.v(node_embedding.unsqueeze(0))
#         else:
#             q = self.q(node_embedding)
#             k = self.k(node_embedding)
#             v = self.v(node_embedding)
#         out = self.attn(q, k, v, adjacency_matrix=adj, distance_matrix=None)
#         return out
#
# class CONSBlock(nn.Module):
#     def __init__(self, use_A: bool = True) -> None:
#         super().__init__()
#         # self.config = config
#         self.use_A = use_A
#         # assert config is not None, f"config must be specified to build {self.__class__.__name__}"
#         # self.use_A_in_attn = getattr(config, "encoder_use_A_in_attn", False) if encoder else getattr(config, "decoder_use_A_in_attn", False)
#         # self.use_D_in_attn = getattr(config, "encoder_use_D_in_attn", False) if encoder else getattr(config, "decoder_use_D_in_attn", False)
#         self.multi_attention = MultiHeadAttention(
#             d_q=256,
#             d_k=256,
#             d_v=256,
#             d_model=256,
#             n_head=4,
#             qkv_bias=True,
#             attn_drop=0.1,
#             use_adjacency=self.use_A,
#             # use_distance=self.use_D_in_attn,
#         )
#         self.add_norm01 = AddNorm(norm_shape=256, dropout=0.1, pre_ln=True)
#         self.position_wise_ffn = PositionWiseFFN(
#             d_in= 256,
#             d_hidden=1024,
#             d_out=256,
#             dropout=0.1,
#         )
#         self.add_norm02 = AddNorm(norm_shape=256, dropout= 0.1, pre_ln=True)
#
#     def forward(self, node_embedding: torch.Tensor,adjacency_matrix: torch.Tensor) -> torch.Tensor:
#         # Using kwargs to make getting inputs more flexible
#         # X, M = inputs.get("node_embedding"), inputs.get("node_mask")
#         # 输入b,l,d的特征和b,l,l的邻接矩阵
#         X=node_embedding
#         adj =adjacency_matrix
#
#
#         # D = inputs.get("distance") if self.use_D_in_attn else None
#
#         attn_out = self.multi_attention(X, X, X, adjacency_matrix=adj, distance_matrix=None)
#         Y, attn_weight = attn_out["out"], attn_out["attn_weight"]
#         X = self.add_norm01(X, Y)
#         Y = self.position_wise_ffn(X)
#         X = self.add_norm02(X, Y)
#         return {
#             "out": X,
#             "attn_weight": attn_weight,
#         }
#
#
#
#
# if __name__ == "__main__":
#     x=torch.randn(5,9)
#
#
#
