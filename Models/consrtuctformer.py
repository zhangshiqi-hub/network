import torch
from torch import nn

from Utils.datautils import align_conformer_hat_to_conformer, _compute_loss, compute_distance_residual_bias, \
    new_compute_distance_residual
from Utils.module import AddNorm, PositionWiseFFN
from Utils.utils import mask_attention_score

import torch.nn.functional as F




class MSRSA(nn.Module):
    def __init__(self, num_heads: int = 1, dropout: float = 0.0, use_adjacency: bool = True, use_distance: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.use_adjacency = use_adjacency
        self.use_distance = use_distance
        self.drop_out = nn.Dropout(dropout)
        self.weight_A, self.weight_D = None, None
        if self.use_adjacency:
            self.weight_A = torch.randn(num_heads).view(1, num_heads, 1, 1)
            self.weight_A = nn.Parameter(self.weight_A, requires_grad=True)
        if self.use_distance:
            self.weight_D = torch.randn(num_heads).view(1, num_heads, 1, 1)
            self.weight_D = nn.Parameter(self.weight_D, requires_grad=True)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor = None,
        adjacency_matrix: torch.Tensor = None,
        row_subtracted_distance_matrix: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute (multi-head) Self-Attention || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads

        Args:
            - Q (torch.Tensor): Query, shape: (b, l, d) or (b, h, l, d)
            - K (torch.Tensor): Key, shape: (b, l, d) or (b, h, l, d)
            - V (torch.Tensor): Value, shape: (b, l, d) or (b, h, l, d)
            - attention_mask (torch.Tensor): Attention mask, shape: (b, l) or (b, l, l), 1 for valid, 0 for invalid
            - adjacency_matrix (torch.Tensor): Adjacency matrix, shape: (b, l, l)
            - row_subtracted_distance_matrix (torch.Tensor): Subtracted distance matrix (every raw is subtracted by raw-max value), shape: (b, l, l)

        Returns:
            torch.Tensor: Weighted sum of value, shape: (b, l, d) or (b, h, l, d)
        """
        M, A, D_s = attention_mask, adjacency_matrix, row_subtracted_distance_matrix
        if self.use_adjacency and A is None:
            raise ValueError(f"Adjacency matrix is not provided when using adjacency matrix in {self.__class__.__name__}")
        if self.use_distance and D_s is None:
            raise ValueError(f"Subtracted distance matrix is not provided when using distance matrix in {self.__class__.__name__}")
        A = A.unsqueeze(1) if self.use_adjacency else None  # (b, 1, l, l)  will broadcast to (b, h, l, l)
        D_s = D_s.unsqueeze(1) if self.use_distance else None  # (b, 1, l, l)
        scale = Q.shape[-1] ** 0.5
        attn_score = Q @ K.mT  # (b, l, l) | (b, h, l, l)
        attn_score = mask_attention_score(attn_score, M, 0.0) if M is not None else attn_score
        B_A = attn_score * (A * self.weight_A) if self.use_adjacency else None  # (b, h, l, l)
        B_D = attn_score * (D_s * self.weight_D) if self.use_distance else None  # (b, h, l, l)
        attn_score = attn_score + B_A if B_A is not None else attn_score
        attn_score = attn_score + B_D if B_D is not None else attn_score
        attn_score = attn_score / torch.tensor(scale)  # (b, h, l, l) scaled by sqrt(d) after adding residual terms
        attention_weight = F.softmax(attn_score, dim=-1)  # (b, l, l) | (b, h, l, l)
        return {
            "out": self.drop_out(attention_weight) @ V,  # (b, l, d) | (b, h, l, d)
            "attn_weight": attention_weight.detach(),  # (b, l, l) | (b, h, l, l)
        }
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_q: int = None,
        d_k: int = None,
        d_v: int = None,
        d_model: int = None,
        n_head: int = 1,
        qkv_bias: bool = False,
        attn_drop: float = 0,
        use_adjacency: bool = False,
        use_distance: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = n_head
        self.hidden_dims = d_model
        self.attention = MSRSA(num_heads=n_head, dropout=attn_drop, use_adjacency=use_adjacency, use_distance=use_distance)
        assert d_q is not None and d_k is not None and d_v is not None and d_model is not None, "Please specify the dimensions of Q, K, V and d_model"
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        # q: q_dims, k: k_dims, v: v_dims, d: hidden_dims, h: num_heads, d_i: dims of each head
        self.W_q = nn.Linear(d_q, d_model, bias=qkv_bias)  # (q, h*d_i=d)
        self.W_k = nn.Linear(d_k, d_model, bias=qkv_bias)  # (k, h*d_i=d)
        self.W_v = nn.Linear(d_v, d_model, bias=qkv_bias)  # (v, h*d_i=d)
        self.W_o = nn.Linear(d_model, d_model, bias=qkv_bias)  # (h*d_i=d, d)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        adjacency_matrix: torch.Tensor = None,
        distance_matrix: torch.Tensor = None,
    ):
        """Compute multi-head attention || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads

        Args:
            - queries (torch.Tensor): Query, shape: (b, l, q)
            - keys (torch.Tensor): Key, shape: (b, l, k)
            - values (torch.Tensor): Value, shape: (b, l, v)
            - attention_mask (torch.Tensor, optional): Attention mask, shape: (b, l) or (b, l, l). Defaults to None.
            - adjacency_matrix (torch.Tensor, optional): Adjacency matrix, shape: (b, l, l). Defaults to None.
            - distance_matrix (torch.Tensor, optional): Distance matrix, shape: (b, l, l). Defaults to None.
        Returns:
            torch.Tensor: Output after multi-head attention pooling with shape (b, l, d)
        """
        # b: batch_size, h:num_heads, l: seq_len, d: d_hidden
        b, h, l, d_i = queries.shape[0], self.num_heads, queries.shape[1], self.hidden_dims // self.num_heads
        Q, K, V = self.W_q(queries), self.W_k(keys), self.W_v(values)  # (b, l, h*d_i=d)
        Q, K, V = [M.view(b, l, h, d_i).permute(0, 2, 1, 3) for M in (Q, K, V)]  # (b, h, l, d_i)
        attn_out = self.attention(Q, K, V, attention_mask, adjacency_matrix, distance_matrix)
        out, attn_weight = attn_out["out"], attn_out["attn_weight"]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, l, h * d_i)  # (b, l, h*d_i=d)
        # return self.W_o(out)  # (b, l, d)
        return {
            "out": self.W_o(out),  # (b, l, d)
            "attn_weight": attn_weight,  # (b, l, l) | (b, h, l, l)
        }

class GTMGCBlock(nn.Module):
    def __init__(self, config = None, encoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.use_A_in_attn = getattr(config, "encoder_use_A_in_attn", False) if encoder else getattr(config, "decoder_use_A_in_attn", False)
        self.use_D_in_attn = getattr(config, "encoder_use_D_in_attn", False) if encoder else getattr(config, "decoder_use_D_in_attn", False)
        self.multi_attention = MultiHeadAttention(
            d_q=getattr(config, "d_q", 256),
            d_k=getattr(config, "d_k", 256),
            d_v=getattr(config, "d_v", 256),
            d_model=getattr(config, "d_model", 256),
            n_head=getattr(config, "n_head", 8),
            qkv_bias=getattr(config, "qkv_bias", True),
            attn_drop=getattr(config, "attn_drop", 0.1),
            use_adjacency=self.use_A_in_attn,
            use_distance=self.use_D_in_attn,
        )
        self.add_norm01 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1), pre_ln=getattr(config, "pre_ln", True))
        self.position_wise_ffn = PositionWiseFFN(
            d_in=getattr(config, "d_model", 256),
            d_hidden=getattr(config, "d_ffn", 1024),
            d_out=getattr(config, "d_model", 256),
            dropout=getattr(config, "ffn_drop", 0.1),
        )
        self.add_norm02 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1), pre_ln=getattr(config, "pre_ln", True))

    def forward(self, **inputs):
        # Using kwargs to make getting inputs more flexible
        X, M = inputs.get("node_embedding"), inputs.get("node_mask")
        A = inputs.get("adjacency") if self.use_A_in_attn else None
        D = inputs.get("distance") if self.use_D_in_attn else None
        attn_out = self.multi_attention(X, X, X, attention_mask=M, adjacency_matrix=A, distance_matrix=D)
        Y, attn_weight = attn_out["out"], attn_out["attn_weight"]
        X = self.add_norm01(X, Y)
        Y = self.position_wise_ffn(X)
        X = self.add_norm02(X, Y)
        return {
            "out": X,
            "attn_weight": attn_weight,
        }
class GTMGCEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n=nn.Linear(9,256)
        # self.embed_style = getattr(config, "embed_style", "atom_tokenized_ids")
        # if self.embed_style == "atom_tokenized_ids":
        #     self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 513), getattr(config, "d_embed", 256), padding_idx=0)
        # elif self.embed_style == "atom_type_ids":
        #     self.node_embedding = nn.Embedding(getattr(config, "atom_vocab_size", 119), getattr(config, "d_embed", 256), padding_idx=0)
        # elif self.embed_style == "ogb":
        #     self.ogb_node_embedding = NodeEmbedding(atom_embedding_dim=getattr(config, "d_embed", 256), attr_reduction="sum")  # for Ogb embedding ablation
        self.encoder_blocks = nn.ModuleList([GTMGCBlock(config, encoder=True) for _ in range(getattr(config, "n_encode_layers", 12))])


    def forward(self, **inputs):
        # Using kwargs to make getting inputs more flexible
        # if self.embed_style == "atom_tokenized_ids":
        #     node_input_ids = inputs.get("node_input_ids")
        #     node_embedding = self.node_embedding(node_input_ids)
        # elif self.embed_style == "atom_type_ids":
        #     node_input_ids = inputs.get("node_type")  # for node type id
        #     node_embedding = self.node_embedding(node_input_ids)
        # elif self.embed_style == "ogb":
        #     node_embedding = self.ogb_node_embedding(inputs["node_attr"])  # for Ogb embedding ablation 这里输出的应该是n*dim
        node_embedding = self.n(inputs["node_encodding"])
        # laplacian positional embedding
        rad = inputs.get("position_encodding")
        node_embedding[:, :, : rad.shape[-1]] = node_embedding[:, :, : rad.shape[-1]] + rad # 对应concat位置编码
        inputs["node_embedding"] = node_embedding # 这里的embedding就是concat了位置编码的特征

        if self.config.encoder_use_D_in_attn:
            C = inputs.get("conformer") # C构象
            D= torch.cdist(C, C) # masked and row-max subtracted distance matrix
            D=compute_distance_residual_bias(cdist=D)
            # D = D * D_M  # for ablation study
            inputs["distance"] = D
        attn_weight_dict = {}
        for i, encoder_block in enumerate(self.encoder_blocks):
            block_out = encoder_block(**inputs)
            node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
            inputs["node_embedding"] = node_embedding
            attn_weight_dict[f"encoder_block_{i}"] = attn_weight
        return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}
class GTMGCDecoder(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config
        self.decoder_blocks = nn.ModuleList([GTMGCBlock(config, encoder=False) for _ in range(6)])


    def forward(self, **inputs):
        node_embedding, lap = inputs.get("node_embedding"), inputs.get("position_encodding")
        # laplacian positional embedding
        rad = inputs.get("position_encodding")
        node_embedding[:, :, : rad.shape[-1]] = node_embedding[:, :, : rad.shape[-1]] + rad  # 对应concat位置编码
        inputs["node_embedding"] = node_embedding
        attn_weight_dict = {}
        for i, decoder_block in enumerate(self.decoder_blocks):
            block_out = decoder_block(**inputs)
            node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
            inputs["node_embedding"] = node_embedding
            attn_weight_dict[f"decoder_block_{i}"] = attn_weight
        return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}




class ConformerPredictionHead(nn.Module):
    def __init__(self, hidden_X_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_X_dim, hidden_X_dim * 3), nn.GELU(), nn.Linear(hidden_X_dim * 3, 3))
        # self.criterion = MultiTaskLearnableWeightLoss(n_task=2)

    def forward(self, conformer: torch.Tensor, hidden_X: torch.Tensor, compute_loss: bool = True) -> torch.Tensor:
        # get conformer_hat
        hidden_X = hidden_X.clone()
        conformer_hat = self.head(hidden_X)
        # align conformer_hat to conformer
        conformer_hat = align_conformer_hat_to_conformer(conformer_hat, conformer)
        if not compute_loss:
            return conformer_hat
        return _compute_loss(conformer, conformer_hat)


class GTMGCForConformerPrediction(nn.Module):
    def  __init__(self, config):
        super().__init__()
        self.encoder = GTMGCEncoder(config)
        self.decoder = GTMGCDecoder(config)
        self.conformer_head = ConformerPredictionHead(hidden_X_dim=getattr(config, "d_model", 256))

    def forward(self, inputs):
        conformer, node_mask = inputs.get("conformer"), inputs.get("node_mask")# 构象：n*d
        # encoder forward
        encoder_out = self.encoder(**inputs)
        node_embedding, encoder_attn_weight_dict = encoder_out["node_embedding"], encoder_out["attn_weight_dict"]
        cache_out = self.conformer_head(conformer=conformer, hidden_X=node_embedding, compute_loss=True)
        loss_cache, conformer_cache = cache_out["loss"], cache_out["conformer_hat"]
        D_cache = torch.cdist(conformer_cache, conformer_cache)
        # D_cache = compute_distance_residual_bias(cdist=D_cache)
        D_cache = new_compute_distance_residual(cdist=D_cache)
        # masked and row-max subtracted distance matrix
        # D_cache = D_cache * D_M  # for ablation study
        inputs["node_embedding"] = node_embedding
        inputs["distance"] = D_cache

        decoder_out = self.decoder(**inputs)
        node_embedding, decoder_attn_weight_dict = decoder_out["node_embedding"], decoder_out["attn_weight_dict"]
        outputs = self.conformer_head(conformer=conformer, hidden_X=node_embedding, compute_loss=True)  # final prediction
        return {
            "loss":(outputs["loss"] + loss_cache) / 2,
            # loss=outputs["loss"],
            "cdist_mae":outputs["cdist_mae"],
            "cdist_mse":outputs["cdist_mse"],
            "coord_rmsd":outputs["coord_rmsd"],
            "conformer":outputs["conformer"],
            "conformer_hat":outputs["conformer_hat"],
            # attentions={**encoder_attn_weight_dict, **decoder_attn_weight_dict}
        }

if __name__ == "__main__":
    input={
        "node_encodding":torch.randn(1, 569, 9),
        "adjacency":torch.randn(1,569,569),
        "conformer":torch.randn(1,569,3),
        "node_mask":torch.randn(1,569,1),
        "position_encodding":torch.randn(1,569,5)
    }
    from config import Config
    config = Config(1,1,1,nhead=4,dropout=0)
    model=GTMGCForConformerPrediction(config)
    output=model(input)
    print(output)