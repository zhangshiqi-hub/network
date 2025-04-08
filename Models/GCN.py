import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv

class GCNEncoder(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNEncoder, self).__init__()
        self.gcn = GraphConv(in_feats, out_feats)

    def forward(self, graph, features):
        return self.gcn(graph, features)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        # Encoder layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, input_dim)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for i in range(self.num_layers):
            # Self-msrsa layer
            x, _ = self.attention_layers[i](x, x, x, key_padding_mask=mask)
            x = x + F.dropout(x, p=0.1, training=self.training)  # Residual connection

            # Feed-forward layer
            x = self.ff_layers[i](x)
            x = x + F.dropout(x, p=0.1, training=self.training)  # Residual connection
        return x


class GraphTransformer(nn.Module):
    def __init__(self, in_feats, gcn_out_feats, transformer_out_feats, num_heads=4, ff_dim=128, num_layers=2):
        super(GraphTransformer, self).__init__()
        self.gcn_encoder = GCNEncoder(in_feats, gcn_out_feats)
        self.transformer_encoder = TransformerEncoder(gcn_out_feats, num_heads, ff_dim, num_layers)
        self.fc = nn.Linear(gcn_out_feats, transformer_out_feats)

    def forward(self, graph, node_features):
        # Step 1: Apply GCN to encode the graph features
        gcn_out = self.gcn_encoder(graph, node_features)

        # Step 2: Transformer expects input of shape (seq_len, batch_size, feature_dim)
        # So we need to permute the output of GCN accordingly
        gcn_out = gcn_out.unsqueeze(1)  # Add batch dimension if necessary
        transformer_out = self.transformer_encoder(gcn_out)

        # Step 3: Apply a final fully connected layer
        out = self.fc(transformer_out)
        return out


# Example usage
if __name__ == "__main__":
    # Simulate a DGL graph with random node features
    g = DGLGraph()
    g.add_nodes(5)  # 10 nodes
    g.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])  # 5 edges for simplicity

    node_features = torch.rand((5, 32))  # 10 nodes, 32-dimensional features

    model = GraphTransformer(in_feats=32, gcn_out_feats=64, transformer_out_feats=32)
    output = model(g, node_features)
    print(output.shape)  # Output shape should be [10, 32] (10 nodes, 32 output features)
