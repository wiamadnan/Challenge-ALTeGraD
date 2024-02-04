import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel
from torch_geometric.nn import GATConv
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import LayerNorm
#------ Default Model ------#

# class GraphEncoder(nn.Module):
#     def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
#         super(GraphEncoder, self).__init__()
#         self.nhid = nhid
#         self.nout = nout
#         self.relu = nn.ReLU()
#         self.ln = nn.LayerNorm((nout))
#         self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
#         self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
#         self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
#         self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
#         self.mol_hidden2 = nn.Linear(nhid, nout)

#     def forward(self, graph_batch):
#         x = graph_batch.x
#         edge_index = graph_batch.edge_index
#         batch = graph_batch.batch
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)
#         x = global_mean_pool(x, batch)
#         x = self.mol_hidden1(x).relu()
#         x = self.mol_hidden2(x)
#         return x

#------ Model Julie ------#

# class GraphEncoder(nn.Module):
#     def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
#         super(GraphEncoder, self).__init__()
#         self.nhid = nhid
#         self.nout = nout
#         self.relu = nn.ReLU()

#         self.conv1 = GCNConv(num_node_features, graph_hidden_channels[0])
#         self.conv2 = GCNConv(graph_hidden_channels[0], graph_hidden_channels[1])
#         self.conv3 = GCNConv(graph_hidden_channels[1], graph_hidden_channels[2])

#         self.mol_hidden1 = nn.Linear(graph_hidden_channels[2], nhid)
#         self.mol_hidden2 = nn.Linear(nhid, nout)

#         self.batchnorm1 = nn.BatchNorm1d(graph_hidden_channels[0])
#         self.batchnorm2 = nn.BatchNorm1d(graph_hidden_channels[1])
#         self.batchnorm3 = nn.BatchNorm1d(graph_hidden_channels[2])
#         self.batchnorm4 = nn.BatchNorm1d(graph_hidden_channels[2])

#         self.dropout = nn.Dropout(0.1)

#     def forward(self, graph_batch):
#         x = graph_batch.x
#         edge_index = graph_batch.edge_index
#         batch = graph_batch.batch
#         x = self.conv1(x, edge_index)
#         x = x.relu()

#         x = self.conv2(x, edge_index)
#         x = self.batchnorm2(x)
#         x = x.relu()

#         x = self.conv3(x,edge_index)
#         x = self.batchnorm4(x)
#         x = x.relu()

#         x = global_mean_pool(x, batch)
        
#         x = self.mol_hidden1(x).relu()
#         x = self.dropout(x)
#         x = self.mol_hidden2(x)

#         return x
    
# class TextEncoder(nn.Module):
#     def __init__(self, model_name):
#         super(TextEncoder, self).__init__()
#         self.bert = AutoModel.from_pretrained(model_name)
#         #self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        
#     def forward(self, input_ids, attention_mask):
#         encoded_text = self.bert(input_ids, attention_mask=attention_mask)
#         #print(encoded_text.last_hidden_state.size())
#         return encoded_text.last_hidden_state[:,0,:]
    
# class Model(nn.Module):
#     def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
#         super(Model, self).__init__()
#         self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
#         self.text_encoder = TextEncoder(model_name)
        
#     def forward(self, graph_batch, input_ids, attention_mask):
#         graph_encoded = self.graph_encoder(graph_batch)
#         text_encoded = self.text_encoder(input_ids, attention_mask)
#         return graph_encoded, text_encoded
    
#     def get_text_encoder(self):
#         return self.text_encoder
    
#     def get_graph_encoder(self):
#         return self.graph_encoder

#------ Model v2 ------#

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder, self).__init__()
        
        # GAT Layers
        self.conv1 = GATConv(num_node_features, graph_hidden_channels[0], heads=heads[0])
        self.conv2 = GATConv(graph_hidden_channels[0] * heads[0], graph_hidden_channels[1], heads=heads[1])
        self.conv3 = GATConv(graph_hidden_channels[1] * heads[1], graph_hidden_channels[2], heads=heads[2])

        # Batch Normalization and Dropout
        self.batchnorm1 = nn.BatchNorm1d(graph_hidden_channels[0] * heads[0])
        self.batchnorm2 = nn.BatchNorm1d(graph_hidden_channels[1] * heads[1])
        self.batchnorm3 = nn.BatchNorm1d(graph_hidden_channels[2] * heads[2])
        self.dropout = nn.Dropout(0.1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(graph_hidden_channels[2] * heads[2], nhid)
        self.fc2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        
        # Apply GAT layers with ReLU and Dropout
        x = F.relu(self.batchnorm1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm3(self.conv3(x, edge_index)))
        x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        #self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:,0,:]
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels, heads)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

    
#------ Model v3 ------#

# class GATBlock(nn.Module):
#     def __init__(self, in_dim, out_dim, num_heads):
#         super(GATBlock, self).__init__()
#         self.gat = GATConv(in_dim, out_dim // num_heads, heads=num_heads, concat=True)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(out_dim, out_dim * 4),
#             nn.ReLU(),
#             nn.Linear(out_dim * 4, out_dim)
#         )
#         self.norm1 = LayerNorm(out_dim)
#         self.norm2 = LayerNorm(out_dim)
#         self.dropout = nn.Dropout(0.1)
        
#         # Projection for the residual connection to match dimensions
#         self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

#     def forward(self, x, edge_index):
        
#         # Project input x to the correct dimension if needed
#         x_proj = self.proj(x)
        
#         # Multi-head attention
#         attn_output = self.gat(x, edge_index)
        
#         # Residual connection and layer normalization
#         x = self.norm1(attn_output + x_proj)
        
#         # Feed-forward network
#         ff_output = self.feed_forward(x)
        
#         # Another residual connection and layer normalization
#         x = self.norm2(ff_output + x)
#         x = self.dropout(x)
        
#         return x

# class GraphEncoder(nn.Module):
#     def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, num_heads=4, num_layers=3):
#         super(GraphEncoder, self).__init__()
#         self.layers = nn.ModuleList([
#             GATBlock(num_node_features if i == 0 else graph_hidden_channels, graph_hidden_channels, num_heads)
#             for i in range(num_layers)
#         ])
#         self.pool = global_mean_pool  # Global pooling layer
        
#         self.fc1 = nn.Linear(graph_hidden_channels, nhid)
#         self.fc2 = nn.Linear(nhid, nout)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         for layer in self.layers:
#             x = layer(x, edge_index)
        
#         # Apply global pooling to get graph-level output
#         x = self.pool(x, batch)

#         # Intermediate Embeddings
#         x = F.relu(self.fc1(x))

#         # Final Embeddings
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x

    
# class TextEncoder(nn.Module):
#     def __init__(self, model_name):
#         super(TextEncoder, self).__init__()
#         self.bert = AutoModel.from_pretrained(model_name)

#     def forward(self, input_ids, attention_mask):
#         # Pretrained model output
#         encoded_text = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         return encoded_text.last_hidden_state[:,0,:]  # Using the pooled output of BERT
        
    
# class Model(nn.Module):
#     def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, num_heads):
#         super(Model, self).__init__()
#         self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels, num_heads)
#         self.text_encoder = TextEncoder(model_name)
        
#     def forward(self, graph_batch, input_ids, attention_mask):
#         final_embeddings_graph = self.graph_encoder(graph_batch)
#         final_embeddings_text = self.text_encoder(input_ids, attention_mask)
#         return final_embeddings_graph, final_embeddings_text
    
#     def get_text_encoder(self):
#         return self.text_encoder
    
#     def get_graph_encoder(self):
#         return self.graph_encoder
