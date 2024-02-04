from torch import nn
import torch.nn.functional as F
import torch

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel
from torch_geometric.nn import GATConv
from torch_geometric.nn import GatedGraphConv
from torch.nn.functional import normalize
from torch.nn import Parameter, Sequential, Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, SGConv, global_add_pool, global_mean_pool

#------ Model v2 ------#

# class GraphEncoder(nn.Module):
#     def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
#         super(GraphEncoder, self).__init__()
        
#         # GAT Layers
#         self.conv1 = GATConv(num_node_features, graph_hidden_channels[0], heads=heads[0])
#         self.conv2 = GATConv(graph_hidden_channels[0] * heads[0], graph_hidden_channels[1], heads=heads[1])
#         self.conv3 = GATConv(graph_hidden_channels[1] * heads[1], graph_hidden_channels[2], heads=heads[2])

#         # Batch Normalization and Dropout
#         self.batchnorm1 = nn.BatchNorm1d(graph_hidden_channels[0] * heads[0])
#         self.batchnorm2 = nn.BatchNorm1d(graph_hidden_channels[1] * heads[1])
#         self.batchnorm3 = nn.BatchNorm1d(graph_hidden_channels[2] * heads[2])
#         self.dropout = nn.Dropout(0.1)

#         # Fully Connected Layers
#         self.fc1 = nn.Linear(graph_hidden_channels[2] * heads[2], nhid)
#         self.fc2 = nn.Linear(nhid, nout)

#     def forward(self, graph_batch):
#         x, edge_index, batch = graph_batch
        
#         # Apply GAT layers with ReLU and Dropout
#         x = F.relu(self.batchnorm1(self.conv1(x, edge_index)))
#         x = self.dropout(x)
#         x = F.relu(self.batchnorm2(self.conv2(x, edge_index)))
#         x = self.dropout(x)
#         x = F.relu(self.batchnorm3(self.conv3(x, edge_index)))
#         x = self.dropout(x)

#         # Global mean pooling
#         x = global_mean_pool(x, batch)
        
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x

# class TextEncoder(nn.Module):
#     def __init__(self, text_model_name, pretrained_text_path, mean_pooling=False):
#         super(TextEncoder, self).__init__()
#         self.bert = AutoModel.from_pretrained(text_model_name)

#         if pretrained_text_path is not None:
#             pretrained_dict = torch.load(pretrained_text_path, map_location='cpu')
#             filtered_pretrained_dict = {
#                 k[11:]: v for k, v in pretrained_dict.items() if k.startswith('distilbert.')
#             }
#             self.bert.load_state_dict(filtered_pretrained_dict, strict=False)    
#         self.mean_pooling = mean_pooling

#     def forward(self, input_ids, attention_mask):
#         encoded_text = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
#         if not self.mean_pooling:
#             return encoded_text.last_hidden_state[:,0,:]
        
#         last_hidden_state = encoded_text.last_hidden_state
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         mean_pooled_output = sum_embeddings / sum_mask

#         return mean_pooled_output

# class Model(nn.Module):
#     def __init__(
#         self,
#         num_node_features,
#         nout,
#         nhid,
#         graph_hidden_channels,
#         heads,
#         text_model_name,
#         pretrained_text_path=None,
#         mean_pooling=False
#     ):
#         super(Model, self).__init__()
#         self.graph_encoder = GraphEncoder(
#             num_node_features,
#             nout,
#             nhid,
#             graph_hidden_channels,
#             heads
#         )
#         self.text_encoder = TextEncoder(
#             text_model_name=text_model_name,
#             pretrained_text_path=pretrained_text_path,
#             mean_pooling=mean_pooling
#         )
        
#     def forward(self, graph_batch, input_ids, attention_mask):
#         graph_encoded = self.graph_encoder(graph_batch)
#         text_encoded = self.text_encoder(input_ids, attention_mask)
#         return graph_encoded, text_encoded
    
#     def get_text_encoder(self):
#         return self.text_encoder
    
#     def get_graph_encoder(self):
#         return self.graph_encoder

    
#------ Model v3 ------#

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder, self).__init__()
        
        # GAT Layers
        self.conv1 = GATConv(
            num_node_features, graph_hidden_channels[0], heads=heads[0], concat=False
        )
        self.conv2 = GATConv(
            graph_hidden_channels[0], graph_hidden_channels[1], heads=heads[1], concat=False
        )
        self.conv3 = GATConv(
            graph_hidden_channels[1], graph_hidden_channels[2], heads=heads[2], concat=False
        )

        # Batch Normalization and Dropout
        self.batchnorm1 = nn.BatchNorm1d(graph_hidden_channels[0])
        self.batchnorm2 = nn.BatchNorm1d(graph_hidden_channels[1])
        self.batchnorm3 = nn.BatchNorm1d(graph_hidden_channels[2])
        self.dropout = nn.Dropout(0.1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(graph_hidden_channels[2], nhid)
        # self.fc2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch
        
        # Apply GAT layers with ReLU and Dropout
        x = F.relu(self.batchnorm1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm3(self.conv3(x, edge_index)))
        x = self.dropout(x)

        # Global mean pooling
        global_rep = global_mean_pool(x, batch)
        
        # Normalize the output
        # global_rep = normalize(global_rep, p=2, dim=1)  # L2 normalization
        
        global_rep = self.fc1(global_rep)
        # global_rep = F.relu(self.fc1(global_rep))
        # global_rep = self.dropout(global_rep)
        # global_rep = self.fc2(global_rep)

        return global_rep

class TextEncoder(nn.Module):
    def __init__(self, text_model_name, pretrained_text_path, mean_pooling=False):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(text_model_name)

        if pretrained_text_path is not None:
            pretrained_dict = torch.load(pretrained_text_path, map_location='cpu')
            filtered_pretrained_dict = {
                k[11:]: v for k, v in pretrained_dict.items() if k.startswith('distilbert.')
            }
            self.bert.load_state_dict(filtered_pretrained_dict, strict=False)    
        self.mean_pooling = mean_pooling

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if not self.mean_pooling:
            emb_cls = encoded_text.last_hidden_state[:,0,:]
            # Normalize the output
            # emb_cls = normalize(emb_cls, p=2, dim=1)  # L2 normalization
            return emb_cls
        
        last_hidden_state = encoded_text.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_output = sum_embeddings / sum_mask

        return mean_pooled_output
    
class Model(nn.Module):
    def __init__(
        self,
        num_node_features,
        nout,
        nhid,
        graph_hidden_channels,
        heads,
        text_model_name,
        pretrained_text_path=None,
        mean_pooling=False
    ):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(
            num_node_features=num_node_features,
            nout=nout,
            nhid=nhid,
            graph_hidden_channels=graph_hidden_channels,
            heads=heads
        )
        self.text_encoder = TextEncoder(
            text_model_name=text_model_name,
            pretrained_text_path=pretrained_text_path,
            mean_pooling=mean_pooling
        )
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    
    def load_graph_encoder_weights(self, weights_path=None):
        """
        Load weights into the graph encoder part of the model.

        Args:
            weights_path (str): Path to the file containing the graph encoder weights.
        """
        if weights_path == None:
            return
        graph_encoder_state_dict = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in graph_encoder_state_dict:  # Handle nested dictionaries
            graph_encoder_state_dict = graph_encoder_state_dict['state_dict']
        self.graph_encoder.load_state_dict(graph_encoder_state_dict)
        print(f"Graph encoder weights loaded successfully from {weights_path}")
    
#------ Other Models ------#
      
    
# class TextEncoder(nn.Module):
#     def __init__(
#         self,
#         text_model_name='recobo/chemical-bert-uncased',
#         pretrained_text_path=None,
#         mean_pooling=False
#     ):
#         super(TextEncoder, self).__init__()
#         self.text_model_name = text_model_name
#         self.bert = AutoModel.from_pretrained(text_model_name)
        
#     def forward(self, input_ids, attention_mask):
#         encoded_text = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         return encoded_text.pooler_output
    

# class GraphEncoder(nn.Module):

#     def __init__(self, num_node_features, nhid, nout, graph_hidden_channels, n_layers=3, pool="mean", heads=1, bn=True, xavier=True):
#         super(GraphEncoder, self).__init__()

#     # Initialize ModuleList for batch normalization layers, even if bn is False
#         self.bns = torch.nn.ModuleList()  # Always define self.bns

#         self.convs = torch.nn.ModuleList()
#         self.acts = torch.nn.ModuleList()
#         self.n_layers = n_layers
#         self.pool = pool
        
#         a = torch.nn.ELU()
        
#         heads = heads[0]
#         for i in range(n_layers):
#             start_dim = nhid if i else num_node_features
#             conv = GATConv(start_dim, nhid, heads=heads, concat=False)
#             if xavier:
#                 self.weights_init(conv)
#             self.convs.append(conv)
#             self.acts.append(a)
#             if bn:  # Conditionally add batch normalization layers
#                 self.bns.append(BatchNorm1d(nhid))
#             else:  # If not using bn, append a placeholder that does nothing
#                 self.bns.append(torch.nn.Identity())  # Use Identity as a placeholder
                
#         self.fc1 = nn.Linear(n_layers*nhid, nhid)


#     def weights_init(self, module):
#         for m in module.modules():
#             if isinstance(m, GATConv):
#                 layers = [m.lin_src, m.lin_dst]
#             if isinstance(m, Linear):
#                 layers = [m]
#             for layer in layers:
#                 torch.nn.init.xavier_uniform_(layer.weight.data)
#                 if layer.bias is not None:
#                     layer.bias.data.fill_(0.0)

#     def forward(self, graph_batch):
#         x, edge_index, batch = graph_batch
#         xs = []
#         for i in range(self.n_layers):
#             x = self.convs[i](x, edge_index)
#             x = self.acts[i](x)
#             if self.bns is not None:
#                 x = self.bns[i](x)
#             xs.append(x)

#         if self.pool == "sum":
#             xpool = [global_add_pool(x, batch) for x in xs]
#         else:
#             xpool = [global_mean_pool(x, batch) for x in xs]
#         global_rep = torch.cat(xpool, 1)
#         global_rep = self.fc1(global_rep)
#         return global_rep