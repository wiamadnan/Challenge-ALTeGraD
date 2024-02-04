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

#------ Graph Encoder v1 ------#

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
        self.fc2 = nn.Linear(nhid, nout)

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
        global_rep = F.relu(self.fc1(global_rep))
        global_rep = self.dropout(global_rep)
        global_rep = self.fc2(global_rep)

        return global_rep

#------ Graph Encoder v2 ------#

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder, self).__init__()
        
        # GAT Layers
        self.conv1 = GATConv(
            num_node_features, graph_hidden_channels[0], heads=heads[0]
        )
        self.conv2 = GATConv(
            graph_hidden_channels[0] * heads[0], graph_hidden_channels[1], heads=heads[1]
        )
        self.conv3 = GATConv(
            graph_hidden_channels[1] * heads[1], graph_hidden_channels[2], heads=heads[2]
        )

        # Batch Normalization and Dropout
        self.batchnorm1 = nn.BatchNorm1d(graph_hidden_channels[0] * heads[0])
        self.batchnorm2 = nn.BatchNorm1d(graph_hidden_channels[1] * heads[1])
        self.batchnorm3 = nn.BatchNorm1d(graph_hidden_channels[2] * heads[2])
        self.dropout = nn.Dropout(0.1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(graph_hidden_channels[2] * heads[2], nhid)
        self.fc2 = nn.Linear(nhid, nout)

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
        global_rep = F.relu(self.fc1(global_rep))
        global_rep = self.dropout(global_rep)
        global_rep = self.fc2(global_rep)

        return global_rep
    
#------ TextEncoder ------#

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
    
#------ Model ------#
    
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
    