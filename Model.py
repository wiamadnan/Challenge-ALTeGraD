from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels[0])
        self.conv2 = GCNConv(graph_hidden_channels[0], graph_hidden_channels[1])
        self.conv3 = GCNConv(graph_hidden_channels[1], graph_hidden_channels[2])
        self.conv4= GatedGraphConv(graph_hidden_channels[2],graph_hidden_channels[2])
        self.mol_hidden1 = nn.Linear(graph_hidden_channels[2], nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
        self.batchnorm1 = nn.BatchNorm1d(graph_hidden_channels[0])
        self.batchnorm2 = nn.BatchNorm1d(graph_hidden_channels[1])
        self.batchnorm3 = nn.BatchNorm1d(graph_hidden_channels[2])
        self.batchnorm4 = nn.BatchNorm1d(nhid)
        self.dropout = nn.Dropout(0.1)
        self.conv1d=nn.Conv1d(graph_hidden_channels[2], graph_hidden_channels[2]//2,2)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.conv3(x,edge_index)
        x = self.batchnorm4(x)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.dropout(x)
        x = self.mol_hidden2(x)
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
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
