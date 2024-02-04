import os
import os.path as osp
import torch
from torch_geometric.data import Dataset 
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch, Dataset
from view_functions import *
import pandas as pd

class GraphTextDataset(Dataset):
    def __init__(self, root, gt, split, tokenizer=None, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.description = pd.read_csv(os.path.join(self.root, split+'.tsv'), sep='\t', header=None)   
        self.description = self.description.set_index(0).to_dict()
        self.cids = list(self.description[1].keys())
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphTextDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
        edge_index  = []
        x = []
        with open(raw_path, 'r') as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = *map(int, line.split()),
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f: #get mol2vec features:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt['UNK'])
            return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            text_input = self.tokenizer([self.description[1][cid]],
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=256,
                                   padding="max_length",
                                   add_special_tokens=True,)
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index, input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    
class GraphDataset(Dataset):
    def __init__(self, root, gt, split, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(os.path.join(self.root, split+'.txt'), sep='\t', header=None)
        self.cids = self.description[0].tolist()
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
        edge_index  = []
        x = []
        with open(raw_path, 'r') as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = *map(int, line.split()),
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt['UNK'])
            return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    def get_idx_to_cid(self):
        return self.idx_to_cid
    
class TextDataset(TorchDataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = self.load_sentences(file_path)

    def load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
    
#----------- Code for pretraining graph encoders -----------#
    
DATA_SPLIT = [0.7, 0.2, 0.1] # Train / val / test split ratio

class GraphDatasetPretrain(Dataset):
    def __init__(self, root, gt, split, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(os.path.join(self.root, split+'.txt'), sep='\t', header=None)
        self.cids = self.description[0].tolist()
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDatasetPretrain, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
    
    def update_split_file(self):
        # Update the split file with the successfully processed cids
        updated_split_path = os.path.join(self.root, self.split + '.txt')
        self.description.to_csv(updated_split_path, sep='\t', index=False, header=None)
       
        
    def process_graph(self, raw_path):
        edge_index  = []
        x = []
        with open(raw_path, 'r') as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = *map(int, line.split()),
                    edge_index.append(edge)
                else:
                    break
            next(f)
            for line in f:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt['UNK'])
            return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        processed_cids = []
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            # Check if the graph has at least two edges
            if edge_index.size(0) == 2 and edge_index.size(1) >= 2:  # Ensuring the correct shape
                data = Data(x=x, edge_index=edge_index)
                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
                processed_cids.append(cid)
                i += 1
            else:
                print(f"Skipping graph {cid} due to insufficient edges: {edge_index.size(), edge_index} edges found.")
        self.cids = processed_cids
        self.description = pd.DataFrame(self.cids)
        self.update_split_file()
        self.idx_to_cid = {i: cid for i, cid in enumerate(self.cids)}

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    def get_idx_to_cid(self):
        return self.idx_to_cid
    
    
def split_dataset(dataset, train_data_percent=1.0):
    """
    Splits the data into train / val / test sets.
    Args:
        dataset (list): all graphs in the dataset.
        train_data_percent (float): Fraction of training data which is labelled. (default 1.0)
    """

    # random.shuffle(dataset)

    n = len(dataset)
    train_split, val_split, test_split = DATA_SPLIT

    train_end = int(n * DATA_SPLIT[0])
    val_end = train_end + int(n * DATA_SPLIT[1])
    train_label_percent = int(train_end * train_data_percent)
    train_dataset, val_dataset, test_dataset = [i for i in dataset[:train_label_percent]], [i for i in dataset[train_end:val_end]], [i for i in dataset[val_end:]]
    return train_dataset, val_dataset, test_dataset

def build_loader(args, dataset, subset):
    shuffle = (subset != "test")
    loader = DataLoader(MyDataset(dataset, subset, args.augment_list),
                        num_workers=args.num_workers, batch_size=args.batch_size, 
                        shuffle=shuffle, follow_batch=["x_anchor", "x_pos"])
    return loader


def build_classification_loader(args, dataset, subset):
    shuffle = (subset != "test")
    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=shuffle)
    return loader


class MyDataset(Dataset):
    """
    Dataset class that returns a graph and its augmented view in get() call.
    Augmentations are applied sequentially based on the augment_list.
    """

    def __init__(self, dataset, subset, augment_list):
        super(MyDataset, self).__init__()

        self.dataset = dataset
        self.augment_list = augment_list

        self.augment_functions = []
        for augment in self.augment_list:
            if augment == "edge_perturbation":
                function = EdgePerturbation()
            elif augment == "diffusion":
                function = Diffusion()
            elif augment == "diffusion_with_sample":
                function = DiffusionWithSample()
            elif augment == "node_dropping":
                function = UniformSample()
            elif augment == "random_walk_subgraph":
                function = RWSample()
            elif augment == "node_attr_mask":
                function = NodeAttrMask()
            self.augment_functions.append(function)

        print("# samples in {} subset: {}".format(subset, len(self.dataset)))

    def get_positive_sample(self, current_graph):
        """
        Possible augmentations include the following:
            edge_perturbation()
            diffusion()
            diffusion_with_sample()
            node_dropping()
            random_walk_subgraph()
            node_attr_mask()
        """

        graph_temp = current_graph
        for function in self.augment_functions:
            graph_temp = function.views_fn(graph_temp)
        return graph_temp

    def get(self, idx):
        graph_anchor = self.dataset[idx]
        assert graph_anchor.edge_index.size(0) == 2 or graph_anchor.edge_index.numel() == 0, f"Edge index shape mismatch: {graph_anchor.edge_index.size()}"

        graph_pos = self.get_positive_sample(graph_anchor)
        return PairData(graph_anchor.edge_index, graph_anchor.x, graph_pos.edge_index, graph_pos.x)

    def len(self):
        return len(self.dataset)


class PairData(Data):
    """
    Utility function to return a pair of graphs in dataloader.
    Adapted from https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    """

    def __init__(self, edge_index_anchor = None, x_anchor = None, edge_index_pos = None, x_pos = None):
        super().__init__()
        self.edge_index_anchor = edge_index_anchor
        self.x_anchor = x_anchor
        
        self.edge_index_pos = edge_index_pos
        self.x_pos = x_pos

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_anchor":
            return self.x_anchor.size(0)
        if key == "edge_index_pos":
            return self.x_pos.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

