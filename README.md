# Challenge-ALTeGraD
Molecule Retrieval with Natural Language Queries

## Overview

The objective of this project is to explore and implement machine learning techniques for retrieving molecules (graphs) based on natural language queries.
In this challenge, participants are provided with a text query and a list of molecules represented as graphs, with no additional reference or textual information about the molecules. The task is to identify and retrieve the molecule that corresponds to the given query.
We aim to develop a model capable of performing this task with promising performance.

## Content 

### main.py

To run the training pipeline:
```python
python main.py --load_config=config/train_config.yaml
```

### Model.py

Contains the implementation of the model, which includes the text encoder and graph encoder.

### dataloader.py

Loading the data.

### loss.py

Includes different loss functions. 
 
### pretrain_graph_model.py

To pretrain the graph encoder model, run the training pipeline:
```python
python pretrain_graph_model.py --load_config=config/pretrain_graph_model.yaml
```
### pretrain_text_model.py

To pretrain the text encoder model, run the training pipeline:
```python
python pretrain_text_model.py
```

### save_graph_names.py

Stores graph names for training, validation and test sets to be used for graph encoder pre-training.

### view_functions.py

Different strategies of data augmentation for pretraining graph 

## Note
Some code sections (view_functions, some functions and classes in dataloader.py, losses.py and pretrain_graph_model.py) related to pretraining the graph are sourced from this repository: https://github.com/paridhimaheshwari2708/GraphSSL.

## Dependencies

- torch
- torch_geometric
- transformers
