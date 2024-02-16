from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
from loss import contrastive_loss 
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from config.config_parser import parse_args

# Load configurations
args = parse_args() 

tokenizer = AutoTokenizer.from_pretrained(args['text_model_name'])
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = args['batch_size']
learning_rate = args['learning_rate']
nb_epochs = args['nb_epochs']
model_name = args['model_name']

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Model(
    text_model_name=args['text_model_name'],
    pretrained_text_path=args['pretrained_text_path'],
    num_node_features=args['num_node_features'],
    nout=args['nout'],
    nhid=args['nhid'],
    graph_hidden_channels=args['graph_hidden_channels'],
    heads=args['heads']
)
model.load_graph_encoder_weights(args['pretrained_graph_path'])
model.to(device)

# Set up logging directories
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir_name = f"{model_name}--{learning_rate}--{timestamp}"

args['save_dir'] = os.path.join(args['save_dir'], log_dir_name)
args['tensorboard_dir'] = os.path.join(args['tensorboard_dir'], log_dir_name)

os.makedirs(args['save_dir'], exist_ok=True)
os.makedirs(args['tensorboard_dir'], exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=args['tensorboard_dir'])

# Initialize the optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=args['learning_rate'],
    betas=tuple(args['optimizer']['betas']),
    weight_decay=args['optimizer']['weight_decay']
)

# Initialize the scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode=args['scheduler']['mode'],
    factor=args['scheduler']['factor'],
    patience=args['scheduler']['patience'],
    verbose=args['scheduler']['verbose']
)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 200
best_validation_loss = 1000000

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for batch in train_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        x_graph, x_text = model(
            (graph_batch.x.to(device), graph_batch.edge_index.to(device), graph_batch.batch.to(device)),
            input_ids.to(device),
            attention_mask.to(device)
        )
        current_loss = contrastive_loss(x_graph, x_text)
        writer.add_scalar('Loss/train', current_loss, count_iter)
        
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()
        
        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print(
                "Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(
                    count_iter,
                    time2 - time1,
                    loss/printEvery,
                )
            )
            losses.append(loss)
            loss = 0
            
    model.eval()       
    val_loss = 0
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        x_graph, x_text = model(
            (graph_batch.x.to(device), graph_batch.edge_index.to(device), graph_batch.batch.to(device)),
            input_ids.to(device),
            attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)
        val_loss += current_loss.item()
    best_validation_loss = min(best_validation_loss, val_loss)
    print(
        "-----EPOCH {0}----- done.  Validation loss: {1:.4f}".format(
            i+1,
            val_loss/len(val_loader)
        )
    )
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning Rate: {current_lr}")
    writer.add_scalar('Loss/val', val_loss/len(val_loader), i)
    writer.add_scalar('Learning rate', current_lr, i)
    
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()
    
    if best_validation_loss==val_loss:
        print('validation loss improved saving checkpoint...')
        # save_path = os.path.join('./', 'model'+str(i)+'.pt')
        save_path = os.path.join(args['save_dir'], 'best_model.pth.pt')
        torch.save(
            {
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_accuracy': val_loss/len(val_loader),
                'loss': loss,
            }, 
            save_path
        )
        print('checkpoint saved to: {}'.format(save_path))

print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model((batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv(f'submission_{best_validation_loss:.4f}.csv', index=False)

# Close TensorBoard writer
writer.close()