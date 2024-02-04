from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
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

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

model_name = 'distilbert-base-uncased'
pretrained_path = 'distilbert-base-uncased.bin'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 50
batch_size = 32
learning_rate = 2e-5

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#------ Default model ------#
# model = Model(
#     model_name=model_name,
#     num_node_features=300,
#     nout=768,
#     nhid=300,
#     graph_hidden_channels=300
# ) # nout = bert model hidden dim

#------ Model Julie ------#
# model = Model(
#     model_name=model_name,
#     num_node_features=300,
#     nout=256,
#     nhid=512,
#     graph_hidden_channels=[500, 400, 300]
# )

#------ Model v2 ------#
# model = Model(
#     model_name=model_name,
#     pretrained_path=pretrained_path,
#     num_node_features=300,
#     nout=768,
#     nhid=512,
#     graph_hidden_channels=[512, 512, 512],
#     heads=[4, 4, 4]
# )

#------ Model v3 ------#
model = Model(
    model_name=model_name,
    pretrained_path=pretrained_path,
    num_node_features=300,
    nout=768,
    nhid=768,
    graph_hidden_channels=[768, 768, 768],
    heads=[4, 4, 4]
)

# model.load_graph_encoder_weights('./best_model.pth')

# Set up logging directories
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir_name = f"modelv3--gat--786--bert--pretrained--{learning_rate}--{timestamp}"
tensorboard_dir = os.path.join('./logs/tensorboard', log_dir_name)
save_dir = os.path.join('./logs/models', log_dir_name)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=tensorboard_dir)

# Model, optimizer, and scheduler setup
model.to(device)
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
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
            graph_batch.to(device),
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
            graph_batch.to(device),
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
        save_path = os.path.join(save_dir, 'best_model.pth.pt')
        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss/len(val_loader),
        'loss': loss,
        }, save_path)
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
    for output in graph_model(batch.to(device)):
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