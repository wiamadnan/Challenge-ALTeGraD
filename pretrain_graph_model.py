import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataloader import GraphDatasetPretrain, split_dataset, build_loader
from Model import GraphEncoder  
from loss import infonce, jensen_shannon, contrastive_loss
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config.config_parser import parse_args

# Load configurations
args = parse_args()

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate a unique directory name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
augment_str = '-'.join(args['augment_list'])
log_dir_name = f"{args['model_name']}--{augment_str}--{args['lr']}--{args['batch_size']}--{timestamp}"

args['tensorboard_dir'] = os.path.join(args['tensorboard_dir'], log_dir_name)
args['save_dir'] = os.path.join(args['save_dir'], log_dir_name)

# Initialize logging and TensorBoard writer
os.makedirs(args['save_dir'], exist_ok=True)
writer = SummaryWriter(log_dir=args['tensorboard_dir'])

# Load dataset
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True).item()
dataset = GraphDatasetPretrain(root='./data/', gt=gt, split='cids')
train_dataset, val_dataset, test_dataset = split_dataset(dataset, args['train_data_percent'])
train_loader = build_loader(args, train_dataset, "train")
val_loader = build_loader(args, val_dataset, "val")
test_loader = build_loader(args, test_dataset, "test")

# Model and optimizer
model = GraphEncoder(
    num_node_features=args['feat_dim'],
    nout=args['nout'],
    nhid=args['nhid'],
    graph_hidden_channels=args['graph_hidden_channels'],
    heads=args['heads']
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args['lr'])

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

def run(epoch, mode, dataloader):
    if mode == "train":
        model.train()
    else:
        model.eval()
    if args['loss'] in ['infonce', 'jensen_shannon']:
        contrastive_fn = eval(args['loss'] + "()")
    else: 
        contrastive_fn = contrastive_loss

    total_loss = 0
    for data in dataloader :
        data.to(device)
        readout_anchor = model((data.x_anchor, data.edge_index_anchor, data.x_anchor_batch))
        readout_positive = model((data.x_pos, data.edge_index_pos, data.x_pos_batch))
        loss = contrastive_fn(readout_anchor, readout_positive)
        
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * data.num_graphs

    avg_loss = total_loss / len(dataloader.dataset)
    writer.add_scalar(f'Loss 1/{mode}', avg_loss, epoch)
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning rate', current_lr, epoch)
    return avg_loss

for epoch in range(args['epochs']):
    train_loss = run(epoch, "train", train_loader)
    val_loss = run(epoch, "val", val_loader)
    print(f"Epoch {epoch}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")

    # Adjust learning rate here if needed
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()
    
    # Save model checkpoint
    is_best = val_loss < best_val_loss if epoch > 0 else True
    if is_best:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(args['save_dir'], "best_model.pth"))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, os.path.join(args['save_dir, "last_model.pth"))

# Load best model for testing
model.load_state_dict(torch.load(os.path.join(args['save_dir, "best_model.pth")))
test_loss = run(args['epochs, "test", test_loader)
print(f"Test Loss: {test_loss:.3f}")

# Close TensorBoard writer
writer.close()
