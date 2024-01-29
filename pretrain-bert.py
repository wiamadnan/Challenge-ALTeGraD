from transformers import DistilBertTokenizer, DistilBertForMaskedLM, get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from dataloader import TextDataset
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

model_name = 'distilbert-base-uncased'  # or 'distilbert-base-cased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 5
batch_size = 32
learning_rate = 2e-5

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # Probability of masking tokens
)

train_dataset = TextDataset(file_path='./data/train.tsv', tokenizer=tokenizer)
val_dataset = TextDataset(file_path='./data/val.tsv', tokenizer=tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
)

model = DistilBertForMaskedLM.from_pretrained(model_name)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * total_steps,  # 10% of total steps
    num_training_steps=total_steps
)

min_val_loss = float('inf')
for epoch in range(epochs):  # Number of epochs
    # Training loop
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch: {epoch}, Training Loss: {avg_train_loss:.4f}, Learning Rate: {current_lr}")

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch: {epoch}, Validation Loss: {avg_val_loss:.4f}")
    
    # Check if this is the best model so far
    if avg_val_loss < min_val_loss:
        print(f"Validation loss decreased ({min_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
        min_val_loss = avg_val_loss
        # Save the model
        torch.save(model.state_dict(), './distilbert_pretrained2.bin')
        
        # Save only the base DistilBert model
        base_model_state_dict = {
            k[len('distilbert.'):]: v for k, v in model.state_dict().items() if k.startswith('distilbert.')
        }
        torch.save(base_model_state_dict, f'./{model_name}.bin')
