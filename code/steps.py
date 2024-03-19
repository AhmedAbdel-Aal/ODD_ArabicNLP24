import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import f1_score

def training_step(model, batch, optimizer, device):
        model.train()  # Set the model to training mode
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids, attention_mask, token_type_ids, labels = batch.values()
        labels = labels.to(device)
        optimizer.zero_grad()  # Clear previous gradients
        logits, embeddings = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # Forward pass
        loss = torch.nn.functional.cross_entropy(logits, labels)  # Compute loss
        loss.backward()  # Backpropagate the loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # Update model parameters
        return loss.item(), logits, embeddings

def eval_step(model, batch, device):
        model.eval()  # Set the model to evaluation mode
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids, attention_mask, token_type_ids, labels = batch.values()
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            predictions = torch.argmax(outputs, dim=1)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss.item(), predictions, labels
    
  