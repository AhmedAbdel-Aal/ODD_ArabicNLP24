import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def training_step(model, batch, optimizer):
        model.train()  # Set the model to training mode
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids, attention_mask, token_type_ids, labels = batch
        optimizer.zero_grad()  # Clear previous gradients
        logits, embeddings = self(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # Forward pass
        loss = torch.nn.functional.cross_entropy(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters
        return loss.item(), logits, embeddings

def eval_step(batch):
        model.eval()  # Set the model to evaluation mode
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids, attention_mask, token_type_ids, labels = batch
        with torch.no_grad():  # Disable gradient calculation
            outputs = self(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            predictions = torch.argmax(outputs, dim=1)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss.item(), predictions, labels
    
    
def train(train_dataloader, eval_dataloader, optimizer, epochs=3):
        
        best_dev_f1 = 0
        
        for epoch in range(epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            total_loss = 0
            # Training loop
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
                loss = training_step(batch, optimizer)
                total_loss += loss
            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss}")

            # Evaluation loop
            all_predictions = []
            all_labels = []
            total_eval_loss = 0
            for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}"):
                loss, predictions, labels = eval_step(batch)
                total_eval_loss += loss
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            eval_accuracy = accuracy_score(all_labels, all_predictions)
            
            eval_f1_score = 
            if eval_f1_score > best_dev_f1:
                best_dev_f1 = eval_f1_score
                best_model = copy.deepcopy(model)
        
            print(f"Average evaluation loss: {avg_eval_loss}")
            print(f"Evaluation accuracy: {eval_accuracy}")
            print(f"Evaluation macro-f1: {eval_f1_score}")