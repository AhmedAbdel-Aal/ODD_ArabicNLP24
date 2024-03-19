from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn

class CustomSequenceClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomSequenceClassificationModel, self).__init__()
        self.num_labels = num_labels
        
        # Load the configuration and model from pre-trained
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Create a classification head
        # Note: The size of the hidden state for the last layer can be obtained from the config (e.g., config.hidden_size for BERT-like models)
        self.classification_head = nn.Linear(self.config.hidden_size, num_labels)
        
        # Optionally add more layers such as dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def get_base_model(self):
        return self.base_model
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_features=True):
        # Get the outputs from the base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # For models like BERT, the first element in outputs is the last hidden state
        last_hidden_state = outputs[0]
        
        # Pool the outputs into a single mean vector for classification
        embeddings = last_hidden_state[:, 0]  # Take the embedding of the [CLS] token for classification tasks
                
        # Pass through the classification head to get the logits
        logits = self.classification_head(embeddings)

        if return_features:
            return logits, embeddings
        return logits