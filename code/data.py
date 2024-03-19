from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class ClassificationDataset(Dataset):
    def __init__(self, text, target, model_name, max_len, label_map):
      super(ClassificationDataset).__init__()
      """
      Args:
      text (List[str]): List of the training text
      target (List[str]): List of the training labels
      tokenizer_name (str): The tokenizer name (same as model_name).
      max_len (int): Maximum sentence length
      label_map (Dict[str,int]): A dictionary that maps the class labels to integer
      """
      self.text = text
      self.target = target
      self.tokenizer_name = model_name
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.max_len = max_len
      self.label_map = label_map


    def __len__(self):
      return len(self.text)

    def __getitem__(self,item):
      text = str(self.text[item])
      text = " ".join(text.split())

      inputs = self.tokenizer(
          text,
          max_length=self.max_len,
          padding='max_length',
          truncation=True,
          return_tensors='pt'
      )
      label = self.label_map[self.target[item]]
      input_ids, attention_masks,token_type_ids = inputs.values()
      return {'input_ids':input_ids.squeeze(), 'attention_mask':attention_masks.squeeze(), 'token_type_ids':token_type_ids.squeeze(), 'label':label} 
      #return InputFeatures(**inputs,label=self.label_map[self.target[item]])