import os
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class CustomDataset(pl.LightningDataModule):
  def __init__(self, model_name, data_frame, text_field: str, label_field: str, max_len = 512, batch_size = 32):
    super().__init__()
    self.model_name = model_name
    self.data_frame = data_frame
    self.text_field = text_field
    self.label_field = label_field
    self.max_len = max_len
    self.batch_size = batch_size
    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = LabelEncoder()
    self.label_encoder = label_encoder.fit(list(self.data_frame[self.label_field]))


  def setup(self, stage = None):
    self.data_frame['input'] = self.data_frame[self.text_field]
    self.data_frame['label'] = self.data_frame[self.label_field]

    self.train_dataset = self.convert_to_features(self.data_frame[self.data_frame.Data_type=='train'])
    self.val_dataset = self.convert_to_features(self.data_frame[(self.data_frame.Data_type == 'valid') | (self.data_frame.Data_type == 'test')])

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
        self.train_dataset,
        batch_size = self.batch_size,
        sampler = RandomSampler(self.train_dataset),
        num_workers=20
    )
  
  def val_dataloader(self) -> DataLoader:
    return DataLoader(
        self.val_dataset,
        batch_size = self.batch_size,
        sampler = RandomSampler(self.val_dataset),
        shuffle = False,
        num_workers=20
    )