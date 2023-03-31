from lib import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from transformers import AutoModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torchmetrics
class ModelClassifier(pl.LightningModule):
    def __init__(self, model_name, num_labels, batch_size, learning_rate = 2e-5):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.accuracy = accuracy
        self.model = AutoModel.from_pretrained(self.model_name)

        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        for param in self.model.encoder.layer[:8].parameters():
            param.requires_grad = False
            
        self.gru = nn.GRU(self.model.config.hidden_size, self.model.config.hidden_size, batch_first=True)
        self.gru2 = nn.GRU(self.model.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)

        self.f1_metric = torchmetrics.F1(num_classes=self.num_labels)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_labels
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_labels
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")
