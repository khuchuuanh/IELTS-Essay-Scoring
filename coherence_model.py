from lib import *

class ModelClassifier(pl.LightningModule):
    def __init__(self, model_name, num_labels, batch_size, learning_rate=2e-5, hidden_size=512, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.model = AutoModel.from_pretrained(model_name)
        self.gru = nn.GRU(self.model.config.hidden_size, hidden_size, batch_first=True)
        self.sentence_gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.out = torch.nn.Softmax(dim=1)

        for param in self.model.encoder.layer[:8].parameters():
            param.requires_grad = False
    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        gru_output, _ = self.gru(encoder_outputs)
        gru_output = self.relu(gru_output)
        gru_output_sentence, _ = self.sentence_gru(gru_output)
        gru_output_sentence = self.relu(gru_output_sentence)
        avg_pooled = torch.mean(gru_output_sentence, 1)
        fc_output = self.fc(avg_pooled)
        outputs = self.relu(fc_output)

        return outputs
    
    def training_step(self, batch, batch_idx):
        logits = self(batch[0], batch[1])
        loss = F.cross_entropy(logits, batch[2])
        preds = torch.argmax(logits, 1)
        accuracy = torch.eq(preds, batch[2].long()).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch[0], batch[1])
        loss = F.cross_entropy(logits, batch[2])
        preds = torch.argmax(logits, 1)
        accuracy = torch.eq(preds, batch[2].long()).float().mean()

        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_accuracy': accuracy}
