from lib import *
from customdata_Gramma import CustomerDataset
import wandb
import torchmetrics

class ModelClassifier(pl.LightningDataModule):
    def __init__(self, model_name, num_labels, batch_size = 32, learning_rate = 2e-5, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.accuracy = accuracy
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.embeddings.parameters():
            param.requires_grad = True
        for param in self.model.enconder.layer[:8].parameter():
            param.requires_grad = False
        
        self.gru = nn.GRU(self.model.config.hidden_size, self.model.config.hidden_size, batch_first = True)
        self.gru2 = nn.GRU(self.model.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.15)
        self.f1_metric = torchmetrics.F1(num_classes = self.num_labels)
        self.precision_marco_metric = torchmetrics.Precision(
            average= 'marco', num_classes= self.num_labels
        )
        self.recall_marco_metric = torchmetrics.Recall(
            average= "marco", num_classes= self.num_labels
        )
        self.precision_marco_metric = torchmetrics.Precision(average="micro")
        self.recall_marco_metric = torchmetrics.Recall(average= "micro")
    
    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)['last_hidden_state']
        gru_output, _ = self.gru(encoder_outputs)
        gru_output = self.relu(gru_output)
        gru_output, _ = self.gru2(gru_output)
        gru_output = self.relu(gru_output)
        avg_pooled = torch.mean(gru_output, 1)
        outputs = self.relu(self.fc2(avg_pooled))

        return outputs
    
    def training_step(self, batch, batch_idx):
        logits = self(batch[0], batch[1])
        loss = F.cross_entropy(logits, batch[2])
        preds = torch.argmax(logits, 1)
        labels = batch[2].long()
        accuracy = self.accuracy(preds, batch[2].long())
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        self.log('train/loss', loss, on_step = True, on_epoch = False, prog_bar = True)
        self.log('train/accuracy', accuracy, on_step = False, on_epoch = True, prog_bar=True)
        self.log("train/precision_macro", precision_macro, prog_bar=True)
        self.log("train/recall_macro", recall_macro, prog_bar=True)
        self.log("train/precision_micro", precision_micro, prog_bar=True)
        self.log("train/recall_micro", recall_micro, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits =self(batch[0], batch[1])
        loss =F.cross_entropy(logits, batch[2])
        preds = torch.argmax(logits, 1)
        labels = batch[2].long()
        accuracy = self.accuracy(preds, batch[2].long())
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)
        self.log('valid/loss', loss, on_step = False, on_epoch = True, prog_bar= True)
        self.log('valid/accuracy', accuracy, on_step = False, on_epoch = True,prog_bar= True)
        self.log("valid/precision_macro", precision_macro, on_step = False, on_epoch = True,prog_bar=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True)
        self.log("valid/f1", f1, prog_bar=True)
        return {'labels': labels, 'logits' : logits}
    
    def test_step(self, batch, batch_idx):
        logits = self(batch[:3])
        loss = F.cross_entropy(logits, batch[3])
        preds = torch.argmax(logits, 1)
        accuracy = self.accuracy(preds, batch[3].long())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_accuracy': accuracy}
    
    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )

    def setup(self, stage = None):
        train_dataloader = self.trainer.datamodule.train_dataloader()

        tb_size = self.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches*float(self.trainer.max_epochs)
        self.total_training_steps = (len(train_dataloader.dataset)// tb_size) // ab_size
    
    def configure_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {
            'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay':0.01
            },
            {
            'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr = self.learning_rate,
                          eps = 1e-5)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warm_step = 0,
            num_training_steps = self.total_training_steps
        )

        return [optimizer],[scheduler]

if __name__ == '__main__':
    model_name ='bert-base-uncased'
    text_field = 'Essay'
    label_field = 'LEXICAL'
    batch_size = 16
    max_len = 512
    data_frame = pd.read_csv('D:/My Projects/IELTS_Scoring/process_data.csv', index_col= 0)
    data_module = CustomerDataset(model_name, data_frame, text_field, label_field,max_len = 512, batch_size=batch_size)
    data_module.setup('fit')
    logger = pl.loggers.TensorBoardLogger('./save_model/lightning_logs', name = label_field)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath = './save_model/',
        filename = 'best_checkpoint_' + label_field + '_' + model_name,
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    print(len(data_module.label_encoder.classes_))

    model = ModelClassifier(model_name, len(data_module.label_encoder.classes_), data_module.batch_size, 7)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=20,
        gpus=AVAIL_GPUS)
    trainer.fit(model, datamodule=data_module)