from lib import *
from task_response import CustomDataset
class ModelClassifier(pl.LightningModule):
  def __init__(self, model_name, num_labels, batch_size, learning_rate = 2e-5, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.accuracy = accuracy
    self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    for param in self.model.transformer.layer[:10].parameters():
      param.requires_grad = False
    self.fc1 = nn.Linear(self.model.config.hidden_size, 256)
    self.fc2 = nn.Linear(256, num_labels)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.1)

  def forward(self, batch):
    input_ids = batch[0]
    attention_mask = batch[1]
    encoder_outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
    h_cls = encoder_outputs.last_hidden_state[:, 0]
    outputs = self.fc1(self.relu(h_cls))
    outputs = self.fc2(outputs)
    return outputs

  def training_step(self, batch, batch_idx):
    logits = self(batch[:2])
    loss = F.cross_entropy(logits, batch[2])
    preds = torch.argmax(logits, 1)
    accuracy = self.accuracy(preds, batch[2].long())
    self.log('train_loss', loss, on_step = False, on_epoch = True, prog_bar = True)
    self.log('train_accuracy', accuracy, on_step = False, on_epoch = True, prog_bar = True)

    return loss

  def validation_step(self, batch, batch_idx):
    logits =self(batch[:2])
    loss =F.cross_entropy(logits, batch[2])
    preds = torch.argmax(logits, 1)
    accuracy = self.accuracy(preds, batch[2].long())
    self.log('val_loss', loss, on_step = False, on_epoch = True, prog_bar = True)
    self.log('val_accuracy', accuracy, on_step = False, on_epoch = True, prog_bar =True)

    return {'val_loss': loss, 'val_accuracy' : accuracy}

  def test_step(self, batch, batch_idx):
    logits = self(batch[:2])
    loss = F.cross_entropy(logits, batch[2])
    preds = torch.argmax(logits, 1)
    accuracy = self.accuracy(preds, batch[2].long())
    self.log('test_loss', loss, on_step = False, on_epoch = True, prog_bar = True) 
    self.log('test_accuracy', accuracy, on_step =False, on_epoch = True, prog_bar = True)
    
    return {'val_loss' : loss, 'val_accuracy' : accuracy}

  def validation_epoch_end(self,validation_step_outputs):
    avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
    avg_accuracy =torch.stack([x['val_accuracy']for x in validation_step_outputs]).mean()
    self.log('val_loss', avg_loss, prog_bar = True, logger = True)
    self.log('val_accuracy',avg_accuracy, prog_bar =True, logger = True)

    return {
        'val_loss' : avg_loss,
        'val_accuracy' : avg_accuracy
    }


  def setup(self, stage = None):
    train_dataloader = self.trainer.datamodule.train_dataloader()
    tb_size = self.batch_size * max(1, self.trainer.gpus)
    ab_size = self.trainer.accumulate_grad_batches*float(self.trainer.max_epochs)
    self.total_training_steps = (len(train_dataloader.dataset)// tb_size) // ab_size

  def configure_optimizers(self):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params':[p for n, p in self.model.named_parmeters() if not any(nd in n for nd in no_decay)],
            'weght_decay' : 0.01
        },
        {
            'params' :[p for n, p in self.named_parameter() if any(nd in n for nd in no_decay)],
            'weight_decay' : 0.0
        }  
    ] 

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr = self.learning_rate,
                      eps = 1e-8
                      )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_step = 0,
        num_training_steps = self.total_training_steps
    )
    return [optimizer],[scheduler]

if __name__ == '__manin__':
    model_name  = 'bert-base-uncased'
    text_field = 'input'
    label_field = 'TASK '
    data_frame = pd.read_csv('D:/My Projects/IELTS_Scoring/data_split.py')
    data_module = CustomDataset(model_name, data_frame, text_field, label_field, max_len =512, batch_size = 32, remove_special_characters = True)
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

    #print(len(data_module.label_encoder.classes_))
    model = ModelClassifier(model_name, len(data_module.label_encoder.classes_), data_module.batch_size)
    trainer = pl.Trainer(
    logger=logger,
    callbacks=[early_stopping_callback, checkpoint_callback],
    max_epochs=20, deterministic=True,)
    trainer.fit(model, datamodule=data_module)