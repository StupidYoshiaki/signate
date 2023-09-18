import torch
import pytorch_lightning as pl
from transformers import LongformerModel, LukeModel


class ModelForPaper(pl.LightningModule):
  def __init__(self, config, lr):
    super().__init__()
    self.save_hyperparameters()
    self.model_abstruct = LongformerModel.from_pretrained(config['model_1']['model_name'])
    self.model_title = LukeModel.from_pretrained(config['model_2']['model_name'])
    self.fc_abstruct = torch.nn.Linear(self.model_abstruct.config.hidden_size, config['num_labels'])
    self.fc_title = torch.nn.Linear(self.model_title.config.hidden_size, self.model_abstruct.config.hidden_size)
    
  def forward(self, x1, x2):
    x1 = self.model_abstruct(x1)
    x2 = self.model_title(x2)
    x2 = self.fc_title(x2)
    x = x1 + x2
    x = self.fc_abstruct(x)
    return x
  
  def training_step(self, batch, batch_idx):
    out = self(**batch)
    loss = out.loss
    self.log('train_loss', loss)
    return loss
  
  def validation_step(self, batch, batch_idx):
    out = self(**batch)
    val_loss = out.loss
    self.log('val_loss', val_loss)
    _, preds = torch.max(out, dim=1)
    acc = torch.sum(preds == batch['labels']).item() / len(preds)
    self.log('val_acc', acc)

  def test_step(self, batch, batch_idx):
    out = self(**batch)
    test_loss = out.loss
    self.log('test_loss', test_loss)
    _, preds = torch.max(out, dim=1)
    acc = torch.sum(preds == batch['labels']).item() / len(preds)
    self.log('test_acc', acc)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
  