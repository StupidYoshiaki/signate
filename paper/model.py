import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments,
    LukeTokenizer, LukeForSequenceClassification,
    pipeline
)
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split


# 乱数シードの固定
def seed_everything(seed=42):
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  generator = torch.Generator()
  generator.manual_seed(seed)


# データの読み込み
class LukeDataset(Dataset):
  def __init__(self, df, tokenizer, max_token_len, text_column, label_column):
    self.df = df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
    self.text_column = text_column
    self.label_column = label_column

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    df_row = self.df.iloc[index]
    text = df_row[self.text_column]
    labels = df_row[self.label_column]

    encoding = self.tokenizer(
      text,
      max_length = self.max_token_len,
      padding = 'max_length',
      return_tensors = 'pt',
      truncation = True # これがないとmax_lengthを超えたときに切り捨てができなくなってしまう。
    ) 
    encoding['labels'] = torch.tensor(labels)
    encoding = {k: torch.squeeze(v) for k, v in encoding.items()}
    return encoding


# データモジュールの作成
class LukeDataModule(pl.LightningDataModule):
  def __init__(self, df_train, df_valid, df_test, tokenizer, batch_size, max_token_len, text_column, label_column):
    super().__init__()
    self.df_train = df_train
    self.df_valid = df_valid
    self.df_test = df_test
    self.batch_size = batch_size
    self.max_token_len = max_token_len
    self.tokenizer = tokenizer
    self.text_column = text_column
    self.label_column = label_column
  
  def setup(self, stage):
    self.train_dataset = LukeDataset(self.df_train, self.tokenizer, self.max_token_len, self.text_column, self.label_column)
    self.valid_dataset = LukeDataset(self.df_valid, self.tokenizer, self.max_token_len, self.text_column, self.label_column)
    self.test_dataset = LukeDataset(self.df_test, self.tokenizer, self.max_token_len, self.text_column, self.label_column)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

  def val_dataloader(self):
    return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())


if __name__ == '__main__':
  # 乱数シードの固定
  seed = 42
  seed_everything(seed)

  # データの読み込み
  df = pd.read_csv('train.csv', index_col=0, engine='python')
  df_train, df_valid = train_test_split(df, test_size=0.4, random_state=seed)
  df_valid, df_test = train_test_split(df_valid, test_size=0.5, random_state=seed)

  # データモジュールの作成
  MODEL_NAME = 'studio-ousia/luke-japanese-base-lite'
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  text_column = 'abstract'
  label_column = 'judgement'
  batch_size = 16
  max_token_len = 128
  data_module = LukeDataModule(df_train, df_valid, df_test, 
                               tokenizer, batch_size, max_token_len, text_column, label_column)

  # モデルの重みを保存する条件を保存
  model_checkpoint = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model_mask/'
  )

  # early_stopping
  early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10 # patienceってエポック数のこと. だからepoch=10でpatience=10だと当然全部学習して終わる
  )

  # csvファイルでログを保存
  csv_logger = CSVLogger('log/', name='paper_model')

  # モデルの作成
  trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=100,
    callbacks=[model_checkpoint, early_stopping],
    logger=csv_logger
  )

  model = LukeForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

  trainer.fit(model, data_module)
  
  trainer.test(datamodule=data_module)

  print('ベストモデルのファイル: ', model_checkpoint.best_model_path)
  print('ベストモデルの検証データに対する損失: ', model_checkpoint.best_model_score)
