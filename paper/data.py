import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
from sklearn.model_selection import train_test_split


# データの読み込み
class MyDataset(Dataset):
  def __init__(self, df, config):
    self.df = df
    self.config = config

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    df_row = self.df.iloc[index]
    labels = df_row[self.config['label_column']]

    text_1 = df_row[self.config['model_1']['text_column']]
    encoding_1 = self.config['model_1']['tokenizer'](
      text_1,
      max_length = self.config['model_1']['max_token_len'],
      padding = 'max_length',
      return_tensors = 'pt',
      truncation = True # これがないとmax_lengthを超えたときに切り捨てができなくなってしまう。
    ) 

    text_2 = df_row[self.config['model_2']['text_column']]
    encoding_2 = self.config['model_2']['tokenizer'](
      text_2,
      max_length = self.config['model_2']['max_token_len'],
      padding = 'max_length',
      return_tensors = 'pt',
      truncation = True # これがないとmax_lengthを超えたときに切り捨てができなくなってしまう。
    ) 

    encoding = {}
    encoding['input_ids'] = torch.cat((encoding_1['input_ids'], encoding_2['input_ids']), dim=1)
    encoding['attention_mask'] = torch.cat((encoding_1['attention_mask'], encoding_2['attention_mask']), dim=1)

    encoding['labels'] = torch.tensor(labels)
    encoding = {k: torch.squeeze(v) for k, v in encoding.items()}

    # print(encoding)

    return encoding


# データモジュールの作成
class MyDataModule(pl.LightningDataModule):
  def __init__(self, df, config, seed):
    super().__init__()
    self.df_train, df_valid = train_test_split(df, test_size=0.4, random_state=seed)
    self.df_valid, self.df_test  = train_test_split(df_valid, test_size=0.5, random_state=seed)
    self.config = config
    self.batch_size = config['batch_size']
  
  def setup(self, stage):
    self.train_dataset = MyDataset(self.df_train, self.config)
    self.valid_dataset = MyDataset(self.df_valid, self.config)
    self.test_dataset = MyDataset(self.df_test, self.config)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

  def val_dataloader(self):
    return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())
