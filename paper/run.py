import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

from model import ModelForPaper
from data import MyDataModule


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


if __name__ == '__main__':
  # 乱数シードの固定
  seed = 42
  seed_everything(seed)

  # データの読み込み
  df = pd.read_csv('/Users/satasukeakira/Desktop/nlp/signate/paper/train.csv', index_col=0, engine='python')

  # データモジュールの作成
  MODEL_NAME_1 = 'allenai/longformer-base-4096'
  config_1 = {'model_name': MODEL_NAME_1, 
            'tokenizer': AutoTokenizer.from_pretrained(MODEL_NAME_1),
            'max_token_len': 4096,
            'text_column': 'abstract'}
  
  MODEL_NAME_2 = 'studio-ousia/luke-base'
  config_2 = {'model_name': MODEL_NAME_2, 
            'tokenizer': AutoTokenizer.from_pretrained(MODEL_NAME_2),
            'max_token_len': 384,
            'text_column': 'title'}
  
  config = {'model_1': config_1, 'model_2': config_2,
            'batch_size': 16, 'label_column': 'judgement', 'num_labels': 2}
  
  data_module = MyDataModule(df, config, seed)

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
  csv_logger = CSVLogger('logs/', name='model_paper')

  # モデルの作成
  trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=100,
    callbacks=[model_checkpoint, early_stopping],
    logger=csv_logger
  )

  model = ModelForPaper(config, lr=1e-5)

  trainer.fit(model, data_module)
  
  trainer.test(datamodule=data_module)

  print('ベストモデルのファイル: ', model_checkpoint.best_model_path)
  print('ベストモデルの検証データに対する損失: ', model_checkpoint.best_model_score)
