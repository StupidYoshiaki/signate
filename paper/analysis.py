import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
  df = pd.read_csv('/Users/satasukeakira/Desktop/nlp/signate/paper/train.csv', engine='python')

  # それぞれのカラムの文字数をカウント
  df['abstract_len'] = df['abstract'].str.len()
  df['title_len'] = df['title'].str.len()
  
  # 文字数の分布を可視化
  df['abstract_len'].hist(bins=100)
  df['title_len'].hist(bins=100)
  plt.tight_layout()
  plt.show()

  # 文字数の平均値を出力
  print(df['abstract_len'].mean())
  print(df['title_len'].mean())

  # 文字数の最大値を出力
  print(df['abstract_len'].max())
  print(df['title_len'].max())
