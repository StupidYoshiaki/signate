import pandas as pd

# データの読み込み
df_bert = pd.read_csv("./spam/model/bert/v1/submit.csv", names=["file", "label"])
df_roberta = pd.read_csv("./spam/model/roberta/v1/submit.csv", names=["file", "label"])
df_deberta = pd.read_csv("./spam/model/deberta/v1/submit.csv", names=["file", "label"])

# アンサンブル
df_ensemble = pd.DataFrame()
df_ensemble["file"] = df_bert["file"]
df_ensemble["label"] = (df_bert["label"] + df_roberta["label"] + df_deberta["label"]) // 3

# 提出用データの作成
df_ensemble.to_csv("./spam/model/ensembled/submit.csv", index=False, header=False)