from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os



def main():
    # モデルとトークナイザの読み込み
    model_dir = "./spam/model/deberta/v1"
    checkpoint = "checkpoint-390"
    model_name = os.path.join(model_dir, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # GPUの設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 提出用データの読み込み
    df_sub = pd.read_csv("./spam/data/raw/sample_submit.csv", names=["file", "label"])
    
    labels = []
    files = df_sub["file"].values
    for file in files:
        file_path = os.path.join("./spam/data/raw/test", file)
        with open(file_path, "r") as f:
            sentence = f.read()
        encoded_input = tokenizer(sentence, return_tensors="pt", max_length=512)
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()} # GPU対応
        output = model(**encoded_input)
        labels.append(output.logits.argmax().item())
        
    # 提出用データの作成
    df_sub["label"] = labels
    df_sub.to_csv(os.path.join(model_dir, "submit.csv"), index=False, header=False)
        

if __name__ == "__main__":
    main()