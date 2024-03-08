from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os


def token_head_tail(token_dict: dict, head: int = 411, tail: int = 100) -> list:
    """トークンの先頭と末尾を表示"""
    return {key: value[:head] + value[-tail:] for key, value in token_dict.items()}


def padding(token_dict: dict, max_length: int = 512) -> list:
    """トークンの長さをmax_lengthに合わせる"""
    return {key: value + [0] * (max_length - len(value)) for key, value in token_dict.items()}


def main():
    # モデルとトークナイザの読み込み
    model_dir = "./spam/model/deberta/v2"
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
            
        encoded_input = tokenizer(sentence)
        # 512トークンを超える場合は先頭と末尾を用いる
        if len(encoded_input["input_ids"]) > 512:
            encoded_input = token_head_tail(encoded_input)
        # 超えない場合は0でパディングする
        else:
            encoded_input = padding(encoded_input)
        # tensorに変換かつ次元を合わせる
        encoded_input = {key: torch.tensor(val).unsqueeze(0) for key, val in encoded_input.items()}
        
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()} # GPU対応
        output = model(**encoded_input)
        labels.append(output.logits.argmax().item())
        
    # 提出用データの作成
    df_sub["label"] = labels
    df_sub.to_csv(os.path.join(model_dir, "submit.csv"), index=False, header=False)
        

if __name__ == "__main__":
    main()