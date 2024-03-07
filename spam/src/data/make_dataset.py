import csv
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    txt_path = "./spam/data/raw/train"
    label_path = "./spam/data/raw/train_master.tsv"
    csv_path = "./spam/data/processed/train.csv"
    
    # csvファイルへの書き込み
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sentence", "label"])
        for file in os.listdir(txt_path):
            with open(os.path.join(txt_path, file), "r") as f:
                sentence = f.read()
            with open(label_path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if row[0] == file:
                        label = row[1]
                        break
            file = file.replace(".txt", "")
            file = file.replace("train_", "")
            file = int(file)
            writer.writerow([file, sentence, label])
            
    # csvファイルの読み込み
    df = pd.read_csv(csv_path)
    
    # trainとvalidに分割
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # idの昇順にソート
    train_df = train_df.sort_values("id")
    valid_df = valid_df.sort_values("id")
    
    # csvファイルへの書き込み
    train_csv_path = "./spam/data/processed/train.csv"
    valid_csv_path = "./spam/data/processed/valid.csv"
    train_df.to_csv(train_csv_path, index=False)
    valid_df.to_csv(valid_csv_path, index=False)
            

if __name__ == "__main__":
    main()