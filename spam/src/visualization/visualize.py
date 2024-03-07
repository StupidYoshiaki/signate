from collections import Counter
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import sys

plt.rcParams["font.size"] = 12  # 文字サイズを大きくする

def visualize_text_length(dataset: Dataset, tokenizer):
    """データセット中のテキストのトークン数の分布をグラフとして描画"""
    # データセット中のテキストの長さを数える
    length_counter = Counter()
    for data in tqdm(dataset):
        length = len(tokenizer.tokenize(data["sentence"]))
        length_counter[length] += 1
        
    # テキストの長さを100の倍数に丸める
    # new_length_counter = Counter()
    # for length, num in length_counter.items():
    #     new_length = length // 100 * 100
    #     new_length_counter[new_length] += num
    # length_counter = new_length_counter
    # print(length_counter)
        
    # length_counterの値から棒グラフを描画する
    fig = plt.figure(figsize=(8,6))
    plt.bar(length_counter.keys(), length_counter.values(), width=1.0)
    plt.xlabel("token_num")
    plt.ylabel("num")
    plt.savefig("./spam/report/text_length.png", bbox_inches='tight')
    plt.show()


def visualize_labels(dataset: Dataset, tokenizer):
    """データセット中のラベル分布をグラフとして描画"""
    # データセット中のラベルの数を数える
    label_counter = Counter()
    for data in dataset:
        label_name = data["label"]
        label_counter[label_name] += 1
    # label_counterを棒グラフとして描画する
    fig = plt.figure(figsize=(8,6))
    plt.bar(label_counter.keys(), label_counter.values(), tick_label=list(label_counter.keys()))
    plt.xlabel("label")
    plt.ylabel("num")
    plt.savefig("./spam/report/label.png", bbox_inches='tight')
    plt.show()
    
    
def visualize_text_length_for_each_label(dataset: Dataset, tokenizer):
    """データセット中のテキストのトークン数の分布をラベルごとにグラフとして描画"""
    # データセット中のテキストの長さを数える
    length_counter = {}
    for data in dataset:
        label_name = data["label"]
        if label_name not in length_counter:
            length_counter[label_name] = Counter()
        length = len(tokenizer.tokenize(data["sentence"]))
        length_counter[label_name][length] += 1
        
    # テキストの長さを100の倍数に丸める
    # new_length_counter = Counter()
    # for label_name, counter in length_counter.items():
    #     new_length_counter[label_name] = Counter()
    #     for length, num in counter.items():
    #         new_length = length // 100 * 100
    #         new_length_counter[label_name][new_length] += num
    # length_counter = new_length_counter
    
    # length_counterの値から棒グラフを描画する
    for label_name, counter in length_counter.items():
        fig = plt.figure(figsize=(8,6))
        plt.bar(counter.keys(), counter.values(), width=1.0)
        plt.xlabel("token_num")
        plt.ylabel("num")
        plt.title(label_name)
        plt.savefig("./spam/report/text_length_{}.png".format(label_name), bbox_inches='tight')
        plt.show()
    

def visualize_statistics(dataset: Dataset, tokenizer):
    """データセット中のテキストのトークン数の統計量を表示"""
    # データセット中のテキストの長さを数える
    length_list = []
    for data in dataset:
        length = len(tokenizer.tokenize(data["sentence"]))
        length_list.append(length)
    # 統計量をtxtファイルに書き込む
    with open("./spam/report/statistics.txt", "w") as f:
        f.write("mean: {}\n".format(sum(length_list) / len(length_list)))
        f.write("median: {}\n".format(sorted(length_list)[len(length_list) // 2]))
        f.write("max: {}\n".format(max(length_list)))
        f.write("min: {}\n".format(min(length_list)))
    
    
def main():
    # モデル名からトークナイザを読み込む
    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # datasetを作成
    dataset_path = "./spam/data/processed"
    train_dataset = load_dataset(dataset_path, split='train')
    
    # 可視化
    visualize_text_length(train_dataset, tokenizer)
    visualize_labels(train_dataset, tokenizer)
    visualize_statistics(train_dataset, tokenizer)
    visualize_text_length_for_each_label(train_dataset, tokenizer)
    
if __name__ == "__main__":
    main()