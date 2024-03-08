import numpy as np
from transformers import (BatchEncoding, AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from transformers.trainer_utils import set_seed
from datasets import Dataset, load_dataset
import os


# 乱数シードを42に固定
set_seed(42)

# モデル名からトークナイザを読み込む
model_name = "microsoft/deberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def token_head_tail(token_dict: dict, head: int = 411, tail: int = 100) -> list:
    """トークンの先頭と末尾を表示"""
    return {key: value[:head] + value[-tail:] for key, value in token_dict.items()}


def padding(token_dict: dict, max_length: int = 512) -> list:
    """トークンの長さをmax_lengthに合わせる"""
    return {key: value + [0] * (max_length - len(value)) for key, value in token_dict.items()}


def preprocess_text_classification(
    example: dict
) -> BatchEncoding:
    """文書分類の事例のテキストをトークナイズし、IDに変換"""
    encoded_example = tokenizer(example["sentence"], max_length=512)
    # 512トークンを超える場合は先頭と末尾を用いる
    if len(encoded_example["input_ids"]) > 512:
        encoded_example = token_head_tail(encoded_example)
    # 超えない場合は0でパディングする
    else:
        encoded_example = padding(encoded_example)
    # モデルの入力引数である"labels"をキーとして格納する
    encoded_example["labels"] = example["label"]
    return encoded_example


def compute_accuracy(
    eval_pred: tuple
) -> dict[str, float]:
    """予測ラベルと正解ラベルから正解率を計算"""
    predictions, labels = eval_pred
    # predictionsは各ラベルについてのスコア
    # 最もスコアの高いインデックスを予測ラベルとする
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


def main():
    # datasetを作成
    train_dataset = load_dataset("csv", data_files="./spam/data/processed/train.csv", split='train')
    valid_dataset = load_dataset("csv", data_files="./spam/data/processed/valid.csv", split='train')
    
    # データの前処理
    encoded_train_dataset = train_dataset.map(
        preprocess_text_classification,
        remove_columns=train_dataset.column_names,
    )
    encoded_valid_dataset = valid_dataset.map(
        preprocess_text_classification,
        remove_columns=valid_dataset.column_names,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # ラベルのIDとラベル名の対応を指定
    class_label = set([data["label"] for data in train_dataset])
    label2id = {label: id for id, label in enumerate(class_label)}
    id2label = {id: label for id, label in enumerate(class_label)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(class_label),
        label2id=label2id,  # ラベル名からIDへの対応を指定
        id2label=id2label,  # IDからラベル名への対応を指定
    )
    
    # modelディレクトリのディレクトリ数を取得する
    v_num = len(os.listdir("./spam/model/deberta")) + 1
    
    # 学習の設定
    training_args = TrainingArguments(
        output_dir=f"./spam/model/deberta/v{v_num}",  # 結果の保存フォルダ
        per_device_train_batch_size=16,  # 訓練時のバッチサイズ
        per_device_eval_batch_size=16,  # 評価時のバッチサイズ
        learning_rate=2e-5,  # 学習率
        lr_scheduler_type="linear",  # 学習率スケジューラの種類
        warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
        num_train_epochs=3,  # エポック数
        save_strategy="epoch",  # チェックポイントの保存タイミング
        logging_strategy="epoch",  # ロギングのタイミング
        evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
        load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
        metric_for_best_model="accuracy",  # 最良のモデルを決定する評価指標
        fp16=True,  # 自動混合精度演算の有効化
        dataloader_num_workers=2,
        dataloader_prefetch_factor=2,
    )
    
    # 学習
    trainer = Trainer(
        model=model,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_valid_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_accuracy,
        tokenizer=tokenizer,
    )
    trainer.train()
    

if __name__ == "__main__":
    main()