from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import json

def read_texts_from_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

def train_tokenizer():
    # 初始化 BPE 模型的分词器
    tokenizer = Tokenizer(models.BPE())

    # 设置预分词器为 ByteLevel
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 设置特殊标记
    special_tokens = ["<unk>", "<s>", "</s>"]
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,
        show_progress=True
    )

    # 读取数据并训练分词器
    data_path = '../data/tokenizer_train.jsonl'
    tokenizer.train_from_iterator(read_texts_from_jsonl(data_path), trainer=trainer)

    # 保存分词器
    tokenizer.save("../tokenizer/my_tokenizer")

if __name__ == "__main__":
    train_tokenizer()