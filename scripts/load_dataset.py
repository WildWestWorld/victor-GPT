import pandas as pd
from datasets import Dataset

def load_dataset(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 数据预处理
    def preprocess_function(examples):
        # 在这里执行任何你需要的预处理，如分词、去除停用词等
        return examples

    # 创建 Hugging Face 的 Dataset 对象
    dataset = Dataset.from_pandas(df)

    # 应用预处理函数
    dataset = dataset.map(preprocess_function)

    return dataset

if __name__ == "__main__":
    # 数据文件路径
    data_file = '../data/questions_answers.csv'

    # 加载数据集
    dataset = load_dataset(data_file)

    # 输出数据集信息
    print(dataset)