from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
from configuration import config


def process_data():
    print("开始处理数据")
    # 读取文件
    dataset_dict = load_dataset('csv', data_files={
        'train': str(config.RAW_DATA_DIR / 'train.txt'),
        'test': str(config.RAW_DATA_DIR / 'test.txt'),
        'valid': str(config.RAW_DATA_DIR / 'valid.txt')

    }, delimiter='\t')

    # 过滤数据
    dataset_dict = dataset_dict.filter(lambda x: x['label'] is not None and x['text_a'] is not None)
    # 构建数据集
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')

    # 统计标题长度
    # df = dataset_dict['train'].to_pandas()
    # print(df['text_a'].apply(lambda x: len(tokenizer.tokenize(x))).max())
    # 统计的长度内容添加到config.py中的seq_len中

    def tokenize(batch):
        tokenized = tokenizer(batch['text_a'], truncation=True, padding='max_length', max_length=config.SEQ_LEN)
        return {'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask']}

    # 处理text_a列
    dataset_dict = dataset_dict.map(tokenize, batched=True, remove_columns=['text_a'])

    # 处理label列 - 先获取所有唯一标签，再创建ClassLabel并转换
    all_labels = (dataset_dict['train'].unique('label'))
    print(f"所有标签 ({len(all_labels)}个): {all_labels}")
    # 创建ClassLabel特征
    class_label_feature = ClassLabel(names=all_labels)
    # 使用map函数将字符串标签转换为ClassLabel
    def convert_label(example):
        example['label'] = class_label_feature.str2int(example['label'])
        return example
    dataset_dict = dataset_dict.map(convert_label)
    # 设置feature类型为ClassLabel
    dataset_dict = dataset_dict.cast_column('label', class_label_feature)

    #保存数据集
    dataset_dict['train'].save_to_disk(config.PROCESSED_DATA_DIR / 'train')
    dataset_dict['test'].save_to_disk(config.PROCESSED_DATA_DIR / 'test')
    dataset_dict['valid'].save_to_disk(config.PROCESSED_DATA_DIR / 'valid')


    print(dataset_dict['train'].features['label'].int2str(0))
    print(dataset_dict['train'][:3])

    print("结束处理数据")