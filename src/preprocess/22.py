from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
from configuration import config


def process_data():
    print("开始处理数据")
    # 读取文件
    dataset_dict = load_dataset('csv', data_files={
        'train': str(config.RAW_DATA_DIR/'train.txt'),
        'test': str(config.RAW_DATA_DIR/'test.txt'),
        'valid': str(config.RAW_DATA_DIR/'valid.txt')

    }, delimiter='\t')   

    # 过滤数据
    dataset_dict = dataset_dict.filter(lambda x: x['label'] is not None and x['text_a'] is not None)
    # 构建数据集
    tokenizer = AutoTokenizer.from_pretrained(str(config.PRE_TRAINED_DIR / 'bert-base-chinese'))
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

    # 处理label列
    all_labels = dataset_dict['train'].unique('label')
    dataset_dict = dataset_dict.cast_column('label' , ClassLabel(names=all_labels))


    print(dataset_dict['train'].features)
    print(dataset_dict['train'][:3])



    print("结束处理数据")
