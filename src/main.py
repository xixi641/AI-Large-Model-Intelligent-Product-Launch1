from preprocess.dataset import get_dataloader, DatasetType
from preprocess.process import process_data
from runner.predict import run_predict
from runner.train import train, get_device

# 测试dataset.py
# if __name__ == '__main__':
#     train_dataloader = get_dataloader(data_type=DatasetType.TRAIN)
#     test_dataloader = get_dataloader(data_type=DatasetType.TEST)
#     print(len(train_dataloader))
#     print(len(test_dataloader))
#
#
#     for batch in train_dataloader:
#         print(batch.keys())
#         break


# 测试train.py
# get_device()
# train()

run_predict() 