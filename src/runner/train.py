import time
import torch
import torch_directml
from pandas.core.computation import check
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataloader, DatasetType


def get_device():
    """获取训练设备并显示信息"""
    try:
        device = torch_directml.device()
        print(f"✓ DirectML 设备已初始化")
        print(f"✓ 设备类型: {device}")
        print(f"✓ 正在使用 AMD GPU ")
        return device
    except Exception as e:
        print(f"⚠ DirectML 初始化失败: {e}")
        print("⚠ 回退到 CPU 模式")
        return torch.device('cpu')

class EarlyStopping:
    def __init__(self,patience=2):
        self.best_loss = float('inf')
        self.counter = 0 # 连续超过best_loss的次数
        self.patience = patience


    def should_stop(self,avg_loss,model,path):
        if avg_loss < self.best_loss:
            self.counter = 0
            self.best_loss = avg_loss
            torch.save(model.state_dict(), path)
            print(f"已保存最优模型到 {path}")
            return  False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"已连续超过{self.patience}次，停止训练")
                return True
            else:
                return False



def run_one_epoch(model, dataloader, loss_function, device,optimizer=None,  is_train =True):
    total_loss = 0
    if is_train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_train): # 在训练时计算梯度/验证时不支持计算梯度
        for batch in tqdm(dataloader, desc=('训练' if is_train else '验证')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask)
            #outputs.shape = [batch_size, num_labels]

            # 计算损失
            loss = loss_function(outputs, label)


            if is_train:
                loss.backward ()# 反向传播
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
        return total_loss / len(dataloader)

def train():
    # 选择设备
    device = torch_directml.device()
    # 加载数据
    train_dataloader = get_dataloader(DatasetType.TRAIN)# 仅为测试
    valid_dataloader = get_dataloader(DatasetType.VALID)
    # 加载模型
    model = ProductClassifier(freeze_bert=True).to(device)
    model = model.to(device)
    #准备损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    # 准备优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 准备日志
    writer = SummaryWriter(log_dir = config.LOG_DIR/time.strftime('%Y-%m-%d_%H-%M-%S'))
    # 确保模型保存目录存在
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # 早停策略
    early_stopping = EarlyStopping()
    # 检查是否存在checkpoint
    # checkpoint路径
    checkpoint_path = config.MODELS_DIR / 'checkpoint.pt'

    start_epoc = 1
    if checkpoint_path.exists():
        print(f"已找到checkpoint，从{checkpoint_path}加载模型")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        early_stopping.best_loss = checkpoint['best_loss']
        early_stopping.counter = checkpoint['counter']
        start_epoc = checkpoint['epoch'] + 1
    else:
        print("未找到checkpoint，从新开始训练")

    for epoch in  range(start_epoc, config.EPOCHS+1):
        print(f"=========== Epoch{epoch} ==========")
        # 训练一轮
        train_avg_loss = run_one_epoch(model, train_dataloader, loss_function, device, optimizer)
        # 验证一轮
        valid_avg_loss = run_one_epoch(model, valid_dataloader, loss_function, device,is_train=False)
        print(f"训练LOSS: {train_avg_loss:.4f}")
        print(f"验证LOSS: {valid_avg_loss:.4f}")

        writer.add_scalar('Loss/Train', train_avg_loss, epoch)
        writer.add_scalar('Loss/Valid', valid_avg_loss, epoch)


        if early_stopping.should_stop(train_avg_loss,model, config.MODELS_DIR / 'best.pt'):
            break
        # 保存训练状态（check point）
        checkpoint= {"model": model.state_dict(),
                     "optimizer":optimizer.state_dict(),
                     "epoch":epoch,
                     "best_loss":early_stopping.best_loss,
                     "counter":early_stopping.counter,
                     }
        torch.save(checkpoint, config.MODELS_DIR / 'checkpoint.pt')

    writer.close() 
