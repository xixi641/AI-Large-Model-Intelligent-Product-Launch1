import time
import torch
import torch_directml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataloader

def get_device():
    """获取训练设备并显示信息"""
    try:
        device = torch_directml.device()
        print(f"✓ DirectML 设备已初始化")
        print(f"✓ 设备类型: {device}")
        print(f"✓ 正在使用 AMD GPU (780M 核显)")
        return device
    except Exception as e:
        print(f"⚠ DirectML 初始化失败: {e}")
        print("⚠ 回退到 CPU 模式")
        return torch.device('cpu')


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    total_loss = 0
    model.train()
    for batch in tqdm(dataloader, desc='训练'):
        inpu_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)

        # 前向传播
        outputs = model(inpu_ids, attention_mask)
        #outputs.shape = [batch_size, num_labels]

        # 计算损失
        loss = loss_function(outputs, label)

        # 反向传播
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train():
    # 选择设备
    device = torch_directml.device()
    # 加载数据
    dataloader = get_dataloader()
    # 加载模型
    model = ProductClassifier(freeze_bert=False)
    model = model.to(device)
    #准备损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    # 准备优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 准备日志
    writer = SummaryWriter(log_dir = config.LOG_DIR/time.strftime('%Y-%m-%d_%H-%M-%S'))
    # 确保模型保存目录存在
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)


    bert_loss = float('inf')
    for epoch in  range(1, config.EPOCHS+1):
        print(f"=========== Epoch{epoch} ==========")
        avg_loss = train_one_epoch(model, dataloader, loss_function, optimizer, device)

        print(f"LOSS: {avg_loss:.4f}")

        writer.add_scalar('Loss', avg_loss, epoch)

        if avg_loss < bert_loss:
            bert_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
            print("保存模型")

    writer.close() 
