import torch
import torch_directml

try:
    device = torch_directml.device()
    print(f"✓ DirectML 设备: {device}")
    print(f"✓ 可以使用 AMD RX 6700 显卡进行训练")

    # 简单测试
    x = torch.randn(2, 3).to(device)
    y = x * 2
    print(f"✓ GPU 计算测试成功")
    print(f"  测试数据: {y}")
except Exception as e:
    print(f"✗ DirectML 错误: {e}")
    print("  请确保已安装: pip install torch-directml")

print("=" * 50)
