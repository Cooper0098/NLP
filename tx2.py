import torch

x = torch.arange(6).view(2, 3).float()  # 将张量x转换为浮点型

print("Shape:", x.shape)
print("Minimum:", torch.min(x))
print("Maximum:", torch.max(x))
print("Mean:", torch.mean(x))
print("Standard Deviation:", torch.std(x))
