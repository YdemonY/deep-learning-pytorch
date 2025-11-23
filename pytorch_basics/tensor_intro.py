import torch

x = torch.randn(3, 4)
w = torch.randn(4, 2, requires_grad=True)

y = x @ w  # 矩阵乘法
loss = y.sum()

loss.backward()

print("x:", x)
print("w.grad:", w.grad)
