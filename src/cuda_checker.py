import torch

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"BF16 (bfloat16) support: {torch.cuda.is_bf16_supported()}")
print(f"TF32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 (cuDNN): {torch.backends.cudnn.allow_tf32}")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")

a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
torch.matmul(a, b)