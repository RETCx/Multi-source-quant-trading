import torch

print("=====================================")
print("System Hardware Check for PyTorch")
print("=====================================")

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    print(f"GPU Found: {device_count} device(s)")
    print(f"GPU Model: {device_name}")
else:
    print("\nWarning: PyTorch is running on CPU.")
    print("If you have an NVIDIA GPU, please check your CUDA Toolkit and PyTorch installation.")

print("=====================================")
