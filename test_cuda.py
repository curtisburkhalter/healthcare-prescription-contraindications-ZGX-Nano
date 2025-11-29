# save as test_cuda.py
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
    
    # Try allocating memory
    try:
        x = torch.randn(1000, 1000).cuda()
        print(f"✓ Successfully allocated tensor on GPU")
        print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ Error: {e}")
