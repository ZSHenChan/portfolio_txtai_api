# check_torch.py - Script to check PyTorch installation details

import torch

def check_torch_installation():
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Display CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("This is a CPU-only PyTorch installation")
    
    # Print torch build info
    print("\nTorch build information:")
    for attr in dir(torch.version):
        if not attr.startswith('_'):
            print(f"  {attr}: {getattr(torch.version, attr)}")

check_torch_installation()