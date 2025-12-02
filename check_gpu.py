import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(0))
    
    # Check memory
    print("GPU memory allocated:", torch.cuda.memory_allocated(0) / 1024**3, "GB")
    print("GPU memory cached:", torch.cuda.memory_reserved(0) / 1024**3, "GB")
else:
    print("No CUDA GPU available")