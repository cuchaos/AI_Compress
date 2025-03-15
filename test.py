# system_info.py
import torch
import platform
import psutil
import pkg_resources
from transformers import __version__ as transformers_version

def print_system_info():
    print("=== System Information ===")
    
    # 作業系統資訊
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    
    # CPU 資訊
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores (Physical): {psutil.cpu_count(logical=False)}")
    print(f"CPU Cores (Total): {psutil.cpu_count(logical=True)}")
    print(f"CPU Frequency: {psutil.cpu_freq().current:.2f} MHz (Max: {psutil.cpu_freq().max:.2f} MHz)")
    
    # 記憶體資訊
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / 1024**3:.2f} GB")
    print(f"Available RAM: {memory.available / 1024**3:.2f} GB")
    
    # GPU 資訊
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA Available: False")
    
    # 軟體環境
    print("\n=== Software Environment ===")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Transformers Version: {transformers_version}")
    
    # 其他關鍵套件版本
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    for pkg in ['bitsandbytes', 'xformers', 'peft', 'torchvision', 'torchaudio', 'nltk']:
        version = installed_packages.get(pkg, "Not installed")
        print(f"{pkg.capitalize()}: {version}")

if __name__ == "__main__":
    print_system_info()