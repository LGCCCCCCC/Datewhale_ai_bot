import torch
print("是否可用：", torch.cuda.is_available()) # 查看GPU是否可用，需要为True
print("torch方法查看CUDA版本：", torch.version.cuda) # torch方法查看CUDA版本，会输出当前torch所对应的cuda版本