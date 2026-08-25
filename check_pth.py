import torch
try:
    data = torch.load("best_cnn_resnet_cbam.pth", map_location="cpu")
    print(type(data))
    if isinstance(data, dict):
        print(data.keys())
    else:
        print(data)
except Exception as e:
    print(f"Error: {e}")
