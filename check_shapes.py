import torch
try:
    data = torch.load("best_cnn_resnet_cbam.pth", map_location="cpu")
    print("Type:", type(data))
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: not a tensor")
except Exception as e:
    print(f"Error: {e}")
