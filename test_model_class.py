import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        avg_out = self.fc(y_avg).view(b, c, 1, 1)
        max_out = self.fc(y_max).view(b, c, 1, 1)
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.cbam = CBAM(512)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = HybridModel()
try:
    state_dict = torch.load("best_cnn_resnet_cbam.pth", map_location="cpu")
    # Actually state_dict has both `resnet.*` and `features.*`. They are the same.
    # Load with strict=False
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", len(missing_keys))
    if len(missing_keys) > 0:
        print("Missing:", missing_keys)
    print("Unexpected keys:", len(unexpected_keys))
    if len(unexpected_keys) > 0:
        print("Unexpected:", unexpected_keys)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
