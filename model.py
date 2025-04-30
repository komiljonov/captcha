import torch
import torch.nn as nn


# class CaptchaModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CaptchaModel, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#         )
#         self.lstm = nn.LSTM(
#             448, 128, num_layers=2, bidirectional=True, batch_first=True
#         )
#         self.fc = nn.Linear(128 * 2, num_classes)  # Because BiLSTM doubles hidden size

#     def forward(self, x):
#         x = self.conv(x)  # (B, 64, 7, 25)
#         b, c, h, w = x.size()
#         x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
#         x = x.view(b, w, c * h)  # (B, W, 448)
#         x, _ = self.lstm(x)
#         x = self.fc(x)
#         return x


class CaptchaModel(nn.Module):
    def __init__(self, num_classes):
        super(CaptchaModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # -> (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (B, 32, H/2, W/2)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (B, 64, H/4, W/4)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (B, 128, H/8, W/8)
        )

        # Image size: (1, 30, 100) => After 3 pools => (128, 3, 12)
        # So: h=3, w=12, c=128
        self.lstm = nn.LSTM(
            input_size=128 * 3,  # 128 channels * height
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.conv(x)  # (B, C, H, W)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.reshape(b, w, c * h)  # (B, W, C*H)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
