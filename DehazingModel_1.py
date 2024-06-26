import torch
import torch.nn as nn
import torch.nn.functional as F

class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        # Define the original layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=15, out_channels=3, kernel_size=3, stride=1, padding=1)  # New layer
        self.conv7 = nn.Conv2d(in_channels=18, out_channels=3, kernel_size=3, stride=1, padding=1)  # New layer
        self.b = 1

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = F.relu(self.conv5(cat3))  # New layer
        cat4 = torch.cat((x1, x2, x3, x4, x5), 1)  # New concatenation
        x6 = F.relu(self.conv6(cat4))  # New layer
        cat5 = torch.cat((x1, x2, x3, x4, x5, x6), 1)  # New concatenation
        k = F.relu(self.conv7(cat5))  # New layer

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)
