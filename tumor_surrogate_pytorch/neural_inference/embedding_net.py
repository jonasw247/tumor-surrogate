from warnings import warn
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, device):
        super(ConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,2,2)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=50, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.AdaptiveAvgPool3d((1,1,1))
        )

        #self.net.to(device=device)

    def forward(self, x):
        if x.shape[0] > 50:
            warn("Embedding Net batchsize is greater 50.")
        x = x.view(-1,1,128,128,128)
        return self.net(x)[:,:,0,0,0]