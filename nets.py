import torch
import torch.nn as nn

def vgg_block_single(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
    
def vgg_block_double(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )


class MyVGG11(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        self.conv_block1 =vgg_block_single(in_ch,64)
        self.conv_block2 =vgg_block_single(64,128)

        self.conv_block3 =vgg_block_double(128,256)
        self.conv_block4 =vgg_block_double(256,512)
        self.conv_block5 =vgg_block_double(512,512)

        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        
        x=self.conv_block1(x)
        x=self.conv_block2(x)

        x=self.conv_block3(x)
        x=self.conv_block4(x)
        x=self.conv_block5(x)

        x=x.view(x.size(0), -1)

        x=self.fc_layers(x)

        return x
