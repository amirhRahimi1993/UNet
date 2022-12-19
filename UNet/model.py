import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np 

class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channel, out_channel, 3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace= True),
            )
    def forward(self,x):
        return self.conv(x)
class UNET(nn.Module):
    def __init__(self,in_channel=3,out_channel=1,features=[64,128,256,512]):
        super(UNET,self).__init__()
        self.ups= []
        self.down= []
        self.pool = nn.MaxPool2d(kernel_size=2,stride= 2)
        #Down
        for f in features:
            self.down.append(DoubleConv(in_channel,f))
            in_channel = f

        #up
        reverse_feautre = reversed(features)
        for f in reverse_feautre:
            self.ups.append(nn.ConvTranspose2d(
                f * 2,f,2,2))
            self.ups.append(DoubleConv(f*2,f))
        self.bottle_neck = DoubleConv(features[-1],features[-1]*2)
        self.final_layer = nn.Conv2d(features[0],out_channel,kernel_size=1)
    def forward(self,x):
        skip_connection = []
        for d in self.down:
            x= d(x)
            skip_connection.append(x)
            x= self.pool(x)
        x= self.bottle_neck(x)
        skip_connection = skip_connection[::-1]
        for idx in range(0,len(self.ups),2):
            x= self.ups[idx](x)
            sc= skip_connection[idx//2]
            if x.shape != sc.shape:
                x= TF.resize(x,sc.shape[2:])
            concat_skip = torch.cat((sc,x),dim=1)
            x= self.ups[idx+1](concat_skip)
        return self.final_layer(x)
def test():
    x= torch.randn((3,1,161,161))
    model= UNET(in_channel=1,out_channel=1)
    pred = model(x)
    assert pred.shape == x.shape
if __name__ == "__main__":
    test()