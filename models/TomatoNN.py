from torch import nn
import torch


class TomatoNN(nn.Module):
    def __init__(self):
        super(TomatoNN, self).__init__()
        self.conv1 = self.conv_block(n=1, nIn=3, nOut=128)
        self.pool3 = nn.MaxPool2d(kernel_size=(4,4))

        self.conv4 = self.conv_block(n=1, nIn=128, nOut=512)
        self.pool6 = nn.MaxPool2d(kernel_size=(3, 3))

        self.conv7 = self.conv_block(n=1, nIn=512, nOut=2048)
        self.pool9 = nn.MaxPool2d(kernel_size=(3, 3))

        # self.conv10 = self.conv_block(n=1, nIn=1024, nOut=1024)
        # self.conv11 = self.conv_block(n=1, nIn=1024, nOut=1024)

        self.flatten = nn.Flatten(1)

        self.dense1 = self.dense_block(2, nIn=2048*6*6, nOut=1024, drop_out=True)
        self.dense2 = self.dense_block(2, nIn=1024, nOut=128, drop_out= True)
        self.dense4 = self.dense_block(2, nIn=128, nOut=1, drop_out=False)

    def conv_block(self, n, nIn, nOut, batch_norm=True, pad=0):
        block = nn.Sequential()
        block.add_module(f"down_conv_{n}",nn.Conv2d(nIn,nOut,kernel_size=(3,3), stride=(1,1), padding=pad))
        if(batch_norm):
            block.add_module(f"down_bn_{n}", nn.BatchNorm2d(nOut))
        block.add_module(f"down_act_{n}", nn.ReLU())
        return block

    def dense_block(self, n, nIn, nOut, batch_norm=True,drop_out=True):
        block = nn.Sequential()
        block.add_module(f"dense_{n}", nn.Linear(nIn, nOut))
        if(batch_norm):
            block.add_module(f"down_bn_{n}", nn.BatchNorm1d(nOut))
        block.add_module(f"down_act_{n}", nn.ReLU())
        if(drop_out):
            block.add_module(f"drop_out_{n}", nn.Dropout(0.2))
        return block

    def forward(self, X):
        X = self.conv1(X)
        # X = self.conv2(X)
        X = self.pool3(X)

        X = self.conv4(X)
        # X = self.conv5(X)
        X = self.pool6(X)

        X = self.conv7(X)
        # X = self.conv8(X)
        X = self.pool9(X)

        X = self.flatten(X)

        X = self.dense1(X)
        X = self.dense2(X)
        X = self.dense4(X)

        return X
