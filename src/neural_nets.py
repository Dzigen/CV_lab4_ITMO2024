### SOURCE: https://www.kaggle.com/code/vyacheslavshen/dcgan-pytorch-tutorial

import torch.nn as nn
import torch

#
class UnconditionalGenerator(nn.Module):
    '''
    Generator class.
    Forward function takes a noise tensor of size Bx100
    and returns Bx3x64x64
    '''
    def __init__(self, noise_size=100):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=noise_size, out_channels=1024, stride=1, kernel_size=4, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            # 4x4
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            # 8x8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # 16x16
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # 32x32
            nn.ConvTranspose2d(in_channels=128, out_channels=3, stride=2, kernel_size=4, padding=1, bias=False),
            nn.Tanh()
        ])
        
    
    def forward(self, x, y=None):
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        return self.model(x)

#
class UnconditionalDiscriminator(nn.Module):
    '''
    Discriminator class.
    Forward function takes a tensor of size BxCx64x64
    and return Bx1
    '''
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=128, stride=2, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, stride=2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=1024, stride=2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=1024, out_channels=1, stride=4, kernel_size=5, padding=1, bias=False),
            nn.Sigmoid()
        ])
    
    def forward(self, x, y=None):
        x = self.model(x)
        return x.reshape((x.shape[0], 1))



######################################

#
class ConditionalGenerator(nn.Module):
    '''
    Generator class.
    Forward function takes a noise tensor of size Bx100
    and returns Bx3x64x64
    '''
    def __init__(self, noise_size=100):
        super().__init__()
        
        self.embedding = nn.Embedding(2,50)
        self.linear = nn.Linear(50, noise_size // 5)

        self.model = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=noise_size + self.linear.out_features, out_channels=1024, stride=1, kernel_size=4, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            # 4x4
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            # 8x8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # 16x16
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # 32x32
            nn.ConvTranspose2d(in_channels=128, out_channels=3, stride=2, kernel_size=4, padding=1, bias=False),
            nn.Tanh()
        ])
        
    
    def forward(self, x, y):
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        #print(x.shape)

        y_emb = self.embedding(y)
        #print(y_emb.shape)
        y_dense = self.linear(y_emb)
        #print(y_dense.shape)
        y_reshape = y_dense.reshape((y_dense.shape[0], -1, 1, 1))
        #print(y_reshape.shape)

        xy_cat = torch.cat([x, y_reshape], 1)
        #print(xy_cat.shape)

        out = self.model(xy_cat)

        return out
    

#
class ConditionalDiscriminator(nn.Module):
    '''
    Discriminator class.
    Forward function takes a tensor of size Bx(C+1)x64x64
    and return Bx1
    '''
    def __init__(self, in_channels=3+1):
        super().__init__()

        
        self.embedding = nn.Embedding(2,50)
        self.linear = nn.Linear(50, 4096)

        self.model = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=128, stride=2, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, stride=2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=1024, stride=2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=1024, out_channels=1, stride=4, kernel_size=5, padding=1, bias=False),
            nn.Sigmoid()
        ])
    
    def forward(self, x, y):
        y_emb = self.embedding(y)
        #print(y_emb.shape)
        y_dense = self.linear(y_emb)
        #print(y_dense.shape)
        y_reshape = y_dense.reshape((y.shape[0], 1, 64, 64))
        #print(y_reshape.shape)

        xy_cat = torch.cat([x, y_reshape], 1)
        #print(xy_cat.shape)

        out = self.model(xy_cat)

        return out.reshape((out.shape[0], 1))

