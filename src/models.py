import torch
import torch.nn as nn
import torch.nn.functional as F

##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)

# The code for self-attention reference from https://github.com/heykeetae/Self-Attention-GAN
class SelfAttn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_channels):
        super(SelfAttn,self).__init__()
        self.in_channels = in_channels
    
        self.query_conv = nn.Conv2d(in_channels , in_channels//8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
        self.next = nn.Conv2d(in_channels, in_channels*2, 1)
        self.avgpool = nn.AvgPool2d(3, 2, 1)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Height*Width)
        """
        m_batchsize,C,height,width  = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,height*width).permute(0,2,1) # B X N X C,   N: H*W
        proj_key =  self.key_conv(x).view(m_batchsize,-1,height*width) # B X C x N
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,height*width) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,height,width)
        
        out = self.gamma*out + x
        
        out_next = self.next(out)
        out_next = self.avgpool(out_next)
        
        return out, out_next

class Generator(nn.Module):
    def __init__(self, PGBFP, n_blocks):
        super(Generator, self).__init__()
        
        self.PGBFP = PGBFP
        
        # Initial convolution block
        out_features = 64
        self.pre = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, out_features, 7),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.downlayer1 = nn.Sequential(
            nn.Conv2d(out_features, out_features*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        out_features *= 2
        if self.PGBFP:
            self.self_attn1 = SelfAttn(out_features)
        
        self.downlayer2 = nn.Sequential(
            nn.Conv2d(out_features, out_features*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        out_features *= 2
        if self.PGBFP:
            self.self_attn2 = SelfAttn(out_features)
            
        # Residual blocks
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks += [ResidualBlock(out_features)]
        self.res = nn.Sequential(*res_blocks)

        self.uplayer1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_features, out_features//2, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features//2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        out_features //= 2
        self.uplayer2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_features, out_features//2, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features//2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        out_features //= 2
        self.post = nn.Sequential(
            nn.ReflectionPad2d(3), 
            nn.Conv2d(out_features, 3, 7), 
            nn.InstanceNorm2d(3, affine=True),
            nn.Tanh(),
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.initialize_weights()
        
    def forward(self, x):
        if not self.PGBFP:
            pre = self.pre(x)

            dl1 = self.downlayer1(pre)
            dl2 = self.downlayer2(dl1)

            res = self.res(dl2)

            ul1 = self.uplayer1(res)
            ul2 = self.uplayer2(ul1)

            residual = self.post(ul2)
        else:
            pre = self.pre(x)

            dl1 = self.downlayer1(pre)
            
            dl2 = self.downlayer2(dl1)
            sa1, sa_next1 = self.self_attn1(dl1)
            
            res = self.res(dl2)
            sa2, _ = self.self_attn2(dl2 + sa_next1)
            
            ul1 = self.uplayer1(res + sa2)
            ul2 = self.uplayer2(ul1 + sa1)
            
            residual = self.post(ul2)
            
        weighted_residual = self.gamma * residual
        out = torch.tanh(x - weighted_residual)
        
        return out, weighted_residual

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
    
    def get_factor(self):
        return self.gamma
    
##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, n_scales, scale_type):
        super(Discriminator, self).__init__()
        
        self.n_scales = n_scales
        self.scale_type = scale_type
        
        self.avgpool = nn.AvgPool2d(2)
        
        self.model_list = nn.ModuleList()
        for _ in range(n_scales):
            self.model_list.append(self.make_net())
        
        self.initialize_weights()
        
    def forward(self, img):
        outs = []
        
        for i in range(len(self.model_list)):
            outs.append(self.model_list[i](img).squeeze(1))
            if i < self.n_scales-1:
                if self.scale_type == 'crop':
                    img = img[:,:,int(img.shape[-2]/4):int(3*img.shape[-2]/4),int(img.shape[-1]/4):int(3*img.shape[-1]/4)]
                elif self.scale_type == 'resize':
                    img = self.avgpool(img)
                else:
                    raise Exception('no such scale_type')
        return outs
    
    def forward_i(self, i, img):
        return self.model_list[i](img).squeeze(1)
    
    def make_net(self):
        out_channels = 64
        
        model = []
        model += [nn.Conv2d(3, out_channels, 4, stride=2, padding=1)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        
        for _ in range(1, 4):
            model += [nn.Conv2d(out_channels, out_channels*2, 4, stride=2, padding=1)]
            model += [nn.InstanceNorm2d(out_channels*2)]
            model += [nn.LeakyReLU(0.2, inplace=True)]
            out_channels *= 2
        
        model += [nn.ZeroPad2d((1, 0, 1, 0))]
        model += [nn.Conv2d(out_channels, 1, 4, padding=1)]
        
        return nn.Sequential(*model)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()