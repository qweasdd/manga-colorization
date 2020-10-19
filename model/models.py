import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
import math
from torch import Tensor
from torch.nn import Parameter

'''https://github.com/orashi/AlacGAN/blob/master/models/standard.py'''

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Selayer(nn.Module):
    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out
    
class SelayerSpectr(nn.Module):
    def __init__(self, inplanes):
        super(SelayerSpectr, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = SpectralNorm(nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1))
        self.conv2 = SpectralNorm(nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))
            
        self.selayer = Selayer(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.selayer(bottleneck)
        
        x = self.shortcut.forward(x)
        return x + bottleneck
    
class SpectrResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(SpectrResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = SpectralNorm(nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv_conv = SpectralNorm(nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False))
        self.conv_expand = SpectralNorm(nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))
            
        self.selayer = SelayerSpectr(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.selayer(bottleneck)
        
        x = self.shortcut.forward(x)
        return x + bottleneck
    
class FeatureConv(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super(FeatureConv, self).__init__()

        no_bn = True
        
        seq = []
        seq.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        seq.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*seq)

    def forward(self, x):
        return self.network(x)
    
class Generator(nn.Module):
    def __init__(self, ngf=64):
        super(Generator, self).__init__()
        
        self.feature_conv = FeatureConv()

        self.to0 =  self._make_encoder_block_first(6, 32)
        self.to1 = self._make_encoder_block(32, 64)
        self.to2 = self._make_encoder_block(64, 128)
        self.to3 = self._make_encoder_block(128, 256)
        self.to4 = self._make_encoder_block(256, 512)
        
        self.deconv_for_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # output is 128 * 128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # output is 256 * 256
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
            nn.Tanh(),
        )

        tunnel4 = nn.Sequential(*[ResNeXtBottleneck(ngf * 8, ngf * 8, cardinality=32, dilate=1) for _ in range(20)])

        self.tunnel4 = nn.Sequential(nn.Conv2d(ngf * 8 + 512, ngf * 8, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel4,
                                     nn.Conv2d(ngf * 8, ngf * 4 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 64

        depth = 2
        tunnel = [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2),
                   ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1)]
        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel3,
                                     nn.Conv2d(ngf * 4, ngf * 2 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 128

        tunnel = [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2),
                   ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel2,
                                     nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        tunnel = [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=1)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=2)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=4)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=2),
                   ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)

        self.tunnel1 = nn.Sequential(nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel1,
                                     nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        self.exit = nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)

        
    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )    
        
    def forward(self, sketch, sketch_feat):

        x0 = self.to0(sketch)
        x1 = self.to1(x0)
        x2 = self.to2(x1)
        x3 = self.to3(x2)  
        x4 = self.to4(x3)

        sketch_feat = self.feature_conv(sketch_feat)
        
        out = self.tunnel4(torch.cat([x4, sketch_feat], 1))
        
        
        
        
        x = self.tunnel3(torch.cat([out, x3], 1))
        x = self.tunnel2(torch.cat([x, x2], 1))
        x = self.tunnel1(torch.cat([x, x1], 1))
        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))
        
        decoder_output = self.deconv_for_decoder(out)

        return x, decoder_output    
'''
class Colorizer(nn.Module):
    def __init__(self, extractor_path = 'model/model.pth'):
        super(Colorizer, self).__init__()
        
        self.generator = Generator()
        self.extractor = se_resnext_half(dump_path=extractor_path, num_classes=370, input_channels=1)
        
    def extractor_eval(self):
        for param in self.extractor.parameters():
            param.requires_grad = False
            
    def extractor_train(self):
        for param in extractor.parameters():
            param.requires_grad = True
            
    def forward(self, x, extractor_grad = False):
        
        if extractor_grad:
            features = self.extractor(x[:, 0:1])
        else:
            with torch.no_grad():
                features = self.extractor(x[:, 0:1]).detach()

        fake, guide = self.generator(x, features)

        return fake, guide
'''

class Colorizer(nn.Module):
    def __init__(self, generator_model, extractor_model):
        super(Colorizer, self).__init__()
        
        self.generator = generator_model
        self.extractor = extractor_model
        
    def load_generator_weights(self, gen_weights):
        self.generator.load_state_dict(gen_weights)
        
    def load_extractor_weights(self, ext_weights):
        self.extractor.load_state_dict(ext_weights)
        
    def extractor_eval(self):
        for param in self.extractor.parameters():
            param.requires_grad = False
        self.extractor.eval()
            
    def extractor_train(self):
        for param in extractor.parameters():
            param.requires_grad = True
        self.extractor.train()
            
    def forward(self, x, extractor_grad = False):
        
        if extractor_grad:
            features = self.extractor(x[:, 0:1])
        else:
            with torch.no_grad():
                features = self.extractor(x[:, 0:1]).detach()

        fake, guide = self.generator(x, features)

        return fake, guide

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()

        self.feed = nn.Sequential(SpectralNorm(nn.Conv2d(3, 64, 3, 1, 1)),
                                nn.LeakyReLU(0.2, True),
                                SpectralNorm(nn.Conv2d(64, 64, 3, 2, 0)),
                                nn.LeakyReLU(0.2, True),
            
            
            
            
                                  SpectrResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
                                  SpectrResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1, stride=2),  # 128
                                  SpectralNorm(nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False)),
                                  nn.LeakyReLU(0.2, True),

                                  SpectrResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1),
                                  SpectrResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),  # 64
                                  SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=False)),
                                  nn.LeakyReLU(0.2, True),

                                  SpectrResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
                                  SpectrResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=2),  # 32,
                                  SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=1, stride=1, padding=1, bias=False)),  
                                  nn.LeakyReLU(0.2, True),
                                  SpectrResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                  SpectrResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 16
                                  SpectrResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                  SpectrResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                  nn.AdaptiveAvgPool2d((1, 1))
                                  )

        self.out = nn.Linear(512, 1)

    def forward(self, color):
        x = self.feed(color)
        
        out = self.out(x.view(color.size(0), -1))
        return out
    
class Content(nn.Module):
    def __init__(self, path):
        super(Content, self).__init__()
        vgg16 = M.vgg16()
        vgg16.load_state_dict(torch.load(path))
        vgg16.features = nn.Sequential(
            *list(vgg16.features.children())[:9]
        )
        self.model = vgg16.features
        self.register_buffer('mean', torch.FloatTensor([0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        return self.model((images.mul(0.5) - self.mean) / self.std)
