import torch
from torch.nn.functional import conv1d
from torch import nn, optim
import torch.nn.functional as F
#nn模块提供了定义神经网络层和模型的类;optim模块提供了训练和优化神经网络的优化器类

class RNNActivate(nn.Module):
    def forward(self, x):
        return F.relu(x) - F.relu(-x)
    
class RandSoftplus(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        epsilon = torch.rand_like(x)
        return torch.log(1 + torch.exp(self.beta * (x + epsilon)))

class NoisySoftplus(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        noise = torch.randn(x.shape) * 0.1
        noise = noise.to(x.device)
        return torch.log(1 + torch.exp(x + noise))
    
    
class inverse_model(nn.Module):
    def __init__(self, in_channels,resolution_ratio=1,nonlinearity="rsp"): #tanh relu swish rnn randsoftplus
        super(inverse_model, self).__init__()#调用了当前类的父类（或祖先类）inverse_model的构造函数，即将子类 (self) 的对象传递到父类中进行初始化
        self.in_channels = in_channels
        self.resolution_ratio = resolution_ratio #vertical scale mismatch between seismic and EI
    
        # self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()#这里使用了Python中的一个三元表达式，即a if condition else b，等价于if condition: a else: b.
      
        
        """激活函数"""
        if nonlinearity == "relu":
            self.activation = nn.ReLU()  # relu
        if nonlinearity == "rnn":
            self.activation = RNNActivate()  # rnn
        if nonlinearity == "rsp":
            self.activation = RandSoftplus()  # randsoftplus
        if nonlinearity == "nsp":
            self.activation = NoisySoftplus()  # NoisySoftplus
        if nonlinearity == "tanh":
            self.activation = nn.Tanh()  # Tanh
        if nonlinearity == "swish":
            self.activation = nn.SiLU()  # 使用Swish激活函数
        elif nonlinearity == "mish":
            self.activation = Mish()  # 使用Mish激活函数
        elif nonlinearity == "gelu":
            self.activation = F.gelu  # 使用GELU激活函数
         
                
        
        
        """ Local pattern analysis"""
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                            out_channels=8,
                                            kernel_size=5,
                                            # padding=2,
                                            padding="same",
                                            dilation=1),
                                   nn.GroupNorm(num_groups=self.in_channels,
                                                num_channels=8))
        #因此，这段代码指的是一个 1D 卷积神经网络模型中的第一层：先做一维卷积操作，得到 8 个特征图，再对这 8 个特征图进行归一化。其中输入通道数 in_channels 在模型构建时被定义，kernel_size 和 padding 的设置可以根据具体问题进行相应地调整。

        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                           out_channels=8,
                                           kernel_size=5,
                                           # padding=6,
                                            padding="same",
                                           dilation=3),
                                  nn.GroupNorm(num_groups=self.in_channels,
                                               num_channels=8))

        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                           out_channels=8,
                                           kernel_size=5,
                                           # padding=12,
                                            padding="same",
                                           dilation=6),
                                  nn.GroupNorm(num_groups=self.in_channels,
                                               num_channels=8))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=24,
                                           out_channels=16,
                                           kernel_size=3,
                                           # padding=1),
                                            padding="same"),
                                 nn.GroupNorm(num_groups=self.in_channels,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=3,
                                           # padding=1),
                                            padding="same"),
                                 nn.GroupNorm(num_groups=self.in_channels,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=1),
                                 nn.GroupNorm(num_groups=self.in_channels,
                                              num_channels=16),
                                 self.activation)
        
        
        """Sequewnce modeling"""
        self.gru = nn.GRU(input_size=self.in_channels,
                          hidden_size=8,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)
        
        
        """Upscaling""" #size_output = (size_input - 1)stride + k - 2padding + outpadding
        self.up = nn.Sequential(nn.ConvTranspose1d(in_channels=16,
                                                   out_channels=8,
                                                   stride=2, #2 or 3
                                                   kernel_size=4,#4 or 5
                                                    padding=1),
                                                    # padding="same"),
                                nn.GroupNorm(num_groups=self.in_channels,
                                             num_channels=8),
                                self.activation,

                                nn.ConvTranspose1d(in_channels=8,
                                                   out_channels=8,
                                                   stride=2,
                                                   kernel_size=4,
                                                    padding=1),
                                                   # padding="same"),
                                nn.GroupNorm(num_groups=self.in_channels,
                                             num_channels=8),
                                self.activation)
        
        """Regression"""
        self.gru_out = nn.GRU(input_size=16, #8 16
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)#也就是分别从前向和后向处理输入序列。这里设置为 True，因此总的输出维度将是 16。
        self.out = nn.Linear(in_features=16, out_features=self.in_channels)

        for m in self.modules():    #用于遍历self对象中包含的所有模块。
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):#判断当前模块
                nn.init.xavier_uniform_(m.weight.data)#xavier初始化函数，权重偏置进行初始化
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):#判断当前模块
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):#判断当前模块
                m.bias.data.zero_()


        self.optimizer = optim.Adam(self.parameters(), 0.005, weight_decay=1e-4)#Adam优化算法：
                                                                                #self.parameters()：这是一个方法，用来获取当前模型的所有可学习参数
                                                                                #0.005：Adam学习率，每个参数的变化量会乘以这个学习率。
                                                                                #weight_decay：L2正则化方法-权重衰减，表示在每次迭代中，每个参数的变化量会减去一个相应的值，这个值是当前参数值乘以权重衰减系数。
    def forward(self, x):           #前向传播
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1,cnn_out2,cnn_out3),dim=1))#拼接
        
        tmp_x = x.transpose(-1, -2)#将输入x沿着最后两个维度进行转置：这样做是因为循环神经网络(RNN)和卷积神经网络(CNN)所处理的数据的形状在时间和空间维度上通常是不同的，对于RNN，时间维度通常是最后一个维度，而对于CNN，空间维度通常是最后两个维度
        rnn_out, _ = self.gru(tmp_x)#将其输入到一个GRU循环神经网络层gru中进行处理，得到输出rnn_out
        rnn_out = rnn_out.transpose(-1, -2)#并将其沿着最后两个维度进行转置
        # print("rnn_out",rnn_out.shape)

        x = rnn_out + cnn_out
        # x = self.up(x)#然后通过上采样层up进行升维
        tmp_x = x.transpose(-1, -2)#将输出x沿着最后两个维度进行转置
        x, _ = self.gru_out(tmp_x)#_隐藏状态张量

        x = self.out(x)#将新的输出x传入一个全连接层out中进行处理
        x = x.transpose(-1,-2)#并将其沿着最后两个维度进行转置
        return x


class forward_model(nn.Module):
    def __init__(self, wavelet, resolution_ratio=1):
        super(forward_model, self).__init__()
        self.wavelet = wavelet.float() if torch.is_tensor(wavelet) else torch.tensor(wavelet).float()
        self.resolution_ratio = resolution_ratio
    def cuda(self):
        self.wavelet = self.wavelet.cuda()#它将wavelet成员变量移到了GPU上运行



    def forward(self, x):
        x_d = x[..., 1:] - x[..., :-1]#每一个时刻上相邻两个时刻的差值
        x_a = (x[..., 1:] + x[..., :-1]) / 2

        rc = x_d / x_a
        for i in range(rc.shape[1]):#每个通道进行卷积
            tmp_synth = conv1d(rc[:, [i]], self.wavelet, padding=int(self.wavelet.shape[-1] / 2))

            if i == 0:
                synth = tmp_synth
            else:
                synth = torch.cat((synth, tmp_synth), dim=1)

        synth = synth[...,::self.resolution_ratio]#下采样
        synth = synth[...,:]
        zero_row = torch.zeros(synth.shape[0], synth.shape[1], 1,  device=synth.device)
        synth = torch.cat((synth, zero_row), dim=2)
        return synth
