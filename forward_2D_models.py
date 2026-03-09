import torch
from torch.nn.functional import conv1d
from torch import nn
from bruges.filters import wavelets


# #nn模块提供了定义神经网络层和模型的类;optim模块提供了训练和优化神经网络的优化器类
# class forward_model(nn.Module):
#     def __init__(self, wavelet, resolution_ratio=6):
#         super(forward_model, self).__init__()
#         self.wavelet = wavelet.float() if torch.is_tensor(wavelet) else torch.tensor(wavelet).float()
#         self.resolution_ratio = resolution_ratio
#     def cuda(self):
#         self.wavelet = self.wavelet.cuda()#它将wavelet成员变量移到了GPU上运行



#     def forward(self, x):
#         x_d = x[..., 1:] - x[..., :-1]#每一个时刻上相邻两个时刻的差值
#         x_a = (x[..., 1:] + x[..., :-1]) / 2

#         rc = x_d / x_a
#         for i in range(rc.shape[1]):#每个通道进行卷积
#             tmp_synth = conv1d(rc[:, [i]], self.wavelet, padding=int(self.wavelet.shape[-1] / 2))

#             if i == 0:
#                 synth = tmp_synth
#             else:
#                 synth = torch.cat((synth, tmp_synth), dim=1)

#         synth = synth[...,::self.resolution_ratio]#下采样

#         return synth




# # #ricker子波
# f=[40]
# wavelet, wavelet_time = wavelets.ricker(0.2, 1e-3, f) #生成主频为 40Hz的雷克子波
# wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()  #转换为 tensor 类型并调整维度
# # fig = plt.figure(figsize=(14, 6))
# # plt.plot(wavelet[0,0,:])

# tmp_synth = np.zeros_like(rc)
# for m in range(Elastic_impedance.shape[1]):
#     rc1 = torch.tensor(rc[:,m,:]).unsqueeze(dim=1).float()
#     tmp_synth1 = conv1d(rc1, wavelet, padding=int(wavelet.shape[-1] / 2))
#     tmp_synth [:,m,:]=tmp_synth1.squeeze().float()
# tmp_synth = tmp_synth.astype(np.float32)



class forward_model(nn.Module):
    def __init__(self, wavelet, resolution_ratio=6):
        super(forward_model, self).__init__()
        self.wavelet = wavelet.float() if torch.is_tensor(wavelet) else torch.tensor(wavelet).float()
        self.resolution_ratio = resolution_ratio
    def cuda(self):
        self.wavelet = self.wavelet.cuda()#它将wavelet成员变量移到了GPU上运行

    def forward(self, x):
        """合成弹性阻抗"""  
        # Vp = x[:,0,:].cpu().detach().numpy()
        # Vs = x[:,1,:].cpu().detach().numpy()
        # Den = x[:,2,:].cpu().detach().numpy()
        x = torch.squeeze(x,dim=3)
        print('x',x.shape)
        x = x.permute(0,2,1) #（10,1,128）
                    
        """计算反射系数"""
        rc = torch.zeros((x.shape[0], x.shape[1], x.shape[2]))       
        rc1 = torch.zeros((x.shape[0], x.shape[1], x.shape[2]-1))
        for m in range(x.shape[1]):
            x_d = x[:,m,1:] - x[:,m,:-1]#每一个时刻上相邻两个时刻的差值
            x_a = (x[:,m,1:] + x[:,m,:-1]) / 2
            rc1[:,m,:] = x_d / x_a 

        rc[:,:,1:] = rc1

        """合成地震数据"""
        f=[40]
        wavelet, wavelet_time = wavelets.ricker(0.2, 1e-3, f) #生成主频为 40Hz的雷克子波
        wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()  #转换为 tensor 类型并调整维度
    
        tmp_synth = torch.zeros_like(rc)
    
        for m in range(x.shape[1]):
            rc1 = torch.tensor(rc[:,m,:]).unsqueeze(dim=1).float()
            tmp_synth1 = conv1d(rc1, wavelet, padding=int(wavelet.shape[-1] / 2))
            tmp_synth [:,m,:]=tmp_synth1.squeeze().float()

          
        tmp_synth = torch.tensor(tmp_synth).to('cuda')
        # tmp_synth = tmp_synth[...,::self.resolution_ratio]#下采样
 
        return tmp_synth
