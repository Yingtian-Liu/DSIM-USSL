from core.functions import *
from core.metric import *
import argparse
import numpy as np
import torch
from torch import nn, optim
from bruges.filters import wavelets
from torch.autograd import Variable
from os.path import isdir
import os
from core.models import inverse_model, FFF, forward_model
from torch.utils import data
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import wget
import hashlib
from core.hilbert import instantaneous_attributes
from itertools import chain
from skimage import metrics

#Manual seeds for reproducibility
random_seed=30
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_data(args, test=False):
    
    """128*64参数"""
    number = 13 #子剖面个数3 9 11 13
    SNR = 5 #-5 -2 0 2 5
    data_dic = np.load('marmousi_small_data_SNR={}.npy'.format(SNR), allow_pickle=True).item()
    elastic_impedance_data = data_dic["impedance"][number,:,:,:] #impedance or impedance_low
    
    print("elastic_impedance_data",elastic_impedance_data.shape)

    """输入地震数据"""
    seismic_data = data_dic["synth_seismic"][number,:,:,:] #synth_seismic or synth_seismic_nosie

    """扩散模型输出地震数据"""
    # test_number = 3
    # diffusion_result = np.load('diffusion_result_SNR={}.npy'.format(SNR), allow_pickle=True).item() 
    # seismic_data = diffusion_result["in_samples"][test_number,:,:,:] #in_samples or out_samples
    # seismic_data = seismic_data[:,:,::1]
    # print("seismic_data",seismic_data.shape)

        


    # seismic_data = seismic_data[:,:,0:2070]
    # elastic_impedance_data = elastic_impedance_data[:,:,0:2070]
    
    print("seismic",seismic_data.shape)
    print("model",elastic_impedance_data.shape)
    
    seismic_data = seismic_data[:,:,::6]
    elastic_impedance_data = elastic_impedance_data[:,:,::]
    
    print('args.incident_angles',args.incident_angles)
    
    print("seismic",seismic_data.shape)
    print("model",elastic_impedance_data.shape)


    assert seismic_data.shape[1]==len(args.incident_angles) ,'Data dimensions are not consistent with incident angles. Got {} incident angles and {} in data dimensions'.format(len(args.incident_angles),seismic_data.shape[1])
    assert seismic_data.shape[1]==elastic_impedance_data.shape[1] ,'Data dimensions are not consistent. Got {} channels for seismic data and {} for elastic elastic impedance dimensions'.format(seismic_data.shape[1],elastic_impedance_data.shape[1])

    seismic_mean = torch.tensor(np.mean(seismic_data,axis=(0,-1),keepdims=True)).float()
    seismic_std = torch.tensor(np.std(seismic_data,axis=(0,-1),keepdims=True)).float()

    elastic_mean= torch.tensor(np.mean(elastic_impedance_data, keepdims=True)).float()
    elastic_std = torch.tensor(np.std(elastic_impedance_data,keepdims=True)).float()


    seismic_data = torch.tensor(seismic_data).float()
    elastic_impedance_data = torch.tensor(elastic_impedance_data).float()
    
    #调用GPU
    if torch.cuda.is_available():
        seismic_data = seismic_data.cuda()
        elastic_impedance_data = elastic_impedance_data.cuda()
        seismic_mean = seismic_mean.cuda()
        seismic_std = seismic_std.cuda()
        elastic_mean = elastic_mean.cuda()
        elastic_std = elastic_std.cuda()
        
    # 归一化   
    seismic_normalization = Normalizaforward_modeltion(mean_val=seismic_mean,
                                          std_val=seismic_std)

    elastic_normalization = Normalizaforward_modeltion(mean_val=elastic_mean,
                                          std_val=elastic_std)


    seismic_data = seismic_normalization.normalize(seismic_data)
    elastic_impedance_data = elastic_normalization.normalize(elastic_impedance_data)


    """设置训练数据集"""
    if not test:
        num_samples = seismic_data.shape[0]#道集参数
        indecies = np.arange(0,num_samples)#数组
        
        train_indecies = indecies[(np.linspace(0,len(indecies)-1,args.num_train_wells)).astype(int)]#训练道集索引
        #train_indecies = indecies[400:400+args.num_train_wells-1]#训练集道集索引

        train_data = data.Subset(data.TensorDataset(seismic_data,elastic_impedance_data), train_indecies)#训练集标签
        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

        unlabeled_loader = data.DataLoader(data.TensorDataset(seismic_data), batch_size=args.batch_size, shuffle=True)
        
        Valid_indecies = np.linspace(0, len(indecies)-1, args.num_val_wells, dtype=int) #验证集道集 #10 or 31
        # Valid_indecies = np.array([280,300,320])
        Valid_indecies = np.setdiff1d(Valid_indecies, train_indecies)
        
        val_data = data.Subset(data.TensorDataset(seismic_data,elastic_impedance_data), Valid_indecies)
        val_loader = data.DataLoader(val_data, batch_size=args.batch_size)
        
        return train_loader, unlabeled_loader, val_loader, seismic_normalization, elastic_normalization
    else:
        test_loader = data.DataLoader(data.TensorDataset(seismic_data,elastic_impedance_data), batch_size=args.batch_size, shuffle=False, drop_last=False)
        return test_loader, seismic_normalization, elastic_normalization

def get_models(args):

    if args.test_checkpoint is None:
        print("len(args.incident_angles)",len(args.incident_angles))
        inverse_net = inverse_model(in_channels=len(args.incident_angles), nonlinearity=args.nonlinearity)
        FFF_net = FFF(in_channels=len(args.incident_angles), n_classes=1, nonlinearity=args.nonlinearity)
    else:
        try:
            inverse_net = torch.load(args.test_checkpoint)
            FFF_net = torch.load(args.test_checkpoint)
        except FileNotFoundError:
            print("No checkpoint found at '{}'- Please specify the model for testing".format(args.test_checkpoint))
            exit()

    f=[20]
    wavelet, wavelet_time = wavelets.ricker(0.2, 1e-3, f) #生成主频为 40Hz的雷克子波
    wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()  #转换为 tensor 类型并调整维度
    
    forward_net = forward_model(wavelet=wavelet)
    

    if torch.cuda.is_available():#GPU运行
        inverse_net.cuda()
        FFF_net.cuda()
        forward_net.cuda()
        
    return inverse_net, FFF_net, forward_net

def train(args):

    #writer = SummaryWriter()
    train_loader, unlabeled_loader, val_loader, seismic_normalization, elastic_normalization = get_data(args)
    inverse_net, FFF_net, forward_net = get_models(args)
    
    """将预训练模型参数加载到新模型中"""
    if args.load_model_low!=0:
        inverse_net = torch.load("./invert_checkpoints/model_low/marmousi2") 
    """--------------------------"""
    
    
    # inverse_net.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=chain(FFF_net.parameters(),inverse_net.parameters()), lr = 0.005, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.98)
    print("Training the model")
    best_loss = np.inf
    trainLosses_property = []
    trainLosses_seismic = []
    trainLosses_seismic1 = []
    trainLosses_frequency = []
    trainLosses_phase = []
    
    train_property_corr = []
    train_property_r2 = []
    # for epoch in tqdm(range(args.max_epoch)):
    for epoch in range(args.max_epoch):
        for x,y in train_loader:
            inverse_net.train()
            FFF_net.train()
            optimizer.zero_grad()
            
            y_pred = inverse_net(x)
            x_rec = FFF_net(y)
            
            property_loss = criterion(y_pred,y)
            seismic_loss = criterion(x_rec,x)

            corr, r2 = metric(y_pred.detach(),y.detach())
            train_property_corr.append(corr)
            train_property_r2.append(r2)
            
            if args.unsupervised!=0:
                try:
                    x_u = next(unlabeled)[0]
                except:
                    unlabeled = iter(unlabeled_loader)
                    x_u = next(unlabeled)[0]
                    
                y_u_pred = inverse_net(x_u)
                x_u_rec = FFF_net(y_u_pred) 
                seismic_loss1 = 0.5*criterion(x_u_rec,x_u)  #2e-2 0.5
                
                """FFT transform"""
                if args.gamma2!=0:
                    x_u_rec_freq = torch.fft.fft(x_u_rec, dim=-1)
                    x_u_freq = torch.fft.fft(x_u, dim=-1) #1
                    frequency_loss = 1*1e-5*criterion(abs(x_u_rec_freq),abs(x_u_freq)) #1e-5
                else:  
                    frequency_loss = torch.tensor(0)
                    
                """Hilbert phase"""
                if args.gamma3!=0:
                    x_u_rec_phase = instantaneous_attributes(x_u_rec)
                    x_u_phase = instantaneous_attributes(x_u)
                    phase_loss = 2*1e-2*criterion(abs(x_u_rec_phase),abs(x_u_phase)) #2e-2

                else:  
                    phase_loss = torch.tensor(0)  
            
 
    
            """SL"""
            # loss = property_loss + seismic_loss
            # loss = property_loss
            """T-SSL"""
            # loss = property_loss + seismic_loss + seismic_loss1
            """F-SSL"""
            # loss = property_loss + seismic_loss + frequency_loss
            """P-SSL"""
            # loss = property_loss + seismic_loss + phase_loss
            
            
            """TFP-SSL"""
            loss = property_loss + seismic_loss + seismic_loss1 + frequency_loss + phase_loss
            # loss = property_loss + seismic_loss + seismic_loss1 
            
            
            property_loss.backward(retain_graph=True)
            seismic_loss.backward(retain_graph=True)
            seismic_loss1.backward(retain_graph=True)
            frequency_loss.backward(retain_graph=True)
            phase_loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            
        # for x, y in val_loader: #bug:验证集要影响训练
        #     FFF_net.eval()
        #     inverse_net.eval()
        #     x_rec = FFF_net(y)
        #     y_pred = inverse_net(x)
        #     val_property = criterion(y_pred,y)
        #     val_seismic = criterion(x_rec,x)

        # print('Epoch: {} | Train property: {:0.4f} | val property: {:0.4f} |'.format(epoch, property_loss, val_property))
        print('Epoch: {} | property_loss: {:0.4f} | seismic_loss: {:0.4f} | seismic_loss1: {:0.4f} | frequency_loss: {:0.4f} | phase_loss: {:0.4f}| '
              .format(epoch, property_loss, seismic_loss, seismic_loss1, frequency_loss, phase_loss))
        
        trainLosses_property.append((Variable(property_loss).data).cpu().numpy())
        trainLosses_seismic.append((Variable(seismic_loss).data).cpu().numpy())
        trainLosses_seismic1.append((Variable(seismic_loss1).data).cpu().numpy())
        trainLosses_frequency.append((Variable(frequency_loss).data).cpu().numpy())
        trainLosses_phase.append((Variable(phase_loss).data).cpu().numpy())
        
    """保存损失"""
    # np.savez('data/TFP_SSL/lossdata.npz', trainLosses_property=trainLosses_property,
    #           trainLosses_seismic=trainLosses_seismic,
    #           trainLosses_seismic1=trainLosses_seismic1,
    #           trainLosses_frequency=trainLosses_frequency,
    #           trainLosses_phase=trainLosses_phase)
    
    torch.save(inverse_net,"./invert_checkpoints/{}".format(args.session_name))

    """低频模型"""
    if args.save_model_low!=0:
        torch.save(inverse_net,"./invert_checkpoints/model_low/marmousi2")
    torch.save(FFF_net,"./forward_checkpoints/{}".format(args.session_name))
    
def test(args):
    #make a direcroty to save precited sections
    if not isdir("output_images"):
        os.mkdir("output_images")

    test_loader, seismic_normalization, elastic_normalization = get_data(args, test=True)
    # inverse_net, FFF_net, forward_net = get_models(args)
    
    FFF_net = torch.load("./forward_checkpoints/{}".format(args.session_name))
    inverse_net = torch.load("./invert_checkpoints/{}".format(args.session_name))
    
    criterion = nn.MSELoss(reduction="sum")
    predicted_impedance = []
    true_impedance = []
    predicted_seismic = []
    true_seismic = []
    
    test_property_corr = []
    test_property_r2 = []
    inverse_net.eval()
    print("\nTesting the model\n")

    with torch.no_grad():
        test_loss = []
        for x,y in test_loader:
            y_pred = inverse_net(x)
            x_pred = FFF_net(y)
            property_loss = criterion(y_pred,y)/np.prod(y.shape)
            corr, r2 = metric(y_pred.detach(),y.detach())
            test_property_corr.append(corr)
            test_property_r2.append(r2)

            true_impedance.append(y)
            predicted_impedance.append(y_pred)
            true_seismic.append(x)
            predicted_seismic.append(x_pred)
            
            
        corr = torch.mean(torch.cat(test_property_corr), dim=0).squeeze()
        r2 = torch.mean(torch.cat(test_property_r2), dim=0).squeeze()
        print("PCC",corr)
        print("r2",r2)   
        # display_results(test_loss, test_property_corr, test_property_r2, args, header="Test")
        predicted_impedance = torch.cat(predicted_impedance, dim=0)
        true_impedance = torch.cat(true_impedance, dim=0)
        predicted_seismic = torch.cat(predicted_seismic, dim=0)
        true_seismic = torch.cat(true_seismic, dim=0)

        predicted_impedance = elastic_normalization.unnormalize(predicted_impedance)
        true_impedance = elastic_normalization.unnormalize(true_impedance)
        
        predicted_seismic = seismic_normalization.unnormalize(predicted_seismic)
        true_seismic = seismic_normalization.unnormalize(true_seismic)


        if torch.cuda.is_available():
            predicted_impedance = predicted_impedance.cpu()
            true_impedance = true_impedance.cpu()
            predicted_seismic = predicted_seismic.cpu()
            true_seismic = true_seismic.cpu()
            

        predicted_impedance = predicted_impedance.numpy()
        true_impedance = true_impedance.numpy()
        predicted_seismic = predicted_seismic.numpy()
        true_seismic = true_seismic.numpy()
        
        def calculate_ssim(data1, data2):
            # 数据归一化到合适的范围（0到255之间）
            data1_normalized = ((data1 - np.min(data1)) / (np.max(data1) - np.min(data1))) * 255
            data2_normalized = ((data2 - np.min(data2)) / (np.max(data2) - np.min(data2))) * 255
            # 将归一化后的数据转换为图像格式
            image1 = data1_normalized.astype(np.uint8)
            image2 = data2_normalized.astype(np.uint8)
            # 计算结构相似性
            ssim_value = metrics.structural_similarity(image1, image2)
            return np.array([ssim_value])
        
        # 分别计算三个通道的结构相似性
        ssim_tensors = []  # 初始化一个空列表用于存储 ssim 张量
        value = calculate_ssim(true_impedance[:, 0, :], predicted_impedance[:, 0, :])
        value_tensor = torch.from_numpy(value)  # 将 NumPy 数组转换为 PyTorch 张量
        ssim_tensors.append(value_tensor)  # 将每个 value_tensor 添加到列表中

        # 使用 torch.cat 将列表中的张量沿着指定的维度拼接
        ssim = torch.cat(ssim_tensors, dim=0) 
        print("ssim {}".format(ssim))
            
        
        def calculate_snr(X_true, X_fit):
            # Calculate Signal Power
            signal_power = np.sum(X_true**2)  # or torch.norm(X_true)**2 for Frobenius norm squared
            
            # Calculate Noise Power
            error_tensor = X_true - X_fit
            noise_power = np.sum(error_tensor**2)  # or torch.norm(error_tensor)**2 for Frobenius norm squared
            # Calculate SNR in dB
            snr = 10 * np.log10(signal_power / noise_power)
            return np.array([snr])  # Convert to Python float
        
        snr_tensors = []  # 初始化一个空列表用于存储 snr 张量
        value = calculate_snr(true_impedance[:, 0, :], predicted_impedance[:, 0, :])
        value_tensor = torch.from_numpy(value)  # 将 NumPy 数组转换为 PyTorch 张量
        snr_tensors.append(value_tensor)  # 将每个 value_tensor 添加到列表中
        # 使用 torch.cat 将列表中的张量沿着指定的维度拼接
        snr = torch.cat(snr_tensors, dim=0) 
        print("snr {}".format(snr))
        
        
        def calculate_mse(X_true, X_fit):
            # Calculate RMSE
            # mean = np.mean(X_true)

            # std = np.std(X_true)
            # X_true = (X_true-mean)/std
            # X_fit = (X_fit-mean)/std
            mse = np.mean((X_true - X_fit)**2)
            rmse = np.sqrt(mse)
            # Calculate range of true values to normalize
            nrmse = rmse / np.mean(X_true)
            return nrmse
        
        nrmse = calculate_mse(true_impedance[:, 0, :], predicted_impedance[:, 0, :])
        print('nrmse', f'{nrmse:.4f}')
        
        data = {'predicted_impedance': predicted_impedance,
                      'true_impedance': true_impedance,
                      'predicted_seismic': predicted_seismic,
                      'true_seismic': true_seismic,
                      'PCC': corr,
                      'r2': r2,
                      'ssim': ssim,
                      'snr': snr,
                      'nrmse': nrmse} #_su
        "无噪情况"
        # np.save('data/TFP_SSL/impedance.npy', data)
        "含噪情况"
        # np.save('data/SNR={}/TFP_SSL/impedance.npy'.format(args.SNR), data)    
        
        
        """ final figure """
        # Plotting the training losses
        data = np.load('data/TFP_SSL/lossdata.npz')
        trainLosses_property = data['trainLosses_property']
        trainLosses_seismic = data['trainLosses_seismic']
        trainLosses_seismic1 = data['trainLosses_seismic1']
        trainLosses_frequency = data['trainLosses_frequency']
        trainLosses_phase = data['trainLosses_phase']
        
        fig, ax = plt.subplots()
        ax.plot(trainLosses_property,'-k', label='train property')
        ax.plot(trainLosses_seismic, label='train seismic')
        ax.plot(trainLosses_seismic1,'-r', label='train seismic1')
        ax.plot(trainLosses_frequency,'-g', label='train_frequency ')
        ax.plot(trainLosses_phase,'-b', label='train_phase')
        # ax.plot(val_loss,'-b', label='val Loss')
        ax.legend(loc='upper right') #右上
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")  

        
        fig, axes = plt.subplots(nrows=len(args.incident_angles), ncols=3, figsize = (12,3))
        for i, theta in enumerate(args.incident_angles):
            axes[0].imshow(predicted_impedance[:,i].T, cmap='jet',aspect="auto", vmin=true_impedance.min(), vmax=true_impedance.max())
            axes[0].axis('off')
            axes[1].imshow(true_impedance[:,i].T, cmap='jet',aspect="auto",vmin=true_impedance.min(), vmax=true_impedance.max())
            axes[1].axis('off')
            axes[2].imshow(abs(true_impedance[:,i].T-predicted_impedance[:,i].T), cmap='gray',aspect="auto")
            axes[2].axis('off')
        fig.tight_layout()

        #diplaying estimated section
        predicted_seismic = 3*predicted_seismic
        true_seismic = 3*true_seismic
        fig, axes = plt.subplots(nrows=len(args.incident_angles), ncols=3, figsize = (12,3))
        print("predicted_seismic",predicted_impedance.shape) #(2721, 4, 2070)
        for i, theta in enumerate(args.incident_angles):
            axes[0].imshow(predicted_seismic[:,i].T, cmap='RdGy',aspect="auto")# vmin=-1, vmax=1
            axes[0].axis('off')
            axes[1].imshow(true_seismic[:,i].T, cmap='RdGy',aspect="auto") # vmin=-1, vmax=1
            axes[1].axis('off')
            axes[2].imshow(abs(true_seismic[:,i].T-predicted_seismic[:,i].T), cmap='gray',aspect="auto") #vmin=0, vmax=1
            axes[2].axis('off')
        fig.tight_layout()
        


if __name__ == '__main__':
    ## Arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-SNR', type=float, default=5, help="20 10 5")
    parser.add_argument('-num_train_wells', type=int, default=18, help="18 12 Number of EI traces from the model to be used for validation")
    parser.add_argument('-num_val_wells', type=int, default=30, help="Number of EI traces from the model to be used for validation")
    parser.add_argument('-max_epoch', type=int, default=500, help="maximum number of training 500 epochs")
    parser.add_argument('-batch_size', type=int, default=40,help="Batch size for training")
    
    parser.add_argument('-save_model_low', type=float, default=0, help="save low model, 0 is close, 1 is open")
    parser.add_argument('-load_model_low', type=float, default=0, help="load low model, 0 is close, 1 is open")
    parser.add_argument('-choice_model', type=float, default=1, help="0 is low model, 1 is model")

    parser.add_argument('-unsupervised', type=float, default=1, help="0 is close, 1 is open")
    parser.add_argument('-alpha', type=float, default=1, help="weight of property loss term")
    parser.add_argument('-gamma1', type=float, default=1, help="1 weight of seismic loss term")
    parser.add_argument('-gamma2', type=float, default=1, help="1*1e-2, weight of Frequency loss term")
    parser.add_argument('-gamma3', type=float, default=1, help="1*1e-1 weight of phase loss term")
    
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None,help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'),help="name of the session to be ised in saving the model")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh",help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh","relu"])

    ## Do not change these values unless you use the code on a different data and edit the code accordingly 
    parser.add_argument('-dt', type=float, default=1e-3, help='Time resolution in seconds')
    parser.add_argument('-wavelet_duration',  type=float, default=0.2, help='wavelet duration in seconds')
    parser.add_argument('-f', default="5, 10, 60, 80", help="Frequency of wavelet. if multiple frequencies use , to seperate them with no spaces, e.g., -f \"5,10,60,80\"", type=lambda x: np.squeeze(np.array(x.split(",")).astype(float)))
    parser.add_argument('-resolution_ratio', type=int, default=6, action="store",help="6 resolution mismtach between seismic and EI")
    parser.add_argument('-incident_angles', type=float, default=np.arange(0, 0+ 1, 10), help="Incident angles of the input seismic and EI")
    args = parser.parse_args()

    if args.test_checkpoint is not None:
        test(args)
    else:
        train(args)
        test(args)
