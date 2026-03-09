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
from core.models import inverse_model, forward_model
from torch.utils import data
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import wget
import hashlib
from sklearn.metrics import r2_score
from scipy.io import savemat
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
    
    """128*64"""
    number = 13 #3 9 11 13
    SNR = 5 #-5 -2 0 2 5
    data_dic = np.load('data/marmousi_small_data_SNR={}.npy'.format(SNR), allow_pickle=True).item()
    elastic_impedance_data = data_dic["impedance"][number,:,:,:] #impedance or impedance_low
    
    print("elastic_impedance_data",elastic_impedance_data.shape)

    """input data"""
    seismic_data = data_dic["synth_seismic"][number,:,:,:] #synth_seismic or synth_seismic_nosie

    """output seismic data of diffusion model"""
    # test_number = 3
    # diffusion_result = np.load('diffusion_result_SNR={}.npy'.format(SNR), allow_pickle=True).item() 
    # seismic_data = diffusion_result["in_samples"][test_number,:,:,:] #in_samples or out_samples
    # seismic_data = seismic_data[:,:,::1]
    # print("seismic_data",seismic_data.shape)

    seismic_mean = torch.tensor(np.mean(seismic_data,axis=(0,-1),keepdims=True)).float()
    seismic_std = torch.tensor(np.std(seismic_data,axis=(0,-1),keepdims=True)).float()

    elastic_mean= torch.tensor(np.mean(elastic_impedance_data, keepdims=True)).float()
    elastic_std = torch.tensor(np.std(elastic_impedance_data,keepdims=True)).float()


    seismic_data = torch.tensor(seismic_data).float()
    elastic_impedance_data = torch.tensor(elastic_impedance_data).float()
    

    if torch.cuda.is_available():
        seismic_data = seismic_data.cuda()
        elastic_impedance_data = elastic_impedance_data.cuda()
        seismic_mean = seismic_mean.cuda()
        seismic_std = seismic_std.cuda()
        elastic_mean = elastic_mean.cuda()
        elastic_std = elastic_std.cuda()
        

    seismic_normalization = Normalizaforward_modeltion(mean_val=seismic_mean,
                                          std_val=seismic_std)

    elastic_normalization = Normalizaforward_modeltion(mean_val=elastic_mean,
                                          std_val=elastic_std)


    seismic_data = seismic_normalization.normalize(seismic_data)
    elastic_impedance_data = elastic_normalization.normalize(elastic_impedance_data)


    if not test:
        num_samples = seismic_data.shape[0]
        indecies = np.arange(0,num_samples)
        
        train_indecies = indecies[(np.linspace(0,len(indecies)-1,args.num_train_wells)).astype(int)]
        #train_indecies = indecies[400:400+args.num_train_wells-1]

        train_data = data.Subset(data.TensorDataset(seismic_data,elastic_impedance_data), train_indecies)
        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

        unlabeled_loader = data.DataLoader(data.TensorDataset(seismic_data), batch_size=args.batch_size, shuffle=True)
        
        
        Valid_indecies = np.linspace(0, len(indecies)-1, 31, dtype=int) #10 or 31
        Valid_data = data.Subset(data.TensorDataset(seismic_data,elastic_impedance_data), Valid_indecies)
        
        seam_val_loader = data.DataLoader(Valid_data, batch_size=args.batch_size)
        
        return train_loader, unlabeled_loader, seam_val_loader, seismic_normalization, elastic_normalization
   
    else:
        test_loader = data.DataLoader(data.TensorDataset(seismic_data,elastic_impedance_data), batch_size=args.batch_size, shuffle=False, drop_last=False)
        return test_loader, seismic_normalization, elastic_normalization

def get_models(args):

    if args.test_checkpoint is None:
        inverse_net = inverse_model(in_channels=len(args.incident_angles), nonlinearity=args.nonlinearity)
    else:
        try:
            inverse_net = torch.load(args.test_checkpoint)
        except FileNotFoundError:
            print("No checkpoint found at '{}'- Please specify the model for testing".format(args.test_checkpoint))
            exit()

    # wavelet, wavelet_time = wavelets.ormsby(args.wavelet_duration, args.dt,args.f, return_t=True)
    # wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()
    
    f=[40]
    wavelet, wavelet_time = wavelets.ricker(0.2, 1e-3, f) 
    wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float() 
    
    
    forward_net = forward_model(wavelet=wavelet)

    if torch.cuda.is_available():#GPU运行
        inverse_net.cuda()
        forward_net.cuda()

    return inverse_net, forward_net

def train(args):

    #writer = SummaryWriter()
    train_loader, unlabeled_loader, seam_val_loader, seismic_normalization, elastic_normalization = get_data(args)
    inverse_net, forward_net = get_models(args)
    inverse_net.train()
    
    criterion = nn.MSELoss()
    optimizer = inverse_net.optimizer

    #make a direcroty to save models if it doesn't exist
    if not isdir("checkpoints"):
        os.mkdir("checkpoints")

    print("Training the model")
    best_loss = np.inf
    trainLosses = []
    valLosses = []
    train_property_corr = []
    train_property_r2 = []
    # for epoch in tqdm(range(args.max_epoch)):
    for epoch in range(args.max_epoch):
        inverse_net.train() 
        for x,y in train_loader:
            
            optimizer.zero_grad()
            
            # print("x",x.shape)
            y_pred = inverse_net(x)
            # print("y_pred",y_pred.shape)
            property_loss = criterion(y_pred,y)
            corr, r2 = metric(y_pred.detach(),y.detach())
            train_property_corr.append(corr)
            train_property_r2.append(r2)
            

            if args.beta!=0:
                #loading unlabeled data
                try:
                    x_u = next(unlabeled)[0]
                except:
                    unlabeled = iter(unlabeled_loader)
                    x_u = next(unlabeled)[0]

                y_u_pred = inverse_net(x_u)
                y_u_pred = elastic_normalization.unnormalize(y_u_pred)              
                x_u_rec = forward_net(y_u_pred)
                x_u_rec = seismic_normalization.normalize(x_u_rec)
                
                seismic_loss = criterion(x_u_rec,x_u)
            else:
                seismic_loss = torch.tensor(0)

            loss = args.alpha*property_loss + args.beta*seismic_loss
            # loss = args.alpha*property_loss
            loss.backward()
            optimizer.step()
            
            
        for x, y in seam_val_loader:
            inverse_net.eval()
            y_pred = inverse_net(x)
            val_loss = criterion(y_pred, y)
            
  
        print('Epoch: {} | Train Loss: {:0.4f} | val_loss: {:0.4f} '.format(epoch, loss.item(), val_loss.item() ))
        # train_loss.append(loss.detach().clone())
        trainLosses.append((Variable(loss).data).cpu().numpy())
        valLosses.append((Variable(val_loss).data).cpu().numpy())
        
        num = 3
        SNR = -5
        np.savez('lossdata.npz'.format(num), trainLosses=trainLosses, valLosses=valLosses)
   
    torch.save(inverse_net,"./checkpoints/{}".format(args.session_name))
def test(args):
    #make a direcroty to save precited sections
    if not isdir("output_images"):
        os.mkdir("output_images")

    test_loader, seismic_normalization, elastic_normalization = get_data(args, test=True)
    if args.test_checkpoint is None:
        args.test_checkpoint = "./checkpoints/{}".format(args.session_name)
    inverse_net, forward_net = get_models(args)
    criterion = nn.MSELoss(reduction="sum")
    
    # predicted_impedance = []
    # true_impedance = []
    
    predicted_impedance = []
    true_impedance = []
    test_property_corr = []
    test_property_r2 = []
    inverse_net.eval()


        
    with torch.no_grad():
        test_loss = []
        for x,y in test_loader:
            y_pred = inverse_net(x)
            property_loss = criterion(y_pred,y)/np.prod(y.shape)
            corr, r2 = metric(y_pred.detach(),y.detach())
            test_property_corr.append(corr)
            test_property_r2.append(r2)

            x_rec = forward_net(elastic_normalization.unnormalize(y_pred))
            x_rec = seismic_normalization.normalize(x_rec)
            seismic_loss = criterion(x_rec, x)/np.prod(x.shape)
            loss = args.alpha*property_loss + args.beta*seismic_loss
            test_loss.append(loss.item())

            true_impedance.append(y)
            predicted_impedance.append(y_pred)

        corr = torch.mean(torch.cat(test_property_corr), dim=0).squeeze()
        r2 = torch.mean(torch.cat(test_property_r2), dim=0).squeeze()
        print("corr",corr)
        print("r2",r2)
    
        # display_results(test_loss, test_property_corr, test_property_r2, args, header="Test")

        predicted_impedance = torch.cat(predicted_impedance, dim=0)
        true_impedance = torch.cat(true_impedance, dim=0)

        predicted_impedance = elastic_normalization.unnormalize(predicted_impedance)
        true_impedance = elastic_normalization.unnormalize(true_impedance)

        if torch.cuda.is_available():
            predicted_impedance = predicted_impedance.cpu()
            true_impedance = true_impedance.cpu()

        predicted_impedance = predicted_impedance.numpy()
        true_impedance = true_impedance.numpy()
        
        def calculate_ssim(data1, data2):
            data1_normalized = ((data1 - np.min(data1)) / (np.max(data1) - np.min(data1))) * 255
            data2_normalized = ((data2 - np.min(data2)) / (np.max(data2) - np.min(data2))) * 255
            image1 = data1_normalized.astype(np.uint8)
            image2 = data2_normalized.astype(np.uint8)
            ssim_value = metrics.structural_similarity(image1, image2)
            return ssim_value
        
        ssim = calculate_ssim(true_impedance[:, 0, :], predicted_impedance[:, 0, :])
        print("ssim:", ssim)

        num = 3
        SNR = -5
        data = {'predicted_impedance': predicted_impedance,
                      'true_impedance': true_impedance,
                      'corr': corr,
                      'r2': r2,
                      'ssim': ssim} #_su


        """ final figure """
        # Plotting the training losses
        data = np.load('lossdata.npz')
        print(data.files)
        train_loss = data['trainLosses']
        val_losses = data['valLosses']        
        fig, ax = plt.subplots() 
        ax.plot(train_loss,'-k', label='Training Loss')
        ax.plot(val_losses,'-r', label='valLosses Loss')
        ax.legend(loc='upper right') 
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")    
        plt.ylim([0,2])


        vmin = 2000
        vmax = 10000
        fig, axes = plt.subplots(nrows=len(args.incident_angles), ncols=3, figsize = (12,3))
        for i, theta in enumerate(args.incident_angles):
            axes[0].imshow(predicted_impedance[:,i].T, cmap='jet',aspect="auto", vmin=vmin, vmax=vmax)
            axes[0].axis('off')
            axes[1].imshow(true_impedance[:,i].T, cmap='jet',aspect="auto",vmin=vmin, vmax=vmax)
            axes[1].axis('off')
            axes[2].imshow(abs(true_impedance[:,i].T-predicted_impedance[:,i].T), cmap='gray',aspect="auto")
            axes[2].axis('off')
        fig.tight_layout()
        

if __name__ == '__main__':
    ## Arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_train_wells', type=int, default=12, help="12 Number of EI traces from the model to be used for validation")
    parser.add_argument('-max_epoch', type=int, default=500 , help="50 300 maximum number of training 500 epochs")
    parser.add_argument('-batch_size', type=int, default=40,help="Batch size for training")
    parser.add_argument('-alpha', type=float, default=1, help="weight of property loss term")
    parser.add_argument('-beta', type=float, default=1, help="weight of seismic loss term")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None,help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'),help="name of the session to be ised in saving the model")
    parser.add_argument('-nonlinearity', action="store", type=str, default="rsp",help="Type of nonlinearity for the CNN [tanh, relu, swish, gelu, rnn, rsp, nsp]", choices=["tanh","relu"])

    ## Do not change these values unless you use the code on a different data and edit the code accordingly 
    parser.add_argument('-dt', type=float, default=1e-3, help='Time resolution in seconds')
    parser.add_argument('-wavelet_duration',  type=float, default=0.2, help='wavelet duration in seconds')
    parser.add_argument('-f', default="5, 10, 60, 80", help="Frequency of wavelet. if multiple frequencies use , to seperate them with no spaces, e.g., -f \"5,10,60,80\"", type=lambda x: np.squeeze(np.array(x.split(",")).astype(float)))
    parser.add_argument('-resolution_ratio', type=int, default=4, action="store",help="resolution mismtach between seismic and EI")
    parser.add_argument('-incident_angles', type=float, default=np.arange(0, 0+ 1, 10), help="np.arange(0, 30+ 1, 10) Incident angles of the input seismic and EI")
    args = parser.parse_args()

    if args.test_checkpoint is not None:
        test(args)
    else:
        train(args)
        test(args)
        
        
