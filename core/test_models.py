import torch
from torch.nn.functional import conv1d
from torch import nn, optim

class inverse_model(nn.Module):
    def __init__(self, in_channels,resolution_ratio=6,nonlinearity="tanh"):
        super(inverse_model, self).__init__()
        self.in_channels = in_channels
        self.resolution_ratio = resolution_ratio #vertical scale mismtach between seismic and EI
        self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()
        """ Local pattern analysis"""
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                           out_channels=12,
                                           kernel_size=5,
                                            # padding=2,
                                            padding="same",
                                           dilation=1),
                                  nn.GroupNorm(num_groups=self.in_channels,
                                               num_channels=12))

        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                           out_channels=12,
                                           kernel_size=5,
                                           # padding=6,
                                            padding="same",
                                           dilation=3),
                                  nn.GroupNorm(num_groups=self.in_channels,
                                               num_channels=12))

        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                           out_channels=12,
                                           kernel_size=5,
                                           # padding=12,
                                            padding="same",
                                           dilation=6),
                                  nn.GroupNorm(num_groups=self.in_channels,
                                               num_channels=12))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=36,
                                           out_channels=24,
                                           kernel_size=3,
                                           # padding=1),
                                            padding="same"),
                                 nn.GroupNorm(num_groups=self.in_channels,
                                              num_channels=24),
                                 self.activation,

                                 nn.Conv1d(in_channels=24,
                                           out_channels=24,
                                           kernel_size=3,
                                           # padding=1),
                                            padding="same"),
                                 nn.GroupNorm(num_groups=self.in_channels,
                                              num_channels=24),
                                 self.activation,

                                 nn.Conv1d(in_channels=24,
                                           out_channels=24,
                                           kernel_size=1),
                                 nn.GroupNorm(num_groups=self.in_channels,
                                              num_channels=24),
                                 self.activation)
        
        """Sequewnce modeling"""
        self.gru = nn.GRU(input_size=self.in_channels,
                          hidden_size=12,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)
        """Upscaling"""
        self.up = nn.Sequential(nn.ConvTranspose1d(in_channels=24,
                                                   out_channels=12,
                                                   stride=3,
                                                   kernel_size=5,
                                                   padding=1),
                                nn.GroupNorm(num_groups=self.in_channels,
                                             num_channels=12),
                                self.activation,

                                nn.ConvTranspose1d(in_channels=12,
                                                   out_channels=12,
                                                   stride=2,
                                                   kernel_size=4,
                                                   padding=1),
                                nn.GroupNorm(num_groups=self.in_channels,
                                             num_channels=12),
                                self.activation)
        """Regression"""
        self.gru_out = nn.GRU(input_size=12,
                              hidden_size=12,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(in_features=24, out_features=self.in_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data) 
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


        self.optimizer = optim.Adam(self.parameters(), 0.005, weight_decay=1e-4)

    def forward(self, x):
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1,cnn_out2,cnn_out3),dim=1))

        tmp_x = x.transpose(-1, -2)
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)

        x = rnn_out + cnn_out
        x = self.up(x)

        tmp_x = x.transpose(-1, -2)
        x, _ = self.gru_out(tmp_x)

        x = self.out(x)
        x = x.transpose(-1,-2)
        return x


class forward_model(nn.Module):
    def __init__(self, wavelet, resolution_ratio=6):
        super(forward_model, self).__init__()
        self.wavelet = wavelet.float() if torch.is_tensor(wavelet) else torch.tensor(wavelet).float()
        self.resolution_ratio = resolution_ratio
    def cuda(self):
        self.wavelet = self.wavelet.cuda()



    def forward(self, x):
        x_d = x[..., 1:] - x[..., :-1]
        x_a = (x[..., 1:] + x[..., :-1]) / 2

        rc = x_d / x_a
        for i in range(rc.shape[1]):
            tmp_synth = conv1d(rc[:, [i]], self.wavelet, padding=int(self.wavelet.shape[-1] / 2))

            if i == 0:
                synth = tmp_synth
            else:
                synth = torch.cat((synth, tmp_synth), dim=1)

        synth = synth[...,::self.resolution_ratio]

        return synth
