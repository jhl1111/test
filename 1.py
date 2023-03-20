import torch
import scipy.io as sio
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils import data
import os
import matplotlib.pyplot as plt



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(       # 输入(b, 1, 100 , 3000)
            #nn.Conv1d参数(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            #n=（n-k+2*p）/s+1;
            nn.Conv1d(1, 20, 4, 4),  # (b, 16, 49, 1499)  
            nn.ReLU(True),
            #nn.MaxPool1d参数(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False
            #n=(n+2*p-k)/s+1
            nn.MaxPool1d(2, stride=1),  # (b, 16, 48, 1498)
 
            nn.Conv1d(20, 10, 5, 4),  # (b, 8, 24, 749)
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=1),  # (b, 8, 23, 748) """
            nn.Linear(186,186),
            #nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            #nn.ConvTranspose1d参数('in_channels','out_channels ','kernel_size','stride','padding','output_padding ','groups','bias','dilation')
            #n=(n-1)*s-2*p+k
            nn.ConvTranspose1d(10, 40, 8, stride=4),  # (b, 16, 48, 1498)
            nn.ReLU(True),
            nn.ConvTranspose1d(40, 20, 8, stride=4),  # (b, 8, 98, 2998)
            nn.ReLU(True),
            nn.ConvTranspose1d(20, 1, 5, stride=1),  # (b, 1, 100, 28)
            nn.Tanh()
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class Mydataset(data.Dataset):
    def __init__(self,root,batch_idx):
        path_dir = os.listdir(root)
        mat_contents = sio.loadmat(root+path_dir[batch_idx])
        raw_data1 = mat_contents.get('test_terminal')
        print("Train %d data loaded:)" % batch_idx)
        dimension = raw_data1.shape
        raw_data=np.ones((dimension[0]*dimension[1],dimension[2]))
        raw_data_norm = np.ones((dimension[0]*dimension[1],dimension[2]))
        for idx in range(dimension[0]):
            for idx1 in range(dimension[1]):
                raw_data[idx*100+idx1,:] = raw_data1[idx,idx1,:]
                self.upper_Bound = np.max(raw_data[ idx*100+idx1, :])
                self.lower_Bound = np.min(raw_data[ idx*100+idx1, :])
                raw_data_norm[idx*100+idx1, :] = self.normalization(raw_data[idx*100+idx1, :])
        self.raw_data = raw_data_norm
        self.sample_list = list()

    def __getitem__(self, index):
        inputdata = self.raw_data[index, :]
        inputdata_tensor = torch.from_numpy(inputdata)
        inputdata_tensor = inputdata_tensor.unsqueeze(0)
        inputdata_tensor = inputdata_tensor.float()
        return inputdata_tensor
    
    def __len__(self):
        return len(self.raw_data)
    
    def normalization(self, data):
        # data = (data - self.avg) / self.dev
        # data = data - self.avg
        _range = self.upper_Bound - self.lower_Bound
        if _range == 0:
            _range = 1e-8
        return (data) / _range
        

if __name__ == "__main__":
    predict_loss_all_all = []
    for u in range(1):
        batch_size = 10+10*u
        # 超参数设置
        # batch_size = 10
        lr = 1e-3
        weight_decay = 1e-5
        epoches = 1
        steps_print = 10
        root = "/home/Jianghelin/data/CPA_LSTM/DATA0902/train/"
        model = autoencoder()
        # x = Variable(torch.randn(1, 28*28))
        # encode, decode = model(x)
        # print(encode.shape)
        criterion = nn.MSELoss()
        optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizier, step_size = 4,gamma = 0.55,last_epoch = -1,verbose=True)
        train_path_dir = os.listdir(root)
        n_train_batches = len(train_path_dir)
        loss_records = []
        train_mean_loss = []
        epoch_plt = []    
        if torch.cuda.is_available():
            model.cuda(3)
        plt.ion()
        for epoch in range(epoches):
            print('\n=================EPOCH %d===================' % epoch)
            
            for batch_idx in range(n_train_batches-1):
                train_dataset = Mydataset(root,batch_idx)
                train_data = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True) 
                loss_records.append([])
                for step,data_in in enumerate(train_data):
                    # img = img.view(img.size(0), -1)
                    data_in = Variable(data_in.cuda(3))
                    # forward
                    _,output = model(data_in)
                    loss = criterion(output, data_in)
                    # backward
                    optimizier.zero_grad()
                    loss.backward()
                    optimizier.step()
                    loss_records[epoch].append(loss.cpu().data.numpy())
                    if step % steps_print == 0:
                        print('Epoch: ', epoch, ' | Step: ', step, '| train loss: %.4e' % loss.cpu().data.numpy())
                torch.set_printoptions(precision=16)
            scheduler.step()
            print('Epoch: ', epoch, '| Mean train loss: %.4e' % np.mean(loss_records[epoch]))

    #        print("epoch=", epoch, loss.data.float())
    #            for param_group in optimizier.param_groups:
    #                print(param_group['lr'])
            train_mean_loss.append(np.mean(loss_records[epoch]))
    #        if (epoch) % 5 == 0:
    #            print("epoch: {}, loss is {}".format((epoch), loss.data))
            epoch_plt.append(epoch)
            plt.cla()
            plt.title("Train loss")
            plt.plot(epoch_plt, train_mean_loss)
            plt.xlabel("epoch")
            plt.ylabel("Train loss")
            plt.pause(0.5)
        plt.ioff()
        plt.savefig("/home/Jianghelin/code/Autoencoder/batch_size_value/pic_save/epoch_loss/epoch_loss{}.png".format(u))
        plt.show()
        torch.save(model, '/home/Jianghelin/code/Autoencoder/batch_size_value/autoencoder/autoencoder{}.pth'.format(u))
        np.savetxt( '/home/Jianghelin/code/Autoencoder/batch_size_value/Training_Mean_Loss/Training_Mean_Loss{}.txt'.format(u), train_mean_loss)


        model1 = torch.load('/home/Jianghelin/code/Autoencoder/batch_size_value/autoencoder/autoencoder{}.pth'.format(u))
        for batch_idx  in range(n_train_batches-1,n_train_batches):
            data1 = []
            data2 = []
            data3 = []
            predict1 = []
            predict2 = []
            predict3 = []
            test_dataset = Mydataset(root,batch_idx)
            test_data = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    #        if hasattr(torch.cuda, 'empty_cache'):
    #	        torch.cuda.empty_cache()
            arr = np.ones((3000,1,3000))
            data_all = torch.tensor(arr) 
            predict_all = torch.tensor(arr) 
            i = 0
            for step,data_in in enumerate(test_data):  
                # img = img.view(img.size(0), -1)
                data_in = Variable(data_in.cuda(3))
                _,predict = model1(data_in)
                data_all[i*batch_size:(i+1)*batch_size,:,:] = data_in
                predict_all[i*batch_size:(i+1)*batch_size,:,:] = predict 
                i = i+1

                predict_loss = criterion(predict,data_in)  
                

                if i ==25:
                    print("predict_loss",predict_loss)
                    data_plt1 = data_in[0,:,0:3000:2].cpu().detach().numpy()+data_in[0,:,1:3000:2].cpu().detach().numpy()*1j
                    data_plt2 = data_in[2,:,0:3000:2].cpu().detach().numpy()+data_in[2,:,1:3000:2].cpu().detach().numpy()*1j
                    data_plt3 = data_in[4,:,0:3000:2].cpu().detach().numpy()+data_in[4,:,1:3000:2].cpu().detach().numpy()*1j
                    data_plt4 = data_in[6,:,0:3000:2].cpu().detach().numpy()+data_in[6,:,1:3000:2].cpu().detach().numpy()*1j
                    data_plt5 = data_in[8,:,0:3000:2].cpu().detach().numpy()+data_in[8,:,1:3000:2].cpu().detach().numpy()*1j
                    data_plt6 = data_in[9,:,0:3000:2].cpu().detach().numpy()+data_in[9,:,1:3000:2].cpu().detach().numpy()*1j 

                    predict_plt1 = predict[0,:,0:3000:2].cpu().detach().numpy()+predict[0,:,1:3000:2].cpu().detach().numpy()*1j
                    predict_plt2 = predict[2,:,0:3000:2].cpu().detach().numpy()+predict[2,:,1:3000:2].cpu().detach().numpy()*1j
                    predict_plt3 = predict[4,:,0:3000:2].cpu().detach().numpy()+predict[4,:,1:3000:2].cpu().detach().numpy()*1j
                    predict_plt4 = predict[6,:,0:3000:2].cpu().detach().numpy()+predict[6,:,1:3000:2].cpu().detach().numpy()*1j
                    predict_plt5 = predict[8,:,0:3000:2].cpu().detach().numpy()+predict[8,:,1:3000:2].cpu().detach().numpy()*1j
                    predict_plt6 = predict[9,:,0:3000:2].cpu().detach().numpy()+predict[9,:,1:3000:2].cpu().detach().numpy()*1j             
                    
                    plt.figure(1)
                    plt.subplot(2,3,1)
                    plt.plot(np.absolute(data_plt1)[0,:],color='red',linestyle='--')
                    plt.plot(np.absolute(predict_plt1)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,2)
                    plt.plot(np.absolute(data_plt2)[0,:],color='red',linestyle='--')
                    plt.plot(np.absolute(predict_plt2)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,3)
                    plt.plot(np.absolute(data_plt3)[0,:],color='red',linestyle='--')
                    plt.plot(np.absolute(predict_plt3)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,4)
                    plt.plot(np.absolute(data_plt4)[0,:],color='red',linestyle='--')
                    plt.plot(np.absolute(predict_plt4)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,5)
                    plt.plot(np.absolute(data_plt5)[0,:],color='red',linestyle='--')
                    plt.plot(np.absolute(predict_plt5)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,6)
                    plt.plot(np.absolute(data_plt6)[0,:],color='red',linestyle='--')
                    plt.plot(np.absolute(predict_plt6)[0,:],color='orange',linestyle=':')
                    plt.savefig("/home/Jianghelin/code/Autoencoder/batch_size_value/pic_save/batch_size_value_Amplitude/batch_size_value_Amplitude{}.png".format(u))
                    plt.show()

                    plt.figure(2)
                    plt.subplot(2,3,1)
                    plt.plot(np.real(data_plt1)[0,:]/np.imag(data_plt1)[0,:],color='red',linestyle='--')
                    plt.plot(np.real(data_plt1)[0,:]/np.imag(data_plt1)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,2)
                    plt.plot(np.real(data_plt2)[0,:]/np.imag(data_plt2)[0,:],color='red',linestyle='--')
                    plt.plot(np.real(data_plt2)[0,:]/np.imag(data_plt2)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,3)
                    plt.plot(np.real(data_plt3)[0,:]/np.imag(data_plt3)[0,:],color='red',linestyle='--')
                    plt.plot(np.real(data_plt3)[0,:]/np.imag(data_plt3)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,4)
                    plt.plot(np.real(data_plt4)[0,:]/np.imag(data_plt4)[0,:],color='red',linestyle='--')
                    plt.plot(np.real(data_plt4)[0,:]/np.imag(data_plt4)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,5)
                    plt.plot(np.real(data_plt5)[0,:]/np.imag(data_plt5)[0,:],color='red',linestyle='--')
                    plt.plot(np.real(data_plt5)[0,:]/np.imag(data_plt5)[0,:],color='orange',linestyle=':')
                    plt.subplot(2,3,6)
                    plt.plot(np.real(data_plt6)[0,:]/np.imag(data_plt6)[0,:],color='red',linestyle='--')
                    plt.plot(np.real(data_plt6)[0,:]/np.imag(data_plt6)[0,:],color='orange',linestyle=':')
                    plt.savefig("/home/Jianghelin/code/Autoencoder/batch_size_value/pic_save/batch_size_value_phase/batch_size_value_phase{}.png".format(u))
                    plt.show()



            predict_loss_all = criterion(predict_all,data_all) 
            print("predict_loss_all",predict_loss_all)
            predict_loss_all_all.append(predict_loss_all.detach().numpy())



            data1 = (data_all[1,:,:]).cpu().numpy()
            data2 = data1[0,:]
            for i in range(100):
                data3.append(data2[i*30])
            predict1 = (predict_all[1,:,:]).detach().numpy()
            predict2 = predict1[0,:]
            for j in range(100):
                predict3.append(predict2[j*30])
            plt.figure()
            plt.plot(data3)
            plt.plot(predict3,color='red',linestyle='--')    #制图
            plt.savefig("/home/Jianghelin/code/Autoencoder/batch_size_value/pic_save/batch_size_value_all/batch_size_value_all{}.png".format(u))
            plt.show()   
    np.savetxt('/home/Jianghelin/code/Autoencoder/batch_size_value/predict_loss_all_all.txt', predict_loss_all_all)
