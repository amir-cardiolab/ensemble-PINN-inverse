import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time

def geo_train(device,xfin,yfin,xd,yd,Td,xbase_fin,ybase_fin,xb_wall_fin, yb_wall_fin, xup_fin, yup_fin, xleft_fin, yleft_fin, xright_fin, yright_fin,Tb_up,Tb_down,batchsize,learning_rate,epochs,path,Flag_batch,Lambda_BC,nPt ):
    if (Flag_batch):
     
     xfin = torch.Tensor(xfin).to(device)
     yfin = torch.Tensor(yfin).to(device)
     xbase_fin = torch.Tensor(xbase_fin).to(device)
     ybase_fin = torch.Tensor(ybase_fin).to(device)
     xb_wall_fin = torch.Tensor(xb_wall_fin).to(device)
     yb_wall_fin = torch.Tensor(yb_wall_fin).to(device)
     xup_fin = torch.Tensor(xup_fin).to(device)
     yup_fin = torch.Tensor(yup_fin).to(device)
     xleft_fin = torch.Tensor(xleft_fin).to(device)
     yleft_fin = torch.Tensor(yleft_fin).to(device)
     xright_fin = torch.Tensor(xright_fin).to(device)
     yright_fin = torch.Tensor(yright_fin).to(device)
     xd = torch.Tensor(xd).to(device)
     yd = torch.Tensor(yd).to(device)
     Td = torch.Tensor(Td).to(device)
     Tb_up = torch.Tensor(Tb_up).to(device)
     Tb_down = torch.Tensor(Tb_down).to(device)
     if(1): #Cuda slower in double? 
        
         xfin = xfin.type(torch.cuda.FloatTensor)
         yfin = yfin.type(torch.cuda.FloatTensor)
         xbase_fin = xbase_fin.type(torch.cuda.FloatTensor)
         ybase_fin = ybase_fin.type(torch.cuda.FloatTensor)
         xb_wall_fin = xb_wall_fin.type(torch.cuda.FloatTensor)
         yb_wall_fin = yb_wall_fin.type(torch.cuda.FloatTensor)
         xup_fin = xup_fin.type(torch.cuda.FloatTensor)
         yup_fin = yup_fin.type(torch.cuda.FloatTensor)
         xleft_fin = xleft_fin.type(torch.cuda.FloatTensor)
         yleft_fin = yleft_fin.type(torch.cuda.FloatTensor)
         xright_fin = xright_fin.type(torch.cuda.FloatTensor)
         yright_fin = yright_fin.type(torch.cuda.FloatTensor)
         xd = xd.type(torch.cuda.FloatTensor)
         yd = yd.type(torch.cuda.FloatTensor)
         Td = Td.type(torch.cuda.FloatTensor)
         Tb_up = Tb_up.type(torch.cuda.FloatTensor)
         Tb_down = Tb_down.type(torch.cuda.FloatTensor)
       
     dataset = TensorDataset(xfin,yfin)
    
     dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False)
    
    else:
     xfluid = torch.Tensor(x_influid).to(device)
     yfluid = torch.Tensor(y_influid).to(device) 
     xfin = torch.Tensor(x_infin).to(device)
     yfin = torch.Tensor(y_infin).to(device) 
     
    h_nd = 200 #neurons to approximate T
    input_n = 2 # Input x,y 
    h_nk = 40 #neurons to approximate k
    
    #activation function
    class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)
                
 

    class Net2_Tc(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_Tc, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                
                
                nn.Linear(h_nd,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            
            return  output        
        
            
    class Net2_kc(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_kc, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_nk),

                Swish(),
                nn.Linear(h_nk,h_nk),
            
                Swish(),
                
                nn.Linear(h_nk,h_nk),
                
                Swish(),
                
                nn.Linear(h_nk,h_nk),
                
                Swish(),
                
                nn.Linear(h_nk,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            
            return  output   
            
    
    ################################################################
    net2_Tc = Net2_Tc().to(device)
    net2_kc = Net2_kc().to(device)
    
    
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            
    # use the modules apply function to recursively apply the initialization
    
    net2_Tc.apply(init_normal)
    net2_kc.apply(init_normal)
    ############################################################################
    optimizer_Tc = optim.Adam(net2_Tc.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_kc = optim.Adam(net2_kc.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    
    ############################################################
    def criterion_2(xfin,yfin):
        
        xfin.requires_grad = True
        yfin.requires_grad = True
        
        net_in = torch.cat((xfin,yfin),1)

        
        Tc = net2_Tc(net_in)
        Tc = Tc.view(len(Tc),-1)
        kc = net2_kc(net_in)
        kc = kc.view(len(kc),-1)
        
        
       
        Tc_x = torch.autograd.grad(Tc,xfin,grad_outputs=torch.ones_like(xfin),create_graph = True,only_inputs=True)[0]
        Tc_xx = torch.autograd.grad(Tc_x,xfin,grad_outputs=torch.ones_like(xfin),create_graph = True,only_inputs=True)[0]
        Tc_y = torch.autograd.grad(Tc,yfin,grad_outputs=torch.ones_like(yfin),create_graph = True,only_inputs=True)[0]
        Tc_yy = torch.autograd.grad(Tc_y,yfin,grad_outputs=torch.ones_like(yfin),create_graph = True,only_inputs=True)[0]
        kc_x = torch.autograd.grad(kc,xfin,grad_outputs=torch.ones_like(xfin),create_graph = True,only_inputs=True)[0]
        kc_y = torch.autograd.grad(kc,yfin,grad_outputs=torch.ones_like(yfin),create_graph = True,only_inputs=True)[0]
        
        loss_5 = ((kc_x)*(Tc_x)) + (kc_y*Tc_y) + (kc*Tc_xx) + (kc*Tc_yy) -1  
        # MSE LOSS
        loss_f = nn.MSELoss()
        

        loss_conduction = loss_f(loss_5,torch.zeros_like(loss_5))

        return loss_conduction
    ###########################################################
    ###################################################################
    def Loss_BC(xbase_fin,ybase_fin,xb_wall_fin, yb_wall_fin, xup_fin, yup_fin, xleft_fin, yleft_fin, xright_fin, yright_fin,Tb_up,Tb_down,xfin,yfin):
        if(0):
            xb = torch.FloatTensor(xb).to(device)
            yb = torch.FloatTensor(yb).to(device)
            ub = torch.FloatTensor(ub).to(device)
            vb = torch.FloatTensor(vb).to(device)
            xb_T = torch.FloatTensor(xb_T).to(device)
            yb_T = torch.FloatTensor(yb_T).to(device)
            Tb = torch.FloatTensor(Tb).to(device)
        
        yup_fin.requires_grad = True
        xleft_fin.requires_grad = True
        xright_fin.requires_grad = True
        
   
        
        net_in_left = torch.cat((xleft_fin, yleft_fin), 1)      
        net_in_right = torch.cat((xright_fin, yright_fin), 1)        
        net_in_up = torch.cat((xup_fin, yup_fin), 1)   
        net_in_base = torch.cat((xbase_fin, ybase_fin), 1)   
        
        kc_left = net2_kc(net_in_left)
        kc_left = kc_left.view(len(kc_left),-1)
        kc_right = net2_kc(net_in_right)
        kc_right = kc_right.view(len(kc_right),-1)
        kc_up = net2_kc(net_in_up)
        kc_up = kc_up.view(len(kc_up),-1)
        
       
        
        out_up_fin = net2_Tc(net_in_up)
        out_up_fin = out_up_fin.view(len(out_up_fin), -1)

       
        out_left_fin = net2_Tc(net_in_left)
        out_left_fin = out_left_fin.view(len(out_left_fin), -1)

        out_right_fin = net2_Tc(net_in_right)
        out_right_fin = out_right_fin.view(len(out_right_fin), -1)
        
     
        out_base = net2_Tc(net_in_base)
        out_base = out_base.view(len(out_base), -1)

        T_y_up_fin = torch.autograd.grad(out_up_fin,yup_fin,grad_outputs=torch.ones_like(yup_fin),create_graph = True,only_inputs=True)[0]
        T_x_left_fin = torch.autograd.grad(out_left_fin,xleft_fin,grad_outputs=torch.ones_like(xleft_fin),create_graph = True,only_inputs=True)[0]
        T_x_right_fin = torch.autograd.grad(out_right_fin,xright_fin,grad_outputs=torch.ones_like(xright_fin),create_graph = True,only_inputs=True)[0]
        

        loss_f = nn.MSELoss()
        
        loss_bc = loss_f(out_base,Tb_down) + loss_f(out_up_fin,Tb_up)+ loss_f(T_x_left_fin,torch.zeros_like(T_x_left_fin)) + loss_f(T_x_right_fin,torch.zeros_like(T_x_right_fin))
        
        return loss_bc
    ##############################################
    ###############################################
    def Loss_data(xd,yd,Td):
    

       
        net_in1 = torch.cat((xd, yd), 1)
        out1_T = net2_Tc(net_in1)
        
        
        out1_T = out1_T.view(len(out1_T), -1)
    
    

        loss_f = nn.MSELoss()
        loss_d = loss_f(out1_T, Td) 


        return loss_d    
        ##############################################################
    # Main loop
    tic = time.time()

    #First learn distance and BC functions
    if (Flag_pretrain):
        print('Reading previous results')
        
         
    # INSTANTIATE STEP LEARNING SCHEDULER CLASS
    if (Flag_schedule):
        
        scheduler_Tc = torch.optim.lr_scheduler.StepLR(optimizer_Tc, step_size=step_epoch, gamma=decay_rate)
        scheduler_kc = torch.optim.lr_scheduler.StepLR(optimizer_kc, step_size=step_epoch, gamma=decay_rate)

    if(Flag_batch):# This one uses dataloader

        
            for epoch in range(epochs):
                loss_bc_n = 0
                loss_eqn_fin_n = 0
                loss_data_n = 0
                n = 0
                for batch_idx, (x_infin, y_infin) in enumerate(dataloader):
                    
                    net2_Tc.zero_grad()
                    net2_kc.zero_grad()
                   
                    loss_eqn_fin = criterion_2(x_infin,y_infin)
                    loss_data = Loss_data(xd,yd,Td)
                    loss_bc = Loss_BC(xbase_fin,ybase_fin,xb_wall_fin, yb_wall_fin, xup_fin, yup_fin, xleft_fin, yleft_fin, xright_fin, yright_fin,Tb_up,Tb_down,xfin,yfin)
                
                    loss =  loss_eqn_fin + Lambda_BC* loss_bc + Lambda_data*loss_data
                    
                    loss.backward()
                    
                    optimizer_Tc.step()
                    optimizer_kc.step()
                    
                    loss_eqn_fin_a =loss_eqn_fin.detach().cpu().numpy()
                    loss_eqn_fin_n += loss_eqn_fin_a
                    loss_bc_a= loss_bc.detach().cpu().numpy()
                    loss_bc_n += loss_bc_a 
                    loss_data_a= loss_data.detach().cpu().numpy()
                    loss_data_n += loss_data_a
                    n += 1         
                      
                            
                    if batch_idx % 40 ==0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f} Loss Eqn_fin {:.15f} Loss BC {:.15f} Loss Data {:.15f}'.format(
                            epoch, batch_idx * len(x_infin), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader), loss.item(), loss_eqn_fin.item(), loss_bc.item(), loss_data.item()))
                
                if (Flag_schedule):
                        
                        scheduler_Tc.step()
                        scheduler_kc.step()
                        
                       
                        
                
                mean_eqn_fin = loss_eqn_fin_n/n
                mean_bc = loss_bc_n/n
                mean_data = loss_data_n/n
                
                print('***Total avg Loss :  Loss eqn_fin {:.10f} Loss BC {:.10f} Loss data {:.10f}'.format( mean_eqn_fin, mean_bc, mean_data) )
                
                print('****Epoch:', epoch,'learning rate is: ', optimizer_Tc.param_groups[0]['lr'])
                
                
                
                if epoch % 1000 == 0:#save network
                
                 torch.save(net2_Tc.state_dict(),path+"inv_Tc_pure_conduction_"+str(epoch)+".pt")
                 torch.save(net2_kc.state_dict(),path+"inv_kc_pure_conduction_"+str(epoch)+".pt")
           
           
            
    toc = time.time()
    elapseTime = toc - tic
    print ("elapse time in parallel = ", elapseTime)
    ###################
    #evaluate models
    net2_Tc.eval()
    net2_kc.eval()
    

    net_in2 = torch.cat((xfin.requires_grad_(),yfin.requires_grad_()),1)
    output_Tc = net2_Tc(net_in2)  #evaluate model
    output_Tc = output_Tc.cpu().data.numpy()
    output_kc = net2_kc(net_in2)  #evaluate model
    output_kc = output_kc.cpu().data.numpy()
    xfin = xfin.cpu()
    yfin = yfin.cpu()


    return


#######################################################
#Main code:

device = torch.device("cuda")

Flag_batch = True  #USe batch or not  
Flag_pretrain = False # for random realization Flag_pretrain_initialization = False
Lambda_BC  = 30  # weight BC loss
Lambda_data = 30 # weight Data loss
Lambda_eqn = 1  # weight Data loss
batchsize = 256  #Total number of batches 
epochs  = 10001

Flag_schedule = True #If true change the learning rate at 3 levels
if (Flag_schedule):
    learning_rate = 3e-4
    step_epoch = 3300 #100
    decay_rate = 0.1

#Domains and number of points
nPt = 200  
nPt1= 100
xStart_out = 0.28
xEnd_in = 0.0
yStart = 0.0
yStart_up = 0.5

xfin = np.linspace(xEnd_in  , xStart_out, nPt)     
yfin = np.linspace(yStart , yStart_up, nPt,endpoint=False)
xfin, yfin = np.meshgrid(xfin, yfin)
xfin = np.reshape(xfin, (np.size(xfin[:]),1))
yfin = np.reshape(yfin, (np.size(yfin[:]),1))

print('shape of xfin',xfin.shape)
print('shape of yfin',yfin.shape)

T_down = 1.0
T_up = 0.85

#boundary conditions
nPt_BC = 3 *nPt
xleft_fin = np.linspace(xEnd_in, xEnd_in, nPt_BC)
yleft_fin = np.linspace(yStart, yStart_up, nPt_BC)
xright_fin = np.linspace(xStart_out, xStart_out, nPt_BC)
yright_fin = np.linspace(yStart, yStart_up, nPt_BC)
xup_fin = np.linspace(xEnd_in, xStart_out, nPt_BC)
yup_fin = np.linspace(yStart_up, yStart_up, nPt_BC)
xbase_fin = np.linspace(xEnd_in, xStart_out, nPt_BC)
ybase_fin = np.linspace(yStart, yStart, nPt_BC)

Tb_down = np.linspace(T_down, T_down, nPt_BC)
Tb_up = np.linspace(T_up, T_up, nPt_BC)



xb_wall_fin = np.concatenate((xleft_fin, xup_fin, xright_fin),0)
yb_wall_fin = np.concatenate((yleft_fin, yup_fin, yright_fin),0)


xup_fin= xup_fin.reshape(-1, 1) #need to reshape to get 2D array
yup_fin= yup_fin.reshape(-1, 1)
xbase_fin= xbase_fin.reshape(-1, 1) #need to reshape to get 2D array
ybase_fin= ybase_fin.reshape(-1, 1)
xleft_fin= xleft_fin.reshape(-1, 1) #need to reshape to get 2D array
yleft_fin= yleft_fin.reshape(-1, 1)
xright_fin= xright_fin.reshape(-1, 1) #need to reshape to get 2D array
yright_fin= yright_fin.reshape(-1, 1)
Tb_up= Tb_up.reshape(-1, 1) #need to reshape to get 2D array
Tb_down= Tb_down.reshape(-1, 1) #need to reshape to get 2D array
############################################
data = []
                
data = np.genfromtxt("result_T_pure_conduction.txt", delimiter= ',');
# data storage:
x_data = data[:,1]
y_data = data[:,2] 
T_data = data[:,0]
 
print("reading and saving cfd done!") 
x_data = np.asarray(x_data)  #convert to numpy 
y_data = np.asarray(y_data) #convert to numpy 
T_data = np.asarray(T_data) #convert to numpy

print('Using input data pts: pts: ',x_data, y_data)
print('Using input data pts: Temp: ',T_data)
xd= x_data.reshape(-1, 1) #need to reshape to get 2D array
yd= y_data.reshape(-1, 1) #need to reshape to get 2D array
Td= T_data.reshape(-1, 1) #need to reshape to get 2D array
#################################################

path = "Results/"


geo_train(device,xfin,yfin,xd,yd,Td,xbase_fin,ybase_fin,xb_wall_fin, yb_wall_fin, xup_fin, yup_fin, xleft_fin, yleft_fin, xright_fin, yright_fin,Tb_up,Tb_down,batchsize,learning_rate,epochs,path,Flag_batch,Lambda_BC,nPt )
