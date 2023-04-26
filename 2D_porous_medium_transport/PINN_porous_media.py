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

def geo_train(device,x_in,y_in,xb_up,yb_up,xb_down,yb_down,xb_p,yb_p,pb,ub,vb,xd,yd,ud,vd,batchsize,learning_rate,epochs,path,Flag_batch,Lambda_BC ):
    if (Flag_batch):
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device)
     xb_down = torch.Tensor(xb_down).to(device)
     yb_down = torch.Tensor(yb_down).to(device)
     xb_up = torch.Tensor(xb_up).to(device)
     yb_up = torch.Tensor(yb_up).to(device)
     xb_p = torch.Tensor(xb_p).to(device)
     yb_p = torch.Tensor(yb_p).to(device)
     xd = torch.Tensor(xd).to(device)
     yd = torch.Tensor(yd).to(device)
     ud = torch.Tensor(ud).to(device)
     vd = torch.Tensor(vd).to(device)
     ub = torch.Tensor(ub).to(device)
     vb = torch.Tensor(vb).to(device)
     pb = torch.Tensor(pb).to(device)
    
     if(1): #Cuda slower in double? 
         x = x.type(torch.cuda.FloatTensor)
         y = y.type(torch.cuda.FloatTensor)
         xb_down = xb_down.type(torch.cuda.FloatTensor)
         yb_down = yb_down.type(torch.cuda.FloatTensor)
         xb_up = xb_up.type(torch.cuda.FloatTensor)
         yb_up = yb_up.type(torch.cuda.FloatTensor)
         ub = ub.type(torch.cuda.FloatTensor) 
         xb_p = xb_p.type(torch.cuda.FloatTensor)
         yb_p = yb_p.type(torch.cuda.FloatTensor)
         vb = vb.type(torch.cuda.FloatTensor)
         pb = pb.type(torch.cuda.FloatTensor)     
         xd = xd.type(torch.cuda.FloatTensor)
         yd = yd.type(torch.cuda.FloatTensor)
         ud = ud.type(torch.cuda.FloatTensor)
         vd = vd.type(torch.cuda.FloatTensor)
     
     dataset = TensorDataset(x,y)
     
     dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
    else:
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device) 
      
    h_n = 170 #neurons to approximate u,v,p
    h_nk = 60 #neurons to approximate k
    input_n = 2 # inputs x,y

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
    
    class Net2_u(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_u, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
            
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            return  output


    class Net2_v(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_v, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
            
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            return  output 

    class Net2_p(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_p, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
            
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            return  output

    class Net2_k(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_k, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_nk),
                
                Swish(),
                nn.Linear(h_nk,h_nk),
                
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
    net2_u = Net2_u().to(device)
    net2_v = Net2_v().to(device)
    net2_p = Net2_p().to(device)
    net2_k = Net2_k().to(device)
   
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)

    # use the modules apply function to recursively apply the initialization
    net2_u.apply(init_normal)
    net2_v.apply(init_normal)
    net2_p.apply(init_normal)
    net2_k.apply(init_normal)
    
    ############################################################################
    #optimizer
    optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_k = optim.Adam(net2_k.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    
    ##############################################################################
    def criterion(x,y):
       
        x.requires_grad = True
        y.requires_grad = True
        
        net_in = torch.cat((x,y),1)

        u = net2_u(net_in)
        u = u.view(len(u),-1)

        v = net2_v(net_in)
        v = v.view(len(v),-1)
        
        P = net2_p(net_in)
        P = P.view(len(P),-1)
        
        K = net2_k(net_in)
        K = K.view(len(K),-1)

       
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

        P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    
        
        
        loss_1 =  v + K*P_y/miu  #Y-dir
        loss_2 = (u *u_scale) + K*P_x/miu #X-dir
        loss_3 = (u_x*u_scale) + v_y #continuity
        
        # MSE LOSS
        loss_f = nn.MSELoss()
        

        #Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1,torch.zeros_like(loss_1))+ loss_f(loss_2,torch.zeros_like(loss_2))+loss_f(loss_3,torch.zeros_like(loss_3))
        
        return loss
    ############################################################
    ###################################################################
    def Loss_BC(xb_up,yb_up,xb_down,yb_down,xb_p,yb_p,x,y):
         
        
        yb_up.requires_grad = True
        yb_down.requires_grad = True
        
        net_in_up = torch.cat((xb_up, yb_up), 1)
        
        out_up_u = net2_u(net_in_up)
        out_up_u = out_up_u.view(len(out_up_u), -1)
        
        out_up_v = net2_v(net_in_up)
        out_up_v = out_up_v.view(len(out_up_v), -1)

        net_in_down = torch.cat((xb_down, yb_down), 1)
       
        out_down_u = net2_u(net_in_down)
        out_down_u = out_down_u.view(len(out_down_u), -1)
        
        out_down_v = net2_v(net_in_down)
        out_down_v = out_down_v.view(len(out_down_v), -1)

        
        net_in_p = torch.cat((xb_p, yb_p), 1)
        out_p = net2_p(net_in_p)
        out_p = out_p.view(len(out_p), -1)

        
        
        u_y_up = torch.autograd.grad(out_up_u,yb_up,grad_outputs=torch.ones_like(yb_up),create_graph = True,only_inputs=True)[0]
        u_y_down = torch.autograd.grad(out_down_u,yb_down,grad_outputs=torch.ones_like(yb_down),create_graph = True,only_inputs=True)[0]

        

        loss_f = nn.MSELoss()
        
        loss_bc = loss_f(out_up_v,torch.zeros_like(out_up_v))+loss_f(out_down_v,torch.zeros_like(out_down_v))+loss_f(out_p, pb) +loss_f(u_y_up,torch.zeros_like(u_y_up))+loss_f(u_y_down,torch.zeros_like(u_y_down))
        
        return loss_bc
########################################################################
    def Loss_data(xd,yd,ud,vd):
    

      
        net_in1 = torch.cat((xd, yd), 1)
        out1_u = net2_u(net_in1)
        out1_v = net2_v(net_in1)
        
        out1_u = out1_u.view(len(out1_u), -1)
        out1_v = out1_v.view(len(out1_v), -1)
        

        loss_f = nn.MSELoss()
        loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd) 


        return loss_d
##################################################################################
    # Main loop
    tic = time.time()
    if (Flag_pretrain):
        print('Reading previous results')
        
   

    # INSTANTIATE STEP LEARNING SCHEDULER CLASS
    if (Flag_schedule):
        scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
        scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
        scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)
        scheduler_k= torch.optim.lr_scheduler.StepLR(optimizer_k, step_size=step_epoch, gamma=decay_rate)


    if(Flag_batch):# This one uses dataloader
            
            for epoch in range(epochs):
                loss_bc_n = 0
                loss_eqn_n = 0
                loss_data_n = 0
                n = 0
                for batch_idx, (x_in,y_in) in enumerate(dataloader):
                    net2_u.zero_grad()
                    net2_v.zero_grad()
                    net2_p.zero_grad()
                    net2_k.zero_grad()
                    
                    
                    loss_eqn = criterion(x_in,y_in) 
                    loss_bc = Loss_BC(xb_up,yb_up,xb_down,yb_down,xb_p,yb_p,x,y)
                    loss_data = Loss_data(xd,yd,ud,vd)
                    
                    loss = loss_eqn + Lambda_BC* loss_bc + Lambda_data*loss_data
                    loss.backward()
                    optimizer_u.step() 
                    optimizer_v.step() 
                    optimizer_p.step()
                    optimizer_k.step()
                    
                    loss_eqn_a =loss_eqn.detach().cpu().numpy()
                    loss_eqn_n += loss_eqn_a
                    loss_bc_a= loss_bc.detach().cpu().numpy()
                    loss_bc_n += loss_bc_a 
                    loss_data_a= loss_data.detach().cpu().numpy()
                    loss_data_n += loss_data_a 
                    n += 1         
                      
                    if batch_idx % 40 ==0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f} Loss eqn {:.10f} Loss BC {:.6f} Loss data {:.6f}'.format(
                            epoch, batch_idx * len(x_in), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader), loss.item(), loss_eqn.item(), loss_bc.item(), loss_data.item()))
                    
                if (Flag_schedule):
                        scheduler_u.step()
                        scheduler_v.step()
                        scheduler_p.step()
                        scheduler_k.step()
                        

                
    
                mean_eqn = loss_eqn_n/n
                mean_bc = loss_bc_n/n
                mean_data = loss_data_n/n
                print('***Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} Loss data {:.10f}'.format(mean_eqn, mean_bc, mean_data) )
                print('****Epoch:', epoch,'learning rate Velocity: ', optimizer_u.param_groups[0]['lr'])
               
                if epoch % 1000 == 0:#save network
                 torch.save(net2_u.state_dict(),path+"inv_step_u_porous_"+str(epoch)+".pt")
                 torch.save(net2_v.state_dict(),path+"inv_step_v_porous_"+str(epoch)+".pt")
                 torch.save(net2_p.state_dict(),path+"inv_step_p_porous_"+str(epoch)+".pt")
                 torch.save(net2_k.state_dict(),path+"inv_step_k_porous_"+str(epoch)+".pt")
                 
            
    toc = time.time()
    elapseTime = toc - tic
    print ("elapse time in parallel = ", elapseTime)
    ###################
    #evaluate models
    net2_u.eval()
    net2_v.eval()
    net2_p.eval()
    net2_k.eval()
   

    net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
    output_u = net2_u(net_in)  #evaluate model
    output_v = net2_v(net_in)  #evaluate model
    output_p = net2_p(net_in)
    output_u = output_u.cpu().data.numpy() #need to convert to cpu before converting to numpy
    output_v = output_v.cpu().data.numpy()
    output_p = output_p.cpu().data.numpy()
    output_k = net2_k(net_in)  #evaluate model
    output_k = output_k.cpu().data.numpy()

   
    x = x.cpu()
    y = y.cpu()

   

    return


    
#######################################################
#Main code:
device = torch.device("cuda")

Flag_batch = True  #USe batch or not  
Flag_pretrain = False

Lambda_BC  = 60.0 # weight BC loss
Lambda_data = 80.0 # weight Data loss

batchsize = 256  #Total number of batches 
epochs  = 10001 
Flag_schedule = True #If true change the learning rate at 3 levels
if (Flag_schedule):
    learning_rate = 3e-4
    step_epoch = 3300 #100
    decay_rate = 0.1


miu = 10.0
#Domain and number of points
nPt = 160 
xStart = 0.0
xEnd = 1.0
yStart = 0.0
yEnd = 1.0

P_in_bc = 1.0
P_out_bc = 0.0


x = np.linspace(xStart, xEnd, nPt)    
y = np.linspace(yStart, yEnd, nPt)
x, y = np.meshgrid(x, y)
x = np.reshape(x, (np.size(x[:]),1))
y = np.reshape(y, (np.size(y[:]),1))

 
print('shape of x',x.shape)
print('shape of y',y.shape)


#boundary conditions
nPt_BC = 2 *nPt
xb_left = np.linspace(xStart, xStart, nPt_BC)
yb_left = np.linspace(yStart, yEnd, nPt_BC)
xb_right = np.linspace(xEnd, xEnd, nPt_BC)
yb_right = np.linspace(yStart, yEnd, nPt_BC)
xb_up = np.linspace(xStart, xEnd, nPt_BC)
yb_up = np.linspace(yEnd, yEnd, nPt_BC)
xb_down = np.linspace(xStart, xEnd, nPt_BC)
yb_down = np.linspace(yStart, yStart, nPt_BC)
P_in= np.linspace(P_in_bc, P_in_bc, nPt_BC)
P_out = np.linspace(P_out_bc, P_out_bc, nPt_BC)
u_wall_BC = np.linspace(0., 0., nPt_BC)
v_wall_BC = np.linspace(0., 0., nPt_BC)

xb_wall = np.concatenate((xb_up, xb_down),0)
yb_wall = np.concatenate((yb_up, yb_down),0) 
ub = np.concatenate((u_wall_BC, u_wall_BC), 0)
vb = np.concatenate((v_wall_BC,v_wall_BC), 0)

xb_p = np.concatenate((xb_left, xb_right),0)
yb_p = np.concatenate((yb_left, yb_right),0) 
pb = np.concatenate((P_in, P_out), 0)



xb_wall= xb_wall.reshape(-1, 1) #need to reshape to get 2D array
yb_wall= yb_wall.reshape(-1, 1) #need to reshape to get 2D array
xb_down= xb_down.reshape(-1, 1) #need to reshape to get 2D array
yb_down= yb_down.reshape(-1, 1) #need to reshape to get 2D array
xb_up= xb_up.reshape(-1, 1) #need to reshape to get 2D array
yb_up= yb_up.reshape(-1, 1) #need to reshape to get 2D array
xb_p= xb_p.reshape(-1, 1) #need to reshape to get 2D array
yb_p= yb_p.reshape(-1, 1) #need to reshape to get 2D array
pb= pb.reshape(-1, 1) #need to reshape to get 2D array
ub= ub.reshape(-1, 1) #need to reshape to get 2D array
vb= vb.reshape(-1, 1) #need to reshape to get 2D array

print('shape of xb_down',xb_down.shape)
print('shape of yb_down',yb_down.shape)
print('shape of xb_up',xb_up.shape) 
print('shape of yb_up',yb_up.shape)
print('shape of xb_p',xb_p.shape)
print('shape of yb_p',yb_p.shape)
print('shape of pb',pb.shape)


############################################
data = []
                
data = np.genfromtxt("result_Darcy_velocity.txt", delimiter= ',');
# data storage:
x_data = data[:,2]
y_data = data[:,3] 
u_data = data[:,0]
v_data = data[:,1]

u_scale = 1.0
u_data = u_data/u_scale

x_data = np.asarray(x_data)  #convert to numpy 
y_data = np.asarray(y_data) #convert to numpy 
u_data = np.asarray(u_data)  #convert to numpy 
v_data = np.asarray(v_data) #convert to numpy 


print("reading and saving cfd done!") 


print('Using input data pts: pts: ',x_data, y_data)
print('Using input data pts: vel: ',u_data, v_data)

xd= x_data.reshape(-1, 1) #need to reshape to get 2D array
yd= y_data.reshape(-1, 1) #need to reshape to get 2D array
ud= u_data.reshape(-1, 1) #need to reshape to get 2D array
vd= v_data.reshape(-1, 1) #need to reshape to get 2D array

path = "Results/"


geo_train(device,x,y,xb_up,yb_up,xb_down,yb_down,xb_p,yb_p,pb,ub,vb,xd,yd,ud,vd,batchsize,learning_rate,epochs,path,Flag_batch,Lambda_BC)







