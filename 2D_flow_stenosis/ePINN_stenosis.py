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
import vtk
from vtk.util import numpy_support as VN
from torch.nn.parameter import Parameter

def geo_train(device,x_in,y_in,xb,yb,ub,vb,xb_in,yb_in,xb_out,yb_out,xd,yd,Td,u_in_BC,v_in_BC,T_bc_in,P_bc_out,Q_wall,batchsize,learning_rate,epochs,path,Flag_batch,Diff,w_eqn,w_bc,w_data,learn_rate_a,step_eph_a,decay_rate_a):
	if (Flag_batch):
	 x = torch.Tensor(x_in).to(device)
	 y = torch.Tensor(y_in).to(device)
	 xb = torch.Tensor(xb).to(device)
	 yb = torch.Tensor(yb).to(device)
	 ub = torch.Tensor(ub).to(device)
	 vb = torch.Tensor(vb).to(device)
	 xb_in = torch.Tensor(xb_in).to(device)
	 yb_in = torch.Tensor(yb_in).to(device)
	 xb_out = torch.Tensor(xb_out).to(device)
	 yb_out = torch.Tensor(yb_out).to(device)
	 u_in_BC = torch.Tensor(u_in_BC).to(device)
	 v_in_BC = torch.Tensor(v_in_BC).to(device)
	 T_bc_in = torch.Tensor(T_bc_in).to(device)
	 P_bc_out = torch.Tensor(P_bc_out).to(device)
	 Q_wall = torch.Tensor(Q_wall).to(device)
	 xd = torch.Tensor(xd).to(device)
	 yd = torch.Tensor(yd).to(device)
	 Td = torch.Tensor(Td).to(device)

	 if(1): #Cuda slower in double? 
		 x = x.type(torch.cuda.FloatTensor)
		 y = y.type(torch.cuda.FloatTensor)
		 xb = xb.type(torch.cuda.FloatTensor)
		 yb = yb.type(torch.cuda.FloatTensor)
		 ub = ub.type(torch.cuda.FloatTensor)
		 vb = vb.type(torch.cuda.FloatTensor)
		 xb_in = xb_in.type(torch.cuda.FloatTensor)
		 yb_in = yb_in.type(torch.cuda.FloatTensor)
		 xb_out = xb_out.type(torch.cuda.FloatTensor)
		 yb_out = yb_out.type(torch.cuda.FloatTensor)
		 u_in_BC = u_in_BC.type(torch.cuda.FloatTensor)
		 v_in_BC = v_in_BC.type(torch.cuda.FloatTensor)
		 T_bc_in = T_bc_in.type(torch.cuda.FloatTensor)
		 P_bc_out = P_bc_out.type(torch.cuda.FloatTensor)
		 Q_wall = Q_wall.type(torch.cuda.FloatTensor)
		 xd = xd.type(torch.cuda.FloatTensor)
		 yd = yd.type(torch.cuda.FloatTensor)
		 Td = Td.type(torch.cuda.FloatTensor)
# 		 
	 dataset = TensorDataset(x,y)
	 dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
	else:
	 x = torch.Tensor(x_in).to(device)
	 y = torch.Tensor(y_in).to(device) 
	     
	h_nd = 200 #neurons to approximate concentration (T)
	h_n = 100 #neurons to approximate u,v
	input_n = 2 #input x,y 

	#Activation function
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
				
	class Swish_u(nn.Module):																		
		
		def __init__(self):						
			super().__init__()


		def forward(self, x):
			output = nn.SiLU()

			return output(a[0]*n*x)
			
	class Swish_v(nn.Module):																		
		
		def __init__(self):						
			super().__init__()


		def forward(self, x):
			output = nn.SiLU()

			return output(a[1]*n*x)
				
	class Swish_T(nn.Module):																		
		
		def __init__(self):						
			super().__init__()


		def forward(self, x):
			output = nn.SiLU()

			return output(a[3]*n*x)	
	

	class Net2_T(Swish_T):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_T, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_nd),
			
				Swish_T(),
				nn.Linear(h_nd,h_nd),
				
				Swish_T(),
				nn.Linear(h_nd,h_nd),
				
				Swish_T(),
				nn.Linear(h_nd,h_nd),
				
				Swish_T(),
				nn.Linear(h_nd,h_nd),
				
				Swish_T(),
				nn.Linear(h_nd,h_nd),
				
				Swish_T(),
				nn.Linear(h_nd,h_nd),
				
				Swish_T(),
				nn.Linear(h_nd,h_nd),
				
				Swish_T(),
				

				nn.Linear(h_nd,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output      
	
	
   
	class U1(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U1, self).__init__()
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
				
				

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
	class U2(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U2, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output        
			
	class U3(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U3, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
			
	class U4(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U4, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
			
	class U5(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U5, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output        
			
	class U6(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U6, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
			
	class U7(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U7, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output 
	
	
	class U8(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U8, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
	class U9(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U9, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output  
			
			
	class U10(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(U10, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output        
			
					
	class V1(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V1, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
	class V2(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V2, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output        
			
	class V3(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V3, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
			
	class V4(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V4, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
			
	class V5(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V5, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output        
			
	class V6(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V6, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
			
	class V7(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V7, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output 
	
	
	class V8(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V8, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output
			
	class V9(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V9, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output  
			
			
	class V10(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(V10, self).__init__()
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

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			
			return  output        
			
	
			
	
	class Net2_u(Swish_u):  

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_u, self).__init__()
   
		 
			self.u1 = U1()
			self.u2 = U2()
			self.u3 = U3()
			self.u4 = U4()
			self.u5 = U5()
			self.u6 = U6()
			self.u7 = U7()
			self.u8 = U8()
			self.u9 = U9()
			self.u10 = U10()
			
			self.final = nn.Sequential(
				nn.Linear(10,100),         #outputs are taken as inputs
				Swish_u(),
				nn.Linear(100,100),
				Swish_u(),
				nn.Linear(100,100),
				Swish(),
				nn.Linear(100,1),
				
			)


		def forward(self,x): 
			u1 = self.u1(x)
			u2 = self.u2(x)
			u3 = self.u3(x)
			u4 = self.u4(x)
			u5 = self.u5(x)
			u6 = self.u6(x)
			u7 = self.u7(x)
			u8 = self.u8(x)
			u9 = self.u9(x)
			u10 = self.u10(x)
			
			
			output = torch.cat((u1,u2,u3,u4,u5,u6,u7,u8,u9,u10), dim=1 )
			output2 = self.final(output)
		   
			
			return  output2
	   
	class Net2_v(Swish_v):  

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_v, self).__init__()
   
		 
			self.v1 = V1()
			self.v2 = V2()
			self.v3 = V3()
			self.v4 = V4()
			self.v5 = V5()
			self.v6 = V6()
			self.v7 = V7()
			self.v8 = V8()
			self.v9 = V9()
			self.v10 = V10()
			
			self.final = nn.Sequential(
				nn.Linear(10,100),         ##outputs are taken as inputs
				Swish_v(),
				nn.Linear(100,100),
				Swish_v(),
				nn.Linear(100,100),
				Swish(),
				nn.Linear(100,1),
				
			)


		def forward(self,x): 
			v1 = self.v1(x)
			v2 = self.v2(x)
			v3 = self.v3(x)
			v4 = self.v4(x)
			v5 = self.v5(x)
			v6 = self.v6(x)
			v7 = self.v7(x)
			v8 = self.v8(x)
			v9 = self.v9(x)
			v10 = self.v10(x)
			
			
			output = torch.cat((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10), dim=1 )
			output2 = self.final(output)
		   
			
			return  output2
	################################################################
	AU = Swish_u().to(device)
	AV = Swish_v().to(device)
	AT = Swish_T().to(device)
	
	net2_u = Net2_u().to(device)
	net2_v = Net2_v().to(device)
	net2_T = Net2_T().to(device)
	
	u10 = U10().to(device)
	u9 = U9().to(device)
	u8 = U8().to(device)
	u7 = U7().to(device)
	u6 = U6().to(device)
	u5 = U5().to(device)
	u4 = U4().to(device)
	u3 = U3().to(device)
	u2 = U2().to(device)
	u1 = U1().to(device)
	
	v10 = V10().to(device)
	v9 = V9().to(device)
	v8 = V8().to(device)
	v7 = V7().to(device)
	v6 = V6().to(device)
	v5 = V5().to(device)
	v4 = V4().to(device)
	v3 = V3().to(device)
	v2 = V2().to(device)
	v1 = V1().to(device)
	
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net2_T.apply(init_normal)

	############################################################################
# Optimizer
	optimizer_w_eqn  = optim.Adam([{'params':[w_eqn], 'lr':learn_rate_a}], maximize = True, betas = (0.9,0.99),eps = 10**-15)
	optimizer_w_bc   = optim.Adam([{'params':[w_bc],  'lr':learn_rate_a}], maximize = True, betas = (0.9,0.99),eps = 10**-15)
	optimizer_w_data   = optim.Adam([{'params':[w_data],  'lr':learn_rate_a}], maximize = True, betas = (0.9,0.99),eps = 10**-15)
	optimizer_af = optim.Adam([{'params':[a], 'lr':learn_rate_a}], betas = (0.9,0.99),eps = 10**-15)
	optimizer_u = optim.Adam(filter(lambda p: p.requires_grad, net2_u.parameters()), lr=learning_rate,betas = (0.9,0.99),eps = 10**-15)
	optimizer_v = optim.Adam(filter(lambda p: p.requires_grad, net2_v.parameters()), lr=learning_rate,betas = (0.9,0.99),eps = 10**-15)
	optimizer_T = optim.Adam(net2_T.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	
	def criterion(x,y):
		
		x.requires_grad = True
		y.requires_grad = True
		
		net_in = torch.cat((x,y),1)
		u = net2_u(net_in)
		u = u.view(len(u),-1)
		v = net2_v(net_in)
		v = v.view(len(v),-1)
		T = net2_T(net_in)
		T = T.view(len(T),-1)

		
		u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		T_x = torch.autograd.grad(T,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		T_xx = torch.autograd.grad(T_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		T_y = torch.autograd.grad(T,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		T_yy = torch.autograd.grad(T_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

		XX_scale = (X_scale**2)
		YY_scale = (Y_scale**2)
		UU_scale  = U_scale **2
		TT_scale = (T_scale**2)
	
		loss_3 = (u_x*U_scale / X_scale + v_y / Y_scale) #continuity
		loss_4 = u*(T_x/X_scale)+v*T_y - Diff*((T_xx/XX_scale)+ T_yy )    #energy

		# MSE LOSS
		loss_f = nn.MSELoss()
		

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_3,torch.zeros_like(loss_3))+loss_f(loss_4,torch.zeros_like(loss_4)) #loss
		
		return loss
	############################################################
	###########################################################
	###################################################################
	def Loss_BC(xb,yb,ub,vb,xb_in,yb_in,u_in_BC,v_in_BC,T_bc_in,xb_out,yb_out,P_bc_out,Q_wall,x,y):
		
		yb.requires_grad = True
		
		net_in = torch.cat((xb_in, yb_in), 1)
		out1_T_in = net2_T(net_in )
		out1_T_in = out1_T_in.view(len(out1_T_in), -1)
		
	
		net_in1 = torch.cat((xb, yb), 1)
		out1_T_wall = net2_T(net_in1 )
		out1_T_wall = out1_T_wall.view(len(out1_T_wall), -1)
		T_y = torch.autograd.grad(out1_T_wall,yb,grad_outputs=torch.ones_like(yb),create_graph = True,only_inputs=True)[0]

		

		loss_f = nn.MSELoss()
		loss_bc = loss_f(out1_T_in, torch.zeros_like(out1_T_in)) + loss_f(T_y, Q_wall)
	
		return loss_bc
	##############################################
	###############################################
	def Loss_data(xd,yd,Td):
	
		net_in1 = torch.cat((xd, yd), 1)
		out1_T = net2_T(net_in1)
		out1_T = out1_T.view(len(out1_T), -1)
		
		loss_f = nn.MSELoss()
		loss_d = loss_f(out1_T, Td) 


		return loss_d    
		####################################################

	# Main loop
	tic = time.time()

	#First learn distance and BC functions
	if (Flag_pretrain_initialization):
		print('Reading previous results')
		
		net2_u.u1.load_state_dict(torch.load(path+"u_stenosis_1_200"+".pt"))
		net2_u.u2.load_state_dict(torch.load(path+"u_stenosis_2_200"+".pt"))
		net2_u.u3.load_state_dict(torch.load(path+"u_stenosis_3_200"+".pt"))
		net2_u.u4.load_state_dict(torch.load(path+"u_stenosis_4_200"+".pt"))
		net2_u.u5.load_state_dict(torch.load(path+"u_stenosis_5_200"+".pt"))
		net2_u.u6.load_state_dict(torch.load(path+"u_stenosis_6_200"+".pt"))
		net2_u.u7.load_state_dict(torch.load(path+"u_stenosis_7_200"+".pt"))
		net2_u.u8.load_state_dict(torch.load(path+"u_stenosis_8_200"+".pt"))
		net2_u.u9.load_state_dict(torch.load(path+"u_stenosis_9_200"+".pt"))
		net2_u.u10.load_state_dict(torch.load(path+"u_stenosis_10_200"+".pt"))
		net2_v.v1.load_state_dict(torch.load(path+"v_stenosis_1_200"+".pt"))
		net2_v.v2.load_state_dict(torch.load(path+"v_stenosis_2_200"+".pt"))
		net2_v.v3.load_state_dict(torch.load(path+"v_stenosis_3_200"+".pt"))
		net2_v.v4.load_state_dict(torch.load(path+"v_stenosis_4_200"+".pt"))
		net2_v.v5.load_state_dict(torch.load(path+"v_stenosis_5_200"+".pt"))
		net2_v.v6.load_state_dict(torch.load(path+"v_stenosis_6_200"+".pt"))
		net2_v.v7.load_state_dict(torch.load(path+"v_stenosis_7_200"+".pt"))
		net2_v.v8.load_state_dict(torch.load(path+"v_stenosis_8_200"+".pt"))
		net2_v.v9.load_state_dict(torch.load(path+"v_stenosis_9_200"+".pt"))
		net2_v.v10.load_state_dict(torch.load(path+"v_stenosis_10_200"+".pt"))
		
	
	net2_u.eval()
	net2_v.eval()
	
	if (Flag_Freezing_layer):
		for parameter_1u in net2_u.u1.main.parameters():
			 parameter_1u.requires_grad = False            
		   
		for parameter_2u in net2_u.u2.main.parameters():
			 parameter_2u.requires_grad = False 
			 
		for parameter_3u in net2_u.u3.main.parameters():
			 parameter_3u.requires_grad = False      

		for parameter_4u in net2_u.u4.main.parameters():
			 parameter_4u.requires_grad = False 
			 
		for parameter_5u in net2_u.u5.main.parameters():
			 parameter_5u.requires_grad = False 
			 
		for parameter_6u in net2_u.u6.main.parameters():
			 parameter_6u.requires_grad = False      
			 
		for parameter_7u in net2_u.u7.main.parameters():
			 parameter_7u.requires_grad = False      

		for parameter_8u in net2_u.u8.main.parameters():
			 parameter_8u.requires_grad = False      

		for parameter_9u in net2_u.u9.main.parameters():
			 parameter_9u.requires_grad = False 
			 
		for parameter_10u in net2_u.u10.main.parameters():
			 parameter_10u.requires_grad = False       
			 
			 
		for parameter_1v in net2_v.v1.main.parameters():
			 parameter_1v.requires_grad = False            
		   
		for parameter_2v in net2_v.v2.main.parameters():
			 parameter_2v.requires_grad = False 
			 
		for parameter_3v in net2_v.v3.main.parameters():
			 parameter_3v.requires_grad = False      

		for parameter_4v in net2_v.v4.main.parameters():
			 parameter_4v.requires_grad = False 
			 
		for parameter_5v in net2_v.v5.main.parameters():
			 parameter_5v.requires_grad = False 
			 
		for parameter_6v in net2_v.v6.main.parameters():
			 parameter_6v.requires_grad = False      
			 
		for parameter_7v in net2_v.v7.main.parameters():
			 parameter_7v.requires_grad = False      

		for parameter_8v in net2_v.v8.main.parameters():
			 parameter_8v.requires_grad = False      

		for parameter_9v in net2_v.v9.main.parameters():
			 parameter_9v.requires_grad = False 
			 
		for parameter_10v in net2_v.v10.main.parameters():
			 parameter_10v.requires_grad = False 

	
	# INSTANTIATE STEP LEARNING SCHEDULER CLASS
	if (Flag_schedule):
		scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
		scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
		scheduler_T = torch.optim.lr_scheduler.StepLR(optimizer_T, step_size=step_epoch, gamma=decay_rate)
		scheduler_af = torch.optim.lr_scheduler.StepLR(optimizer_af, step_size=step_eph_a, gamma=decay_rate)
		scheduler_w_eqn  = torch.optim.lr_scheduler.StepLR(optimizer_w_eqn,  step_size=step_eph_a, gamma=decay_rate_a)
		scheduler_w_bc   = torch.optim.lr_scheduler.StepLR(optimizer_w_bc,   step_size=step_eph_a, gamma=decay_rate_a)
		scheduler_w_data   = torch.optim.lr_scheduler.StepLR(optimizer_w_data,   step_size=step_eph_a, gamma=decay_rate_a)
	
	# ADAPTIVE ACTIVATION FUNCTION
	Param_au  = torch.empty(size=(epochs, 1))
	Param_av  = torch.empty(size=(epochs, 1))
	Param_aT  = torch.empty(size=(epochs, 1))
	
	Param_au [0] = a[0]
	Param_av [0] = a[1]
	Param_aT [0] = a[3]
	
	# LAMBDA FUNCTIONS
	Lambda_eqn  = torch.empty(size=(epochs, 1))
	Lambda_bc   = torch.empty(size=(epochs, 1))
	Lambda_data   = torch.empty(size=(epochs, 1))
	Lambda_eqn [0] = w_eqn
	Lambda_bc [0] = w_bc
	Lambda_data[0] = w_data	
	
################################
	if(Flag_batch):# This one uses dataloader

			
			for epoch in range(epochs):
				loss_bc_n = 0
				loss_eqn_n = 0
				loss_data_n = 0
				N = 0
				for batch_idx, (x_in,y_in) in enumerate(dataloader):
					optimizer_af.zero_grad()
					optimizer_u.zero_grad() 
					optimizer_v.zero_grad() 
					optimizer_T.zero_grad()
					# optimizer_w_eqn.zero_grad()
					optimizer_w_bc.zero_grad()
					optimizer_w_data.zero_grad()
					
					loss_eqn = criterion(x_in,y_in)
					loss_bc = Loss_BC(xb,yb,ub,vb,xb_in,yb_in,u_in_BC,v_in_BC,T_bc_in,xb_out,yb_out,P_bc_out,Q_wall,x,y)
					loss_data = Loss_data(xd,yd,Td)
				
					loss = loss_eqn +  w_bc* loss_bc + w_data*loss_data
					loss.backward()
					
					optimizer_af.step()
					# optimizer_w_eqn.step()
					optimizer_w_bc.step()
					optimizer_w_data.step()
					optimizer_u.step() 
					optimizer_v.step() 
					optimizer_T.step()
					
					loss_data_a= loss_data.detach().cpu().numpy()
					loss_data_n += loss_data_a
					loss_eqn_a =loss_eqn.detach().cpu().numpy()
					loss_eqn_n += loss_eqn_a
					loss_bc_a= loss_bc.detach().cpu().numpy()
					loss_bc_n += loss_bc_a  
					N += 1         
					  
					if batch_idx % 40 ==0:
						print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f} Loss Eqn {:.10f} Loss BC {:.6f} Loss Data {:.6f}'.format(
							epoch, batch_idx * len(x_in), len(dataloader.dataset),
							100. * batch_idx / len(dataloader), loss.item(), loss_eqn.item(), loss_bc.item(), loss_data.item()))
					
				if (Flag_schedule):
						scheduler_u.step()
						scheduler_v.step()
						scheduler_T.step()
						# scheduler_w_eqn.step()
						scheduler_w_bc.step()  
						scheduler_w_data.step()
						scheduler_af.step()  
				
				mean_eqn = loss_eqn_n/N
				mean_bc = loss_bc_n/N
				mean_data = loss_data_n/N
				
				Param_au  [epoch + 1] = a[0]
				Param_av  [epoch + 1] = a[1]
				Param_aT  [epoch + 1] = a[3]
				
				print('***Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} Loss Data {:.10f}'.format(mean_eqn, mean_bc, mean_data) )
				print('****Epoch:', epoch,'learning rate is: ', optimizer_u.param_groups[0]['lr'])
				print("AF PARAMETERS:\n", "AU =", Param_au[epoch].item(), "AV =", Param_av[epoch].item(), "AT =", Param_aT[epoch].item())
				print("LAMBDA PARAMETERS:")
				print("W_BC =", w_bc.item(), "W_DATA =", w_data.item())
				
				
				if epoch % 1000 == 0:#save network
				 torch.save(net2_u.state_dict(),path+"parallel_step_u_stenosis_"+str(epoch)+".pt")
				 torch.save(net2_v.state_dict(),path+"parallel_step_v_stenosis_"+str(epoch)+".pt")
				 torch.save(net2_T.state_dict(),path+"parallel_step_T_stenosis_"+str(epoch)+".pt")
				 
			
	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	###################
	net2_u.eval()
	net2_v.eval()
	net2_T.eval()
	
	
	net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
	output_u = net2_u(net_in)  #evaluate model
	output_v = net2_v(net_in)  #evaluate model
	output_T = net2_T(net_in)  #evaluate model
	

	output_u = output_u.cpu().data.numpy() #need to convert to cpu before converting to numpy
	output_v = output_v.cpu().data.numpy()
	output_T = output_T.cpu().data.numpy()
	x = x.cpu()
	y = y.cpu()


	
	return


#######################################################
#Main code:

device = torch.device("cuda")

Flag_batch = True #USe batch or not  
Flag_pretrain_initialization = True # for random realization Flag_pretrain_initialization = False
Flag_Freezing_layer = True # freeze layers, set to False without freezing 

n = 1.0  	# Scaling factor 
a = Parameter(torch.ones(4))	# Parameter 	
batchsize = 256  #Total number of batches 
epochs  = 10001

Flag_schedule = True #If true change the learning rate at 3 levels
if (Flag_schedule):
	learning_rate = 3e-4
	learn_rate_a  = 1e-4
	step_epoch = 3300 
	step_eph_a = 3300
	decay_rate = 0.1
	decay_rate_a = 0.1
w_eqn  = Parameter(torch.tensor(20.0))
w_bc   = Parameter(torch.tensor(30.0))
w_data   = Parameter(torch.tensor(80.0))

Diff = 0.05  #Diffusion Coeff.
X_scale = 2.0 #The length of the  domain 
Y_scale = 1.0 
U_scale = 1.0
T_scale = 1.0 
U_BC_in = 0.5 
T_BC_in = 0.0
q_wall = 0.0001 #heat flux

Directory = "/scratch/ma3367/Files/Stenosis/Parallel/"
mesh_file = Directory + "sten_mesh.vtu"
bc_file_in = Directory + "inlet_BC.vtk"
bc_file_wall = Directory + "wall_BC.vtk"
bc_file_out = Directory + "outlet_BC.vtk"

print ('Loading', mesh_file)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of the mesh:' ,n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]   
	y_vtk_mesh[i] = pt_iso[1]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

x  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))


print('shape of x',x.shape)
print('shape of y',y.shape)
print ('Loading', bc_file_in)
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(bc_file_in)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of at inlet' ,n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb_in  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh[:]),1)) 
yb_in  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh[:]),1))

print ('Loading', bc_file_wall)
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(bc_file_wall)
reader.Update()
data_vtk = reader.GetOutput()
n_pointsw = data_vtk.GetNumberOfPoints()
print ('n_points of at wall' ,n_pointsw)
x_vtk_mesh = np.zeros((n_pointsw,1))
y_vtk_mesh = np.zeros((n_pointsw,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_pointsw):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
yb  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))

print ('Loading', bc_file_out)
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(bc_file_out)
reader.Update()
data_vtk = reader.GetOutput()
n_pointso = data_vtk.GetNumberOfPoints()
print ('n_points of at out' ,n_pointso)
x_vtk_mesh = np.zeros((n_pointso,1))
y_vtk_mesh = np.zeros((n_pointso,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_pointso):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb_out  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh[:]),1)) 
yb_out  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh[:]),1))
u_in_BC = (yb_in[:]) * ( 0.3 - yb_in[:] )  / 0.0225 * U_BC_in #parabolic

v_in_BC = np.linspace(0., 0., n_points)
ub = np.linspace(0., 0., n_pointsw)
vb = np.linspace(0., 0., n_pointsw)
T_bc_in = np.linspace(0., 0., n_points)
P_bc_out = np.linspace(0., 0., n_pointso)
Q_wall = np.linspace(q_wall, q_wall, n_pointsw)

xb= xb.reshape(-1, 1) #need to reshape to get 2D array
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
ub= ub.reshape(-1, 1) #need to reshape to get 2D array
vb= vb.reshape(-1, 1) #need to reshape to get 2D array
xb_in= xb_in.reshape(-1, 1) #need to reshape to get 2D array
yb_in= yb_in.reshape(-1, 1) #need to reshape to get 2D array
u_in_BC= u_in_BC.reshape(-1, 1) #need to reshape to get 2D array
v_in_BC= v_in_BC.reshape(-1, 1) #need to reshape to get 2D array
T_bc_in = T_bc_in.reshape(-1,1)
P_bc_out = P_bc_out.reshape(-1,1)
Q_wall = Q_wall.reshape(-1,1)
print('shape of xb',xb.shape)
print('shape of yb',yb.shape)
print('shape of ub',ub.shape)
print('shape of vb',vb.shape)

data = np.genfromtxt("result_temp_stenosis.txt", delimiter= ',');
# data storage:
x_data = data[:,1]
y_data = data[:,2] 
T_data = data[:,0]
 
print("reading and saving cfd done!") 
x_data = np.asarray(x_data)  #convert to numpy 
x_data = x_data/X_scale
y_data = np.asarray(y_data) #convert to numpy 
T_data = np.asarray(T_data) #convert to numpy
T_data = T_data / T_scale

print('Using input data pts: pts: ',x_data, y_data)
print('Using input data pts: Temp: ',T_data)
xd= x_data.reshape(-1, 1) #need to reshape to get 2D array
yd= y_data.reshape(-1, 1) #need to reshape to get 2D array
Td= T_data.reshape(-1, 1) #need to reshape to get 2D array



path = "Results/"


geo_train(device,x,y,xb,yb,ub,vb,xb_in,yb_in,xb_out,yb_out,xd,yd,Td,u_in_BC,v_in_BC,T_bc_in,P_bc_out,Q_wall,batchsize,learning_rate,epochs,path,Flag_batch,Diff,w_eqn,w_bc,w_data,learn_rate_a,step_eph_a,decay_rate_a )










