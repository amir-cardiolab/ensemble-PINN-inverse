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


def geo_train(device,x_influid,y_influid,x_infin,y_infin,xd,yd,Td,xb,yb,xleft,yleft,xbase_fin,ybase_fin,xb_wall,yb_wall,xb_wall_fluid,yb_wall_fluid,xb_wall_fin, yb_wall_fin,xup_fluid, yup_fluid, xleft_fluid, yleft_fluid, xright_fluid, yright_fluid, xup_fin, yup_fin, xleft_fin, yleft_fin, xright_fin, yright_fin,ub,vb,Tb,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Lambda_BC,nPt ):
	if (Flag_batch):
	 xfluid = torch.Tensor(x_influid).to(device)
	 yfluid = torch.Tensor(y_influid).to(device)
	 xfin = torch.Tensor(x_infin).to(device)
	 yfin = torch.Tensor(y_infin).to(device)
	 xb = torch.Tensor(xb).to(device)
	 yb = torch.Tensor(yb).to(device)
	 ub = torch.Tensor(ub).to(device)
	 vb = torch.Tensor(vb).to(device)
	 Tb = torch.Tensor(Tb).to(device)
	 xb_wall = torch.Tensor(xb_wall).to(device)
	 yb_wall = torch.Tensor(yb_wall).to(device)
	 xleft = torch.Tensor(xleft).to(device)
	 yleft = torch.Tensor(yleft).to(device)
	 xbase_fin = torch.Tensor(xbase_fin).to(device)
	 ybase_fin = torch.Tensor(ybase_fin).to(device)
	 xb_wall_fluid = torch.Tensor(xb_wall_fluid).to(device)
	 yb_wall_fluid = torch.Tensor(yb_wall_fluid).to(device)
	 xb_wall_fin = torch.Tensor(xb_wall_fin).to(device)
	 yb_wall_fin = torch.Tensor(yb_wall_fin).to(device)
	 xup_fluid = torch.Tensor(xup_fluid).to(device)
	 yup_fluid = torch.Tensor(yup_fluid).to(device)
	 xleft_fluid = torch.Tensor(xleft_fluid).to(device)
	 yleft_fluid = torch.Tensor(yleft_fluid).to(device)
	 xright_fluid = torch.Tensor(xright_fluid).to(device)
	 yright_fluid = torch.Tensor(yright_fluid).to(device)
	 xup_fin = torch.Tensor(xup_fin).to(device)
	 yup_fin = torch.Tensor(yup_fin).to(device)
	 xleft_fin = torch.Tensor(xleft_fin).to(device)
	 yleft_fin = torch.Tensor(yleft_fin).to(device)
	 xright_fin = torch.Tensor(xright_fin).to(device)
	 yright_fin = torch.Tensor(yright_fin).to(device)
	 xd = torch.Tensor(xd).to(device)
	 yd = torch.Tensor(yd).to(device)
	 Td = torch.Tensor(Td).to(device)
	 if(1): #Cuda slower in double? 
		 xfluid = xfluid.type(torch.cuda.FloatTensor)
		 yfluid = yfluid.type(torch.cuda.FloatTensor)
		 xfin = xfin.type(torch.cuda.FloatTensor)
		 yfin = yfin.type(torch.cuda.FloatTensor)
		 xb = xb.type(torch.cuda.FloatTensor)
		 yb = yb.type(torch.cuda.FloatTensor)
		 ub = ub.type(torch.cuda.FloatTensor)
		 vb = vb.type(torch.cuda.FloatTensor)
		 Tb = Tb.type(torch.cuda.FloatTensor)
		 xb_wall = xb_wall.type(torch.cuda.FloatTensor)
		 yb_wall = yb_wall.type(torch.cuda.FloatTensor)
		 xleft = xleft.type(torch.cuda.FloatTensor)
		 yleft = yleft.type(torch.cuda.FloatTensor)
		 xbase_fin = xbase_fin.type(torch.cuda.FloatTensor)
		 ybase_fin = ybase_fin.type(torch.cuda.FloatTensor)
		 xb_wall_fluid = xb_wall_fluid.type(torch.cuda.FloatTensor)
		 yb_wall_fluid = yb_wall_fluid.type(torch.cuda.FloatTensor)
		 xb_wall_fin = xb_wall_fin.type(torch.cuda.FloatTensor)
		 yb_wall_fin = yb_wall_fin.type(torch.cuda.FloatTensor)
		 xup_fluid = xup_fluid.type(torch.cuda.FloatTensor)
		 yup_fluid = yup_fluid.type(torch.cuda.FloatTensor)
		 xleft_fluid = xleft_fluid.type(torch.cuda.FloatTensor)
		 yleft_fluid = yleft_fluid.type(torch.cuda.FloatTensor)
		 xright_fluid = xright_fluid.type(torch.cuda.FloatTensor)
		 yright_fluid = yright_fluid.type(torch.cuda.FloatTensor)
		 xup_fin = xup_fin.type(torch.cuda.FloatTensor)
		 yup_fin = yup_fin.type(torch.cuda.FloatTensor)
		 xleft_fin = xleft_fin.type(torch.cuda.FloatTensor)
		 yleft_fin = yleft_fin.type(torch.cuda.FloatTensor)
		 xright_fin = xright_fin.type(torch.cuda.FloatTensor)
		 yright_fin = yright_fin.type(torch.cuda.FloatTensor)
		 xd = xd.type(torch.cuda.FloatTensor)
		 yd = yd.type(torch.cuda.FloatTensor)
		 Td = Td.type(torch.cuda.FloatTensor)
	
	 dataset = TensorDataset(xfluid,yfluid,xfin,yfin)
	 dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False)
	else:
	 xfluid = torch.Tensor(x_influid).to(device)
	 yfluid = torch.Tensor(y_influid).to(device) 
	 xfin = torch.Tensor(x_infin).to(device)
	 yfin = torch.Tensor(y_infin).to(device) 
	 
	h_nd = 180 #number of neurons to approximate T
	h_n = 150  #number of neurons to approximate u,v,p
	input_n = 2 #input x,y
	h_nk = 40 #number of neurons to approximate k

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
				nn.Linear(h_n,h_n),
				
				Swish(),
				nn.Linear(h_n,h_n),
				

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
				nn.Linear(h_n,h_n),
				
				Swish(),
				nn.Linear(h_n,h_n),
				

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
		   
			return  output

	class Net2_T(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_T, self).__init__()
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
			# with torch.no_grad():
			 output = self.main(x)
			
			 return  output   
			
	
   
	################################################################
	net2_u = Net2_u().to(device)
	net2_v = Net2_v().to(device)
	net2_p = Net2_p().to(device)
	net2_T = Net2_T().to(device)
	net2_Tc = Net2_Tc().to(device)
	net2_kc = Net2_kc().to(device)
	
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)
		   
	# use the modules apply function to recursively apply the initialization
	net2_u.apply(init_normal)
	net2_v.apply(init_normal)
	net2_p.apply(init_normal)
	net2_T.apply(init_normal)
	net2_Tc.apply(init_normal)
	net2_kc.apply(init_normal)
	############################################################################
	#optimizer
	optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_T = optim.Adam(net2_T.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_Tc = optim.Adam(net2_Tc.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_kc = optim.Adam(net2_kc.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	
########################################
	def criterion(xfluid,yfluid):
	
		xfluid.requires_grad = True
		yfluid.requires_grad = True
		
		net_in = torch.cat((xfluid,yfluid),1)
		u = net2_u(net_in)
		u = u.view(len(u),-1)
		v = net2_v(net_in)
		v = v.view(len(v),-1)
		P = net2_p(net_in)
		P = P.view(len(P),-1)
		T = net2_T(net_in)
		T = T.view(len(T),-1)
		
		u_x = torch.autograd.grad(u,xfluid,grad_outputs=torch.ones_like(xfluid),create_graph = True,only_inputs=True)[0]
		u_xx = torch.autograd.grad(u_x,xfluid,grad_outputs=torch.ones_like(xfluid),create_graph = True,only_inputs=True)[0]
		u_y = torch.autograd.grad(u,yfluid,grad_outputs=torch.ones_like(yfluid),create_graph = True,only_inputs=True)[0]
		u_yy = torch.autograd.grad(u_y,yfluid,grad_outputs=torch.ones_like(yfluid),create_graph = True,only_inputs=True)[0]
		v_x = torch.autograd.grad(v,xfluid,grad_outputs=torch.ones_like(xfluid),create_graph = True,only_inputs=True)[0]
		v_xx = torch.autograd.grad(v_x,xfluid,grad_outputs=torch.ones_like(xfluid),create_graph = True,only_inputs=True)[0]
		v_y = torch.autograd.grad(v,yfluid,grad_outputs=torch.ones_like(yfluid),create_graph = True,only_inputs=True)[0]
		v_yy = torch.autograd.grad(v_y,yfluid,grad_outputs=torch.ones_like(yfluid),create_graph = True,only_inputs=True)[0]

		P_x = torch.autograd.grad(P,xfluid,grad_outputs=torch.ones_like(xfluid),create_graph = True,only_inputs=True)[0]
		P_y = torch.autograd.grad(P,yfluid,grad_outputs=torch.ones_like(yfluid),create_graph = True,only_inputs=True)[0]

		T_x = torch.autograd.grad(T,xfluid,grad_outputs=torch.ones_like(xfluid),create_graph = True,only_inputs=True)[0]
		T_xx = torch.autograd.grad(T_x,xfluid,grad_outputs=torch.ones_like(xfluid),create_graph = True,only_inputs=True)[0]
		T_y = torch.autograd.grad(T,yfluid,grad_outputs=torch.ones_like(yfluid),create_graph = True,only_inputs=True)[0]
		T_yy = torch.autograd.grad(T_y,yfluid,grad_outputs=torch.ones_like(yfluid),create_graph = True,only_inputs=True)[0]
		
		# loss_1 = u*(v_x/x_scale)+v*v_y - Diff*(v_xx/(x_scale**2)+v_yy)+1/rho*P_y #Y-dir
		# loss_2 = u*(u_x/x_scale)+v*u_y - Diff*(u_xx/(x_scale**2)+u_yy)+1/rho*(P_x/x_scale)#X-dir
		# loss_3 = (u_x/x_scale + v_y) #continuity
		loss_4 = u*(T_x/x_scale)+v*T_y - (K/rho/Cp)*(T_xx/(x_scale**2)+T_yy)         #energy
		
		
		# MSE LOSS
		loss_f = nn.MSELoss()
		

		
		loss = loss_f(loss_4,torch.zeros_like(loss_4))
		
		return loss
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
		
		loss_5 = ((kc_x/x_scale)*(Tc_x/x_scale)) + (kc_y*Tc_y) + (kc*(Tc_xx/(x_scale**2))) + (kc*Tc_yy) 
		# MSE LOSS
		loss_f = nn.MSELoss()
		

		loss_conduction = loss_f(loss_5,torch.zeros_like(loss_5))

		return loss_conduction
	###########################################################
	###################################################################
	def Loss_BC(xb,yb,xleft,yleft,xbase_fin,ybase_fin,xb_wall,yb_wall,xb_wall_fluid,yb_wall_fluid,xb_wall_fin, yb_wall_fin,xup_fluid, yup_fluid, xleft_fluid, yleft_fluid, xright_fluid, yright_fluid, xup_fin, yup_fin, xleft_fin, yleft_fin, xright_fin, yright_fin,ub,vb,Tb,xfluid,yfluid,xfin,yfin):
		if(0):
			xb = torch.FloatTensor(xb).to(device)
			yb = torch.FloatTensor(yb).to(device)
			ub = torch.FloatTensor(ub).to(device)
			vb = torch.FloatTensor(vb).to(device)
			xb_T = torch.FloatTensor(xb_T).to(device)
			yb_T = torch.FloatTensor(yb_T).to(device)
			Tb = torch.FloatTensor(Tb).to(device)
		
		yb_wall.requires_grad = True
		yup_fluid.requires_grad = True
		yup_fin.requires_grad = True
		xleft_fluid.requires_grad = True
		xleft_fin.requires_grad = True
		xright_fluid.requires_grad = True
		xright_fin.requires_grad = True
		
		
		net_in_left = torch.cat((xleft_fin, yleft_fin), 1)      
		net_in_right = torch.cat((xright_fin, yright_fin), 1)        
		net_in_up = torch.cat((xup_fin, yup_fin), 1)        
		kc_left = net2_kc(net_in_left)
		kc_left = kc_left.view(len(kc_left),-1)
		kc_right = net2_kc(net_in_right)
		kc_right = kc_right.view(len(kc_right),-1)
		kc_up = net2_kc(net_in_up)
		kc_up = kc_up.view(len(kc_up),-1)
		
		
		
		
		net_in1 = torch.cat((xb, yb), 1)           #velocity Bcs on all walls
		out1_u = net2_u(net_in1 )
		out1_v = net2_v(net_in1 )
		out1_u = out1_u.view(len(out1_u), -1)
		out1_v = out1_v.view(len(out1_v), -1)

		net_in2 = torch.cat((xleft, yleft), 1)      #Temp at the inlet
		out1_T = net2_T(net_in2 )
		out1_T = out1_T.view(len(out1_T), -1)

		net_in3 = torch.cat((xbase_fin, ybase_fin), 1)   #Temp on the fin base
		out1_Tc = net2_Tc(net_in3 )
		out1_Tc = out1_Tc.view(len(out1_Tc), -1)

		net_in6 = torch.cat((xb_wall,yb_wall),1)  #Temp on the top and bottom walls
		out2_T = net2_T(net_in6)
		out2_T = out2_T.view(len(out2_T), -1)

		net_in5 = torch.cat((xb_wall_fluid,yb_wall_fluid),1)   #Temperature on fluid interface
		out2_Tf = net2_T(net_in5)
		out2_Tf = out2_Tf.view(len(out2_Tf), -1)

		net_in4 = torch.cat((xb_wall_fin,yb_wall_fin),1)      #Temp on fin interface
		out3_Tc = net2_Tc(net_in4)
		out3_Tc = out3_Tc.view(len(out3_Tc), -1)
		
		net_in_up_fluid = torch.cat((xup_fluid,yup_fluid),1)        #Temp on top wall, fluid
		out_up_fluid = net2_T(net_in_up_fluid)
		out_up_fluid = out_up_fluid.view(len(out_up_fluid), -1)

		net_in_left_fluid = torch.cat((xleft_fluid,yleft_fluid),1)
		out_left_fluid = net2_T(net_in_left_fluid)
		out_left_fluid = out_left_fluid.view(len(out_left_fluid), -1)

		net_in_right_fluid = torch.cat((xright_fluid,yright_fluid),1)
		out_right_fluid = net2_T(net_in_right_fluid)
		out_right_fluid = out_right_fluid.view(len(out_right_fluid), -1)

		net_in_up_fin = torch.cat((xup_fin,yup_fin),1)
		out_up_fin = net2_Tc(net_in_up_fin)
		out_up_fin = out_up_fin.view(len(out_up_fin), -1)

		net_in_left_fin = torch.cat((xleft_fin,yleft_fin),1)
		out_left_fin = net2_Tc(net_in_left_fin)
		out_left_fin = out_left_fin.view(len(out_left_fin), -1)

		net_in_right_fin = torch.cat((xright_fin,yright_fin),1)
		out_right_fin = net2_Tc(net_in_right_fin)
		out_right_fin = out_right_fin.view(len(out_right_fin), -1)

		T_y_up_fluid = torch.autograd.grad(out_up_fluid,yup_fluid,grad_outputs=torch.ones_like(yup_fluid),create_graph = True,only_inputs=True)[0]
		T_y_up_fin = torch.autograd.grad(out_up_fin,yup_fin,grad_outputs=torch.ones_like(yup_fin),create_graph = True,only_inputs=True)[0]
		T_y_wall = torch.autograd.grad(out2_T,yb_wall,grad_outputs=torch.ones_like(yb_wall),create_graph = True,only_inputs=True)[0]
		
		T_x_left_fluid = torch.autograd.grad(out_left_fluid,xleft_fluid,grad_outputs=torch.ones_like(xleft_fluid),create_graph = True,only_inputs=True)[0]
		T_x_left_fin = torch.autograd.grad(out_left_fin,xleft_fin,grad_outputs=torch.ones_like(xleft_fin),create_graph = True,only_inputs=True)[0]
		
		T_x_right_fluid = torch.autograd.grad(out_right_fluid,xright_fluid,grad_outputs=torch.ones_like(xright_fluid),create_graph = True,only_inputs=True)[0]
		T_x_right_fin = torch.autograd.grad(out_right_fin,xright_fin,grad_outputs=torch.ones_like(xright_fin),create_graph = True,only_inputs=True)[0]
		
		T_up_fluid_coupling = -K* T_y_up_fluid
		T_left_fluid_coupling = -K* T_x_left_fluid
		T_right_fluid_coupling = -K* T_x_right_fluid
		T_up_fin_coupling = kc_up* T_y_up_fin
		T_left_fin_coupling = kc_left* T_x_left_fin
		T_right_fin_coupling = kc_right* T_x_right_fin
		 #coupling conditions
		coupling_bc = out3_Tc - out2_Tf
		coupling_up = T_up_fluid_coupling - T_up_fin_coupling
		coupling_left = T_left_fluid_coupling - T_left_fin_coupling
		coupling_right = T_right_fluid_coupling - T_right_fin_coupling

		loss_f = nn.MSELoss()
		loss_bc_fluid = loss_f(out1_T,torch.zeros_like(out1_T) )+ loss_f(T_y_wall,torch.zeros_like(T_y_wall))#+loss_f(out1_u, ub)+loss_f(out1_v,vb)#+ loss_f(T_y_up_fluid,torch.zeros_like(T_y_up_fluid)) +loss_f(T_x_left_fluid,torch.zeros_like(T_x_left_fluid))+loss_f(T_x_right_fluid,torch.zeros_like(T_x_right_fluid))+ loss_f(T_y_wall,torch.zeros_like(T_y_wall))#+loss_f(T_y_down,torch.zeros_like(T_y_down))
		
		loss_bc_fin =loss_f(out1_Tc,torch.ones_like(out1_Tc) )
		loss_coupling = loss_f(coupling_bc,torch.zeros_like(coupling_bc) )+ loss_f(coupling_up,torch.zeros_like(coupling_up) )+loss_f(coupling_left,torch.zeros_like(coupling_left) )+loss_f(coupling_right,torch.zeros_like(coupling_right) )
		
		return loss_bc_fluid + loss_coupling +loss_bc_fin
	##############################################

	def Loss_data(xd,yd,Td):
	

		
		net_in1 = torch.cat((xd, yd), 1)
		out1_T = net2_T(net_in1)
		
		
		out1_T = out1_T.view(len(out1_T), -1)
   
		loss_f = nn.MSELoss()
		loss_d = loss_f(out1_T, Td) 


		return loss_d   
	###############################################
	# Main loop
	tic = time.time()


	#load results
	if (Flag_pretrain):
		print('Reading previous results')
		net2_u.load_state_dict(torch.load(path+"fwd_step_u_velocity_from_fluent_10000"+".pt"))
		net2_v.load_state_dict(torch.load(path+"fwd_step_v_velocity_from_fluent_10000"+".pt"))
		net2_p.load_state_dict(torch.load(path+"fwd_step_p_velocity_from_fluent_10000"+".pt"))
		
	

		 
	# INSTANTIATE STEP LEARNING SCHEDULER CLASS
	if (Flag_schedule):
		scheduler_T = torch.optim.lr_scheduler.StepLR(optimizer_T, step_size=step_epoch, gamma=decay_rate)
		scheduler_Tc = torch.optim.lr_scheduler.StepLR(optimizer_Tc, step_size=step_epoch, gamma=decay_rate)
		scheduler_kc = torch.optim.lr_scheduler.StepLR(optimizer_kc, step_size=step_epoch, gamma=decay_rate)
		scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
		scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
		scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)
	if(Flag_batch):# This one uses dataloader

			for epoch in range(epochs):
				loss_bc_n = 0
				loss_eqn_fluid_n = 0
				loss_eqn_fin_n = 0
				loss_data_n = 0
				n = 0
				for batch_idx, (x_influid,y_influid, x_infin, y_infin) in enumerate(dataloader):
					net2_T.zero_grad()
					net2_Tc.zero_grad()
					net2_kc.zero_grad()
					net2_u.zero_grad()
					net2_v.zero_grad()
					net2_p.zero_grad()
					
					loss_eqn_fluid = criterion(x_influid,y_influid)
					loss_eqn_fin = criterion_2(x_infin,y_infin)
					loss_data = Loss_data(xd,yd,Td)
					loss_bc = Loss_BC(xb,yb,xleft,yleft,xbase_fin,ybase_fin,xb_wall,yb_wall,xb_wall_fluid,yb_wall_fluid,xb_wall_fin, yb_wall_fin,xup_fluid, yup_fluid, xleft_fluid, yleft_fluid, xright_fluid, yright_fluid, xup_fin, yup_fin, xleft_fin, yleft_fin, xright_fin, yright_fin,ub,vb,Tb,xfluid,yfluid,xfin,yfin)
			
					loss = loss_eqn_fluid + loss_eqn_fin + Lambda_BC* loss_bc + Lambda_data*loss_data
					loss.backward()
				
					optimizer_T.step()
					optimizer_Tc.step()
					optimizer_kc.step()
					optimizer_u.step() 
					optimizer_v.step() 
					optimizer_p.step()
					
					
					loss_eqn_fluid_a =loss_eqn_fluid.detach().cpu().numpy()
					loss_eqn_fluid_n += loss_eqn_fluid_a
					loss_eqn_fin_a =loss_eqn_fin.detach().cpu().numpy()
					loss_eqn_fin_n += loss_eqn_fin_a
					loss_bc_a= loss_bc.detach().cpu().numpy()
					loss_bc_n += loss_bc_a 
					loss_data_a= loss_data.detach().cpu().numpy()
					loss_data_n += loss_data_a
					n += 1         
					
					  
					if batch_idx % 40 ==0:
						print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f} Loss Eqn_fluid {:.10f} Loss Eqn_fin {:.10f} Loss BC {:.6f} Loss Data {:.6f}'.format(
							epoch, batch_idx * len(x_influid), len(dataloader.dataset),
							100. * batch_idx / len(dataloader), loss.item(), loss_eqn_fluid.item(), loss_eqn_fin.item(), loss_bc.item(), loss_data.item()))
					
				if (Flag_schedule):
						
						scheduler_T.step()
						scheduler_Tc.step()
						scheduler_kc.step()
						scheduler_u.step()
						scheduler_v.step()
						
						
				mean_eqn_fluid = loss_eqn_fluid_n/n
				mean_eqn_fin = loss_eqn_fin_n/n
				mean_bc = loss_bc_n/n
				mean_data = loss_data_n/n
				
				print('***Total avg Loss : Loss eqn_fluid {:.10f} Loss eqn_fin {:.10f} Loss BC {:.10f} Loss data {:.10f}'.format(mean_eqn_fluid, mean_eqn_fin, mean_bc, mean_data) )
				print('****Epoch:', epoch,'learning rate is: ', optimizer_T.param_groups[0]['lr'])
				
				
				
				if epoch % 1000 == 0:#save network
				 torch.save(net2_T.state_dict(),path+"inv_step_T_"+str(epoch)+".pt")
				 torch.save(net2_Tc.state_dict(),path+"inv_step_Tc_"+str(epoch)+".pt")
				 torch.save(net2_kc.state_dict(),path+"inv_step_kc_"+str(epoch)+".pt")
				 torch.save(net2_u.state_dict(),path+"inv_step_u_"+str(epoch)+".pt")
				 torch.save(net2_v.state_dict(),path+"inv_step_v_"+str(epoch)+".pt")
				 torch.save(net2_p.state_dict(),path+"inv_step_p_"+str(epoch)+".pt")
		  
		   
	   

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	###################
#evaluate models
	net2_T.eval()
	net2_Tc.eval()
	net2_kc.eval()
	net2_u.eval()
	net2_v.eval()
	net2_p.eval()
	
	net_in = torch.cat((xfluid.requires_grad_(),yfluid.requires_grad_()),1)
	net_in2 = torch.cat((xfin.requires_grad_(),yfin.requires_grad_()),1)
	output_T = net2_T(net_in)  #evaluate model
	output_Tc = net2_Tc(net_in2)  #evaluate model
	output_T = output_T.cpu().data.numpy()
	output_Tc = output_Tc.cpu().data.numpy()
	output_kc = net2_kc(net_in2)  #evaluate model
	output_kc = output_kc.cpu().data.numpy()
	xfluid = xfluid.cpu()
	yfluid = yfluid.cpu()
	xfin = xfin.cpu()
	yfin = yfin.cpu()




	return


#######################################################
#Main code:
# device = torch.device("cpu")
device = torch.device("cuda")

Flag_batch = True #USe batch or not  
Flag_pretrain = False # load previous resulta
Lambda_BC  = 30 # weight BC loss
Lambda_data = 10 # weight Data loss
Lambda_eqn = 1 # weight Data loss
batchsize = 512 #Total number of batches 
epochs  = 10001

Flag_schedule = True #If true change the learning rate at 3 levels
if (Flag_schedule):
	learning_rate = 3e-4
	step_epoch = 3300 
	decay_rate = 0.1

x_scale = 2.8 #The length of the  domain 
K = 0.02  #Fluid
Diff = 0.01
rho = 1 #1.
Cp =1.0
nPt = 200  
nPt1= 100
xStart_out = 0.4
xEnd_out = 1.0
xStart_up = 0.3
xEnd_up = 0.4
xStart_in = 0
xEnd_in = 0.3
yStart = 0.
yEnd = 1.0
yStart_up = 0.5

x = np.linspace(xStart_in, xEnd_in, nPt1)    #inlet
y = np.linspace(yStart, yEnd, nPt1)
x, y = np.meshgrid(x, y)
x = np.reshape(x, (np.size(x[:]),1))
y = np.reshape(y, (np.size(y[:]),1))

x2 = np.linspace(xStart_up  , xEnd_up, nPt1)      #upper
y2 = np.linspace(yStart_up , yEnd, nPt1,endpoint=False)
x2, y2 = np.meshgrid(x2, y2)
x2 = np.reshape(x2, (np.size(x2[:]),1))
y2 = np.reshape(y2, (np.size(y2[:]),1))

x3 = np.linspace(xStart_out  , xEnd_out, nPt1)     #outlet
y3 = np.linspace(yStart , yEnd, nPt,endpoint=False)
x3, y3 = np.meshgrid(x3, y3)
x3 = np.reshape(x3, (np.size(x3[:]),1))
y3 = np.reshape(y3, (np.size(y3[:]),1))

xfin = np.linspace(xEnd_in  , xStart_out, nPt)     #Solid
yfin = np.linspace(yStart , yStart_up, nPt,endpoint=False)
xfin, yfin = np.meshgrid(xfin, yfin)
xfin = np.reshape(xfin, (np.size(xfin[:]),1))
yfin = np.reshape(yfin, (np.size(yfin[:]),1))

xfluid = np.concatenate((x,x2,x3), axis=0)
yfluid = np.concatenate((y,y2,y3), axis=0)


print('shape of xfluid',xfluid.shape)
print('shape of yfluid',yfluid.shape)
print('shape of xfin',xfin.shape)
print('shape of yfin',yfin.shape)


#boundary pt and boundary condition

U_BC_in = 0.5
T_BC_in = 0.0
T_base = 1.0

#boundary conditions
nPt_BC = 2 *nPt
xleft = np.linspace(xStart_in, xStart_in, nPt_BC)
yleft = np.linspace(yStart, yEnd, nPt_BC)
xright = np.linspace(xEnd_out, xEnd_out, nPt_BC)
yright = np.linspace(yStart, yEnd, nPt_BC)
xup = np.linspace(xStart_in, xEnd_out, nPt_BC)
yup = np.linspace(yEnd, yEnd, nPt_BC)
xdown_1 = np.linspace(xStart_in, xEnd_in, nPt_BC)
ydown_1 = np.linspace(yStart, yStart, nPt_BC)
xdown_2 = np.linspace(xStart_out, xEnd_out, nPt_BC)
ydown_2 = np.linspace(yStart, yStart, nPt_BC)
xleft_fin = np.linspace(xEnd_in, xEnd_in, nPt_BC)
yleft_fin = np.linspace(yStart, yStart_up, nPt_BC)
xright_fin = np.linspace(xStart_out, xStart_out, nPt_BC)
yright_fin = np.linspace(yStart, yStart_up, nPt_BC)
xup_fin = np.linspace(xEnd_in, xStart_out, nPt_BC)
yup_fin = np.linspace(yStart_up, yStart_up, nPt_BC)
xbase_fin = np.linspace(xEnd_in, xStart_out, nPt_BC)
ybase_fin = np.linspace(yStart, yStart, nPt_BC)
xleft_fluid = np.linspace(xEnd_in, xEnd_in, nPt_BC)
yleft_fluid = np.linspace(yStart, yStart_up, nPt_BC)
xright_fluid = np.linspace(xStart_out, xStart_out, nPt_BC)
yright_fluid = np.linspace(yStart, yStart_up, nPt_BC)
xup_fluid = np.linspace(xEnd_in, xStart_out, nPt_BC)
yup_fluid = np.linspace(yStart_up, yStart_up, nPt_BC)
T_in_BC = np.linspace(T_BC_in, T_BC_in, nPt_BC)
T_base_BC = np.linspace(T_base, T_base, nPt_BC)
u_in_BC = U_BC_in*(yleft[:])*(1-yleft[:])/0.25
v_in_BC = np.linspace(0., 0., nPt_BC)
u_wall_BC = np.linspace(0., 0., nPt_BC)
v_wall_BC = np.linspace(0., 0., nPt_BC)
xb = np.concatenate((xleft, xup, xdown_1,xdown_2,xleft_fluid, xup_fluid, xright_fluid), 0)
yb = np.concatenate((yleft, yup, ydown_1,ydown_2,yleft_fluid, yup_fluid, yright_fluid), 0)
ub = np.concatenate((u_in_BC, u_wall_BC, u_wall_BC,u_wall_BC,u_wall_BC, u_wall_BC,u_wall_BC), 0)
vb = np.concatenate((v_in_BC, v_wall_BC, v_wall_BC,v_wall_BC,v_wall_BC, v_wall_BC,v_wall_BC), 0)
xb_T = np.concatenate((xleft,xbase_fin), 0)
yb_T = np.concatenate((yleft,ybase_fin), 0)
Tb = np.concatenate((T_in_BC, T_base_BC), 0)
xb_wall = np.concatenate((xdown_1,xdown_2,xup),0)
yb_wall = np.concatenate((ydown_1,ydown_2,yup),0)
xb_wall_fluid = np.concatenate((xleft_fluid, xup_fluid, xright_fluid),0)
yb_wall_fluid = np.concatenate((yleft_fluid, yup_fluid, yright_fluid),0)
xb_wall_fin = np.concatenate((xleft_fin, xup_fin, xright_fin),0)
yb_wall_fin = np.concatenate((yleft_fin, yup_fin, yright_fin),0)


xb= xb.reshape(-1, 1) #need to reshape to get 2D array
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
ub= ub.reshape(-1, 1) #need to reshape to get 2D array
vb= vb.reshape(-1, 1) #need to reshape to get 2D array

xb_T= xb_T.reshape(-1, 1) #need to reshape to get 2D array
yb_T= yb_T.reshape(-1, 1) #need to reshape to get 2D array
Tb= Tb.reshape(-1, 1) #need to reshape to get 2D array
xb_wall= xb_wall.reshape(-1, 1) #need to reshape to get 2D array
yb_wall= yb_wall.reshape(-1, 1)
xb_wall_fin= xb_wall_fin.reshape(-1, 1) #need to reshape to get 2D array
yb_wall_fin= yb_wall_fin.reshape(-1, 1)
xb_wall_fluid= xb_wall_fluid.reshape(-1, 1) #need to reshape to get 2D array
yb_wall_fluid= yb_wall_fluid.reshape(-1, 1)
xup_fin= xup_fin.reshape(-1, 1) #need to reshape to get 2D array
yup_fin= yup_fin.reshape(-1, 1)
xbase_fin= xbase_fin.reshape(-1, 1) #need to reshape to get 2D array
ybase_fin= ybase_fin.reshape(-1, 1)
xleft_fin= xleft_fin.reshape(-1, 1) #need to reshape to get 2D array
yleft_fin= yleft_fin.reshape(-1, 1)
xright_fin= xright_fin.reshape(-1, 1) #need to reshape to get 2D array
yright_fin= yright_fin.reshape(-1, 1)
xup_fluid= xup_fluid.reshape(-1, 1) #need to reshape to get 2D array
yup_fluid= yup_fluid.reshape(-1, 1)
xleft_fluid= xleft_fluid.reshape(-1, 1) #need to reshape to get 2D array
yleft_fluid= yleft_fluid.reshape(-1, 1)
xright_fluid= xright_fluid.reshape(-1, 1) #need to reshape to get 2D array
yright_fluid= yright_fluid.reshape(-1, 1)
xleft= xleft.reshape(-1, 1) #need to reshape to get 2D array
yleft= yleft.reshape(-1, 1)
print('shape of xb',xb.shape)
print('shape of yb',yb.shape)
print('shape of x_left',xleft.shape)
print('shape of yleft',yleft.shape)
print('shape of ub',ub.shape) 
print('shape of vb',vb.shape)
print('shape of xb_wall',xb_wall.shape)
print('shape of yb_wall',yb_wall.shape)
print('shape of xb_wall_fin',xb_wall_fin.shape)
print('shape of yb_wall_fin',yb_wall_fin.shape)
print('shape of xb_wall_fluid',xb_wall_fluid.shape)
print('shape of yb_wall_fluid',yb_wall_fluid.shape)

x_data = [0.826,0.83291,0.82588,0.83295,0.847,0.868,0.889,1.015,1.12,1.134,1.134,1.127]
y_data = [0.0071621,0.11678,0.37772,0.47133,0.514,0.514,0.507,0.507,0.507,0.40123,0.28851,0.0070328]
T_data = [0.88613,0.77616,0.61111,0.59538,0.53096,0.56122,0.60464,0.62735,0.60743,0.65435,0.69408,0.93972]

x_data = np.asarray(x_data)  #convert to numpy 
y_data = np.asarray(y_data) #convert to numpy 
T_data = np.asarray(T_data) #convert to numpy 

x_data = x_data/x_scale
print('Using input data pts: pts: ',x_data, y_data)
print('Using input data pts: Temp: ',T_data)
xd= x_data.reshape(-1, 1) #need to reshape to get 2D array
yd= y_data.reshape(-1, 1) #need to reshape to get 2D array
Td= T_data.reshape(-1, 1) #need to reshape to get 2D array

path = "Results/"


geo_train(device,xfluid,yfluid,xfin,yfin,xd,yd,Td,xb,yb,xleft,yleft,xbase_fin,ybase_fin,xb_wall,yb_wall,xb_wall_fluid,yb_wall_fluid,xb_wall_fin, yb_wall_fin,xup_fluid, yup_fluid, xleft_fluid, yleft_fluid, xright_fluid, yright_fluid, xup_fin, yup_fin, xleft_fin, yleft_fin, xright_fin, yright_fin,ub,vb,Tb,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Lambda_BC,nPt )
