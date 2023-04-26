# ensemble-PINN-inverse
Using ePINN approach to solve inverse problems where an ensemble of parallel neural networks is used, and each sub-network is initialized with a meaningful pattern of the unknown parameter. Subsequently, these parallel networks provide a basis that is fed into a main neural network that is trained using PINN. Additionally, a traditional PINN simulation with random initialization was used to compare the convergence speed and accuracy of the two PINN approaches.


##########################################################<br/>
Codes and data used in the test cases presented in the paper:<br/>
Ensemble physics informed neural networks: A framework to improve inverse transport modeling in heterogeneous domains.


##########################################################<br/>
Pytorch codes are included for the different test cases presented in the paper. Namely, 2D Multiphysics heat transfer in a fin, 2D diffusion, 2D porous medium transport, 2D flow in a stenosis.


##########################################################<br/>
Codes:<br/>
Codes for ePINN and PINN are provided.<br/>
Note: Same codes are used for ePINN with random initialization and without freezing layers in each sub-network. Set Flag_pretrain_initialization = False and Flag_Freezing_layer = False for random initialization and without freezing layers, respectively. (For ePINN, current codes are for initialization and freezing all layers in each sub_network. where Flag_pretrain_initialization = True and Flag_Freezing_layer = True).


###########################################################<br/>
Data: <br/>
The input data for all test cases are provided in the Data folder. All .pt files are generated using a purely data driven deep neural network to map input coordinates to the low-fidelity CFD data. The .pt files were used to initialize ePINN.<br/>

###########################################################<br/>
Installation: <br/>
Install Pytorch: <br/>
https://pytorch.org/ <br/>
