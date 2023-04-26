Codes and data for the paper Aliakbari et al. "Ensemble physics informed neural networks: A framework to improve inverse transport modeling in heterogeneous domains", Physics of Fluids.

# ensemble-PINN-inverse
The ePINN approach utilizes an ensemble of parallel neural networks to tackle inverse problems. Each sub-network is initialized with a meaningful pattern of the unknown parameter, creating a foundation for a main neural network to be trained using PINN. In comparison, a traditional PINN simulation with random initialization was also employed to evaluate the convergence speed and accuracy of the two approaches.


##########################################################<br/>
Codes and data used in the test cases presented in the paper:<br/>
Ensemble physics informed neural networks: A framework to improve inverse transport modeling in heterogeneous domains.


##########################################################<br/>
Pytorch codes are included for the different test cases presented in the paper. Namely, 2D Multiphysics heat transfer in a fin, 2D diffusion, 2D porous medium transport, 2D flow in a stenosis.


##########################################################<br/>
Codes:<br/>
Codes for ePINN and PINN are provided.<br/>
Note: Same codes are used for ePINN with random initialization and without freezing layers in each sub-network. Set Flag_pretrain_initialization = False and Flag_Freezing_layer = False for random initialization and without freezing layers, respectively. The current code for ePINN involves initializing and freezing all layers in each sub-network, which is achieved by setting Flag_pretrain_initialization and Flag_Freezing_layer to True.


###########################################################<br/>
Data: <br/>
The input data for all test cases are provided in the Data folder. All .pt files are generated using a purely data driven deep neural network to map input coordinates to the low-fidelity CFD data. The .pt files were used to initialize ePINN.<br/>

###########################################################<br/>
Installation: <br/>
Install Pytorch: <br/>
https://pytorch.org/ <br/>
