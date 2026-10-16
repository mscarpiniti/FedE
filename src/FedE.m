
%**************************************************************************                                                                   
%*                  Federated learning Emulator (FedE)                    *
%*                                  _                                     *
%*                                                                        *                
%*   A supporting SW platform for the parallel distributed emulation of   *   
%* Asynchronous, Synchronous and Partially Asynchronous Federated Learning*                                                            
%*                                                                        *
%*                                                                        *
%*                                                                        *                                                                                                                             *
%* Current version: 2.0                                                   *
%* Author: Enzo Baccarelli                                                *
%* Last updating: September, 2026                                         *         
%**************************************************************************
%
% The FedE SW toolkit emulates process-based
% distributed-memory parallel computing platforms for the client-server 
% model-based orchestration of Federated Learning (FL) 
% protocols/algorithms. The FedE main script relies on two main routines (namely, the 
% Orchestrator_routine and the CoWorker_routine) to manage the 
% inter-played actions of the central server and distributed parallel 
% coworkers of the emulated FL process. These routines allow the emulation of
% Asynchronous (A-FL), Synchronous (S-FL) and Semi-ASynchronous (SAS-FL)
% FL processes.The Orchestrator_routine and CoWorker_routine 
% are offered according to the  Platform-as-Service (PaaS) paradigm. 
%
clear; close all; fprintf('\n'); clc;
%
%--------------Load the traing/test datasets-------------------------------
load("../data/FMNIST_MLP_Train.mat","X","Y");
X = double(X);
% Load the full matrix of the training feature vectors stored row-wise and 
% casted into double
Y = double(Y);
% Full vector (resp., matrix) of the corresponding labels (resp., vectors
% stored row-wise) and casted into double
X = X/255;
%
classification_classes_number = 10; 
% Number of the decision classes of the adopted train/test dataset
%
%------------------Setup of the FedE input data----------------------------
%
coworker_number = 3;
% Number of coworkers that participate to the emulated FL process. 
%
if (mod(coworker_number,3) ~= 0 )
    fprintf('The input coworker_number must be an integer multiple by 3\n');
    fprintf('in order to emulate up to 3 classes of heterogeneous coworkers\n');
    error('The FedE script aborts');
end% End if
%
local_solver_flag = 1; 
% Binary-valued 0-1 flag that indicates the local solver used by each coworker 
% for locally updating BOTH his primal and dual local vector variables.
% local_solver_flag == 1 -> Adam is used as local solver
% local_solver_flag == 0 -> SGD is used as local solver
%
hidden_layer_number = 3;
% Number of layers of the trained machine.
%
average_neurons_number_per_hidden_layer= 298.67;%MLP2_new, 26.4 MLP2 by Michele
% Average number of neurons per each HIDDEN layer of the trained machine.
%
MB_size = 16;
% Size of the local mini-batches for EACH local iteration at the coworkers.
%
primal_vec_length = 1462538; 
% Number of weights of the trained MLP1_new
fprintf('Message from FedE: the trained ML model is MLP1 under the FMNIST data set and the cross-entropy training loss function\n');
%
dual_vec_length = 1;
% Length of the row vector storing the global and local dual variables.
% It must be NON_negative and integer-valued, e.g., dual_vec_length >= 0.
% In the case dual_vec_length == 0, dual variables are not present in the 
% implemented FL algorithm.  
% It must be the same value at the coworker-side.
%
dataset_distribution_type = 0;
% Binary 0/1-input that indicates the desired distribution of the training per-coworker local datasets
% dataset_distribution_type == 0 -> IID per-coworker training datasets
% dataset_distribution_type == 1 -> 1-NONIID per-coworker training datasets.
% The unit setting of dataset_distribution_type is allowed ONLY when the number 
% of classes to be classified is NOT larger than the number of available coworkers.
% A data-loading service is implemented by cw_primal_fun and invocated by
% the CoWorker_routine. It performs data loading according to the required IID/1-NONIID dataset
% distribution type. 
%
epsilon = 0.036;
% Per-weight vector maximum value that is allowed the imperfect consensus
% tolerance threshold. Perfect consensus is enforced for vanishing epsilon values. 
%
alpha = 0.01;    
% Forgetting factor used for updating the global weights' vector.
% alpha must fall in the SEMI-open interval [0,1).
% LARGER alphas' give rise to smoother behaviors of the resulting global objective function
% but also stretch the corresponding convergence times.
% The basic Fed-Averaging corresponds to the case of alpha == 0.
%
cw_max_iter_number = 2;
% Maximum number of consecutive primal-dual iterations that each coworker 
% can execute in local.
%
local_iter_setting_flag = 0;
% It is a 0-1-2-valued flag that specifies the strategy to be adopted for
% setting the number of consecutive local iterations to be carried out by
% each coworker.
% local_iter_setting_flag == 0 -> all coworkers always execute cw_max_iter_number consecutive local iterations;   
% local_iter_setting_flag == 1 -> all coworkers execute a number of consecutive 
% local iterations that is randomly sampled from the set
% {2,...,cw_max_iter_number};
% local_iter_setting_flag == 2 -> the number of consecutive 
% local iterations performed by each coworker after each global aggregation
% is adaptively tuned in the set: {1,...,cw_max_iter_number} on a coworker-
% basis according to the actual squared gaps between global and per-coworker
% local models.  
%
or_max_loop = 5; 
% Number of global aggregation rounds iterations performed by the Orchestrator_routine before
% halting.  
%
rc_TH = 1;
% Minimum number of D_packets that the orchestrator
% must receive at the be4ginning of each global round 
% from DIFFERENT coworkers before performing model aggregation. 
% It must be: 1 <= rc_TH <= (coworker_number).
% The setting: rc_TH == 1 corresponds to the Asynchronous FL (A-FL).
% The setting: 2 <= rc_TH <= (coworker_number-1) corresponds to a Semi-
% ASynchronous FL (SAS-FL).
% The setting: rc_TH == (coworker_number) corresponds to the Synchronous FL (S-FL). 
%
dynamic_cw_number_flag = 0;
% It is a 0/1 binary flag.
% dynamic_cw_number_flag == 0 means that the number of coworkers which
% participate to each global iteration round is held fixed as specified by the input parameter rc_TH.
% dynamic_cw_number_flag == 1 means that the number of coworkers which
% participate to each global iteration round is dynamically computed by the
% orchestrator at the beginning of each global iteration and, at each global
% iteration, falls into the set: {1, 2,..., coworker_number}. 
%
inter_perturbation_window = round(or_max_loop/4);
% It is the minimum number of global rounds performed by the orchestrator
% between two consecutive perturbation actions. It must be integer-valued
% and: inter_perturbation_window >= 1. 
% The setting: inter_perturbation_window = Inf,
% guarantees that perturbations will be NOT NEVER performed. This is the
% suitable setting when the adopted training loss function is convex,so
% that it does not present saddle points and/or local minima.
%
perturbation_TH = 3.5; 
% This is the threshold of the value of the Euclidean norm of the last
% available value of the objective function below which the perturbation
% condition is met. Perturbation_TH must be positive and, typically,  
% Larger values of perturbation_TH -> MORE perturbation actions are applied.
%                                                              
perturbation_strength = 0.038;
% Relative strength of the perturbation to be applied to each scalar entry of
% gl_primal_vec. It must be strictly positive.
% The perturbation magnitude increases for increasing values of 
% perturbation_strength. As rules of thumb, perturbation_strength should be 
% around 0.01 times the squared value of the maximum deep of the LOCAL holes of the 
% adopted training loss function, e.g., 0.01 times the squared value of the 
% maximum gap between a LOCAL minimum and the adjacent LOCAL maximum of the 
% adopted training loss function.      
%
net_idle_power_vec = 0.000001*[ones(1, (coworker_number/3)), ones(1,(coworker_number/3)), ones(1,(coworker_number/3))];
% Row vector of the per-coworker idle power (in (Watt)) consumed by the 
% network interfaces equipping the co-worker transceivers. Its entries may
% differ in the case of transceiver heterogeneity.
%
pr_overhead = 0.10;
% Per-D_packet overhead in the packet size induced by the used communication protocol stack. 
% pr_overhead >= 0.
%
mp_mdw_send_rec_delay_overhead_vec = (10^(-6))*[10*ones(1,(coworker_number/3)), 20*ones(1,(coworker_number/3)), 30*ones(1,(coworker_number/3))];
% Row vector of the per-coworker delay overheads (measured in seconds) that is
% induced by blocking send/receive operations in message passing-based Middleware
% communication. These delay overheads account for:
% (i) per-coworker Middleware-managed memory-CPU data transfer delays; 
% (ii) per-coworker Middleware-managed cache miss-induced data access delays;
% (iii) per-coworker send/receive buffering delays induced by Middleware-managed
% data packing/un-packing operations.  
% Its entries may be profiled on a per-coworker basis and they can 
% differ from coworker to coworker in the case of per-coworker Middleware
% heterogeneity.
% Typical values of these Middleware-induced communication delays range over
% the interval 10 (microsecond) - 50 (microsecond). 
%
q_bit_number = 64; 
% Number of bits used to quantize each scalar weight of the trained ML model.
% q_bit_number must be integer and falling in the set: {2, 3,..., 64}.
% By definition, the case: q_bit_number = 64 is understood as the UNQUANTIZED one,
% in which each weight of the trained ML model is coded as a 64-bit DOUBLE
% data.  
%
MTU_size = q_bit_number*2048; 
% Size (in (bit)) of the Maximum Transfer Unit (MTU) that is supported by
% the underlying Transport Layer. 
%
net_rate_vec = 1.6*0.75*4.5*78*(10^6)*[ones(1, (coworker_number/3)), 0.75*ones(1,(coworker_number/3)), 0.5*ones(1,(coworker_number/3))];
% Row vector of the per-coworker TX/RX rates (in (bit/sec)) of the network 
% interfaces equipping the coworker transceivers.
% Its entries can be different from coworker to coworker in the case of 
% coworker transceiver heterogeneity.
%
idle_comp_power_vec = (10^-1)*[ones(1, (coworker_number/3)), 0.75*ones(1,(coworker_number/3)), 0.50*ones(1,(coworker_number/3))]; 
% Row vector of the per-coworker idle components (measured in (Watt)) of the
% computing powers wasted by the CPU of each coworker. 
% It can be different from coworker to coworker in the case of coworker 
% CPU heterogeneity.
% It accounts also for the (possible) multi-core and/or multi-thread CPU architecture
% equipping each coworker device, as well as the (possibly) adopted
% virtualization/containerization technique. 
%
CPU_frequency_vec = (10^26)*[ones(1, (coworker_number/3)), 0.75*ones(1,(coworker_number/3)), 0.50*ones(1,(coworker_number/3))];
% Row vector of the per-coworker CPU frequencies (measured in (CPU cycles/sec)) 
% of the coworker CPU units equipping the coworker devices. 
% Its entries can differ from coworker to
% coworker in the case of CPU coworker heterogeneity.
% It accounts also for the (possible) multi-core and/or multi-thread CPU architecture
% equipping each coworker device, as well as the (possibly) adopted
% virtualization/containerization technique. 
%
dyn_comp_power_coeff_vec = (10^(-61))*[ones(1, (coworker_number/3)), 0.97*ones(1,(coworker_number/3)), 0.95*ones(1,(coworker_number/3))];
% Row vector of the scaling coefficients (measured in: (Watt/((Hz)^3)) ) of 
% the dynamic components of the CPU powers consumed by the coworker computing units. 
% Its entries can differ from coworker to coworker in the case of CPU coworker
% heterogeneity.
% It accounts also for the (possible) multi-core and/or multi-thread CPU architecture
% equipping each coworker device.
%
OPS_total_number = hidden_layer_number*(average_neurons_number_per_hidden_layer^2);
% Total number of Multiply-and-Accumulate scalar OPerationS (OPS) needed by the 
% considered ML model to process a single input instance through FORWARD propagation. 
%
per_MA_CPU_cycles = 0.31;
% Average number of CPU cycles required by each coworker device to carry out a SINGLE scalar
% Multiply-and-Accumulate (MA) operation. 
% Under per-coworker HW execution platforms equipped with vector ALUs and/or multi-kernel GPUs,   
% the value of per_MA_CPU_cycles may be less than the unit. 
%
per_ACT_CPU_cycles = 0.031*per_MA_CPU_cycles;
% Average number of CPU cycles required by each coworker device to carry out a SINGLE scalar
% nonlinear ACTivation (ACT) operation,like as, 
% for example, as a scalar ReLU and/or Dropout and/or MaxPool and/or batch-normalization 
% and/or softmax operation.
%
per_error_propagation_CPU_cycles = 0.031*per_MA_CPU_cycles;
% Average number of CPU cycles required by each coworker device to carry out
% a single back-propagation error operation during the training phase.
%
per_iteration_CPU_cycles = MB_size*((3*OPS_total_number*per_MA_CPU_cycles) + (hidden_layer_number*average_neurons_number_per_hidden_layer*(per_ACT_CPU_cycles + per_error_propagation_CPU_cycles))) + (primal_vec_length*per_MA_CPU_cycles) + (dual_vec_length*per_MA_CPU_cycles) + (local_solver_flag*5*primal_vec_length*per_MA_CPU_cycles) + (local_solver_flag*5*dual_vec_length*per_MA_CPU_cycles);
% Average number of CPU cycles needed to perform a SINGLE primal/dual vector training iteration at 
% the coworker side. It is equal for ALL coworkers and mainly scales up with
% the sizes of the local primal and dual vectors, as well as the minibatch size. 
% It also depends on the computational complexity of the adopted local
% solver,e.g., Adam or SGD.
% The reported formula of per_iteration_CPU_cycles is aligned with the
% so-called "DeLight" model developed by B.D. Rouhani et al. for the characterization of
% the average energy consumed by the back propagation-based training of 
% deep learning architectures.
%    
max_retry = 2; 
% It fixes the per-D_packet maximum number of consecutive transmission-PLUS-
% re-transmissions that is allowed each coworker before entering 
% the permanent failure state. When max_retry == 1, 
% re-transmissions of lost D_Packets are NOT allowed. 
%
BW_vec = 80*(10^6)*[ones(1, (coworker_number/3)), ones(1,(coworker_number/3)), ones(1,(coworker_number/3))];  
% Row vector of the per-coworker bandwidths (measured in (Hz)) of the 
% coworker-to-orchestrator up-links. Its scalar entries can be different in the
% case of per-coworker transceiver heterogeneity. 
%
path_loss_exponent= 2.1;  
% Path_loss_exponent (in linear scale) of each coworker-to-orchestrator up-link.
%
carrier_f = 4.45*(10^9);  
% Carrier frequency (in (Hz)) of each coworker-to-orchestrator up-link. 
%
N0 = 0.50*(10^(-16));
% Noise spectral density power (measured in (Watt/Hz)) of each 
% coworker-to-Orchestrator up-link.
%
TX_by_RX_antenna_gains_product_vec = 750*ones(1,coworker_number); % 750*ones(1,coworker_number); % 
% Row vector of the per-coworker products of the gains (in linear scale) of 
% the per-coworker TX-by-RX antennas of the
% coworker-to-orchestrator up-links. 
%
FL_environment_flag = 1; 
% It is a 0-1 binary flag.
% FL_environment_flag == 0 -> the simulated FL environment is wireless and of terrestrial type, 
% e.g, the Orchestrator and all coworkers are hosted on the ground. Specifically, the
% (possibly, mobile) coworkers are evenly distributed over a terrestrial
% annulus with inner (resp., outer) radius of d_min (resp., d_max), while the
% orchestrator stays fixed at the center of the annulus. Each
% coworker-to-orchestrator wireless up-link can be impaired by terrestrial path loss, 
% noise, fast-fading and packet loss phenomena, with the default fast-fading
% being of Rayleigh type.
%
% FL_environment_flag == 1 -> the simulated FL environment is wireless and
% of ground-air type, that is:
% (i) the (possibly mobile) coworkers are of terrestrial type and are evenly
% distributed over a disk-shaped region of radius COV_R, with COV_R > 0;
% (ii) the Orchestrator is hosted by a drone that flies at (fixed) altitude: 
% UAV_H and constant speed v along circular trajectory of radius: UAV_R (with 0 <= UAV_R <= COV_R).
% The orthogonal projection of the UAV trajectory onto the ground coincides with both
% the center of the terrestrial coverage area and the origin of the adopted
% Cartesian reference system. Each coworker-to-orchestrator (e.g., ground-to-air)
% up-link is affected by probabilistic (specifically, logistic type) path loss, noise and packet loss
% phenomena, with the average per-up link packet loss probability equal to
% the corresponding average per-up link Non-Line-of-Sight probability
% P_NLoS. 
%
switch FL_environment_flag
    case 0 %The simulated FL environment is of terrestrial type
         d_max = 400;  
         % Maximum allowed coworker-to-orchestrator distance (measured in
         % (m)) under the terrestrial scenario. 
         %
         d_min = 10;
         % Minimum allowed coworker-to-orchestrator distance (measured in
         % (m)) under the terrestrial scenario.
         % It must be: d_max >= d_min >= 0. The setting: d_max == d_min corresponds
         % to fixed (e.g., no mobile) terrestrial coworkers. 
         %
         packet_loss_probability_vec = 0.001*ones(1, coworker_number);  
         % Row vector of the per-coworker packet probabilities of the 
         % coworker-to-orchestrator up-links under the terrestrial scenario. 
         % Its entries can be different 
         % from coworker to coworker in the case of terrestrial link heterogeneity.
         % It must be: 0 <= packet_loss_probability <= 1.0 entry-wise. 
         %
         B = 0; C = 0; eta_LoS = 0; eta_NLoS = 0; UAV_H = 0; 
         % These last five input parameters are NOT used under terrestrial FL environments.
         % However, they must be passed as input parameters to the CoWorker_routine in all cases. 

    case 1 %The simulated FL environment is drone-supported
         COV_R = 380;
         % Radius (in (m)) of the disk-shaped radio coverage are on the ground.
         % It must be (strictly) positive and, typically, less than 400 (m).
         %
         UAV_R = 15; 
         % Radius (in (m)) of the drone circular trajectory in the air. 
         % It must be nonnegative, with UAV_R <= COV_R. 
         % The case of UAV_R == 0 corresponds to the case in which the drone 
         % hovers at fixed altitude: UAV_H aligned with the center of the
         % underlying ground coverage are.
         %
         UAV_H = 30;
         % Fixed (but adjustable) drone altitude (in (m)). 
         %
         B = 0.272;
         % Dimensionless B parameter of the adopted logistic path loss
         % model.
         % B = 0.272, 0.120, 0.095, and 0.048 under Sub-Urban, Urban, Dense
         % Urban and High Rise Urban UAV-supported scenarios.
         %
         C = 0.08; 
         % C parameter (in deg) of the adopted logistic path loss
         % model.
         % C = 0.08 (deg), 0.16 (deg), 0.23 (deg) and 0.11 (deg) under Sub-Urban, Urban, Dense
         % Urban and High Rise Urban UAV-supported scenarios.
         %
         eta_LoS = 0.1; 
         % Supplementary path loss (in (dB)) in the LoS-state under the
         % logistic path loss model.
         % eta_LoS = 0.1 (dB), 1.0 (dB), 1.6 (dB) and 2.3 (dB) under Sub-Urban, Urban, Dense
         % Urban and High Rise Urban UAV-supported scenarios.
         %
         eta_NLoS = 21; 
         % Supplementary path loss (in (dB)) in the NLoS-state under the
         % logistic path loss model.
         % eta_NLoS = 21 (dB), 20 (dB), 23 (dB) and 34 (dB) under Sub-Urban, Urban, Dense
         % Urban and High Rise Urban UAV-supported scenarios.
         %
         sample_number = 3000; 
         % Number of samples used by the ground_air_setting function to numerically 
         % profile the per-link average packet loss probability under the considered 
         % drone-supported scenario and according to the logistic probability
         % model. 
         %
         gamma_1 = 0.3*0.5*(504.60);                    %Parameters featuring
         % It is measured in (Watt)
         gamma_2 = 6.53*(10^(-6));                      %the propulsion-power
         % It is measured in ((m/sec)^-2)
         gamma_3 = 0.3*6.35*(10^(-4));                  %consumption profile
         % It is measured in (Watt x ((m/sec)^-3))
         gamma_4 = 0.3*0.5*(687.88);
         % It is measured in (Watt)                      %of the simulated 
         gamma_5 = 0.4*90.20;
         % It is measured in ((m/sec)^2)                 %Rotary-Wing (RW)-drone
         %
         UAV_angular_speed = (2*pi)/10.0;
         % Angular speed (in (rad/sec)) of the simulated RW-drone.
         % It must be nonnegative and vanishing when the simulated RW-drone
         % stays aloft in the air.
         %
         [d_max, d_min, av_ground_to_air_per_link_packet_loss_probability] = Ground_Air_Setting_fun(COV_R, UAV_R, UAV_H, B, C, sample_number);
         packet_loss_probability_vec = av_ground_to_air_per_link_packet_loss_probability*ones(1, coworker_number);
         % Row vector of the per-coworker packet loss probabilities of the 
         % coworker-to-orchestrator up-links under the drone-supported scenario. 
         % All its entries are equal. 
end
%End switch        
%
%
%--------------------End input data setting--------------------------------
%
warning('off','all');
%
%----Initializing the weight vector of the trained machine model-----------
load("../data/FedE_trained_MLP1_workspace.mat","init_primal_vec");
init_primal_vec = double(init_primal_vec);                                                    
%                                             
%---------------------End weights' initialization-------------------------
%
%----------------Display the summary of the input setting-----------------
if(dynamic_cw_number_flag == 0)
    switch rc_TH
    case 1
        fprintf('Message from FedE: the executed FL is of ASYNCHRONOUS type\n');
        fprintf('The number of coworkers partecipating to each global iteration is held FIXED to %d\n',1);
        case (coworker_number)
        fprintf('Message from FedE: the executed FL is of SYNCHRONOUS type\n');
        fprintf('The number of coworkers partecipating to each global iteration is held FIXED to %d\n', coworker_number);
    otherwise
        fprintf('Message from FedE: the executed FL is of SEMI-ASYNCHRONOUS type\n');
        fprintf('The number of coworkers partecipating to each global iteration is held FIXED to %d\n', rc_TH);
    end
elseif (dynamic_cw_number_flag == 1)
      fprintf('Message from FedE: the executed FL adaptively sets the number of partecipating coworkers on a per-global iteration basis over the set {%d,...,%d}\n', 1, coworker_number);
else
   error('Rrror message from FedE: the input adaptive_cw_number_flag parameter must be 0/1-valued. The FedE run aborts\n');
end 
% End if
%
switch dataset_distribution_type
    case 0
        fprintf('Message from FedE: the training per-coworker datasets are IID\n');
    case 1
      fprintf('Message from FedE: the training per-coworker datasets are 1-NONIID\n');
    otherwise
        error('Error message from FedE: the input value of data_distribution_type is NOT allowed. FedE run aborts\n');
end
%
switch local_solver_flag
    case 0
        fprintf('Message from FedE: the adopted local solver for updating both primal and dual local variables is SGD\n');
    case 1
      fprintf('Message from FedE: the adopted local solver for updating both primal and dual local variables is Adam\n');
    otherwise
        error('Error message from FedE: unrecognized value of local_solver_flag. FedE run aborts\n');
end
%
switch local_iter_setting_flag 
    case 0 % Default static strategy
         fprintf('Message from FedE: the number of consecutive local iterations performed by each coworker is held to be FIXED to %d\n', cw_max_iter_number);
    case 1 % Default time-varying strategy 
            fprintf('Message from FedE: the number of consecutive local iterations performed by each coworker is TIME-VARYING and lifted up/down over the set: {1,...,%d}\n', cw_max_iter_number);
    case 2 % Default per-coworker adaptive strategy
            fprintf('Message from set_iter: the number of local consecutive iterations performed by each coworker is ADAPTIVELY tuned on a per-coworker basis in the set: {1,...,%d} according to the actual squared gaps between global and per-coworker local models\n', cw_max_iter_number);
    otherwise
        error('Error message from FedE: the passed value of local_iter_setting_flag is invalid. FedE aborts');
end % End switch
%
q_bit_number = round(q_bit_number);
if((q_bit_number < 2)||(q_bit_number > 64))
   error('Error message from FedE: q_bit_number must be integer valued and falling in the set: {2, 3, ..., 64}. FedE run aborts');
elseif(q_bit_number == 64)
   fprintf('Message from fedE: the weights of the trained ML model are NOT quantized and are coded as 64-bit double data\n');
else
   fprintf('Message from FedE: each weight of the trained ML model is QUANTIZED with %d bits\n', q_bit_number);
   fprintf('The expected quantization-induced reduction in the volume ((bit) or (byte)) of the overall conveyed up/down network traffic is around: %g(%%) compared to the corresponding 64-bit double unquantized case\n', round(100*(1-(((q_bit_number*primal_vec_length) + (64*(dual_vec_length + 4)))/((64*primal_vec_length+dual_vec_length + 4)))),2));
end % End if
%
%
switch FL_environment_flag 
    case 0 % Terrestrial FL
        fprintf('Message from FedE: a TERRESTRIAL FL environment is emulated\n');
    case 1 % UAV-supported FL
         if ((B==0.272)&&(C==0.08)&&(eta_LoS ==0.1)&&(eta_NLoS==21))
            fprintf('Message from FedE: an UAV-supported  SUB-URBAN FL environment is emulated. The profiled average per-uplink packet loss probability of the emulated UAV-supported scenario is %g(%%)\n', round(av_ground_to_air_per_link_packet_loss_probability*100,2));
         elseif((B==0.120)&&(C==0.16)&&(eta_LoS == 1.0)&&(eta_NLoS==20))
            fprintf('Message from FedE: an UAV-supported URBAN FL environment is emulated. The profiled average per-uplink packet loss probability of the emulated UAV-supported scenario is %g(%%)\n', round(av_ground_to_air_per_link_packet_loss_probability*100,2));
         elseif((B==0.095)&&(C==0.23)&&(eta_LoS == 1.6)&&(eta_NLoS==23))
            fprintf('Message from FedE: an UAV-supported DENSE URBAN FL environment is emulated. The profiled average per-uplink packet loss probability of the emulated UAV-supported scenario is %g(%%)\n', round(av_ground_to_air_per_link_packet_loss_probability*100,2));
         elseif((B== 0.048)&&(C==0.11)&&(eta_LoS == 2.3)&&(eta_NLoS== 34))
            fprintf('Message from FedE: an UAV-supported HIGH RISE URBAN FL environment is emulated. The profiled average per-uplink packet loss probability of the emulated UAV-supported scenario is %g(%%)\n', round(av_ground_to_air_per_link_packet_loss_probability*100,2));
         else
            fprintf('Message from FedE: an UAV-supported FL environment is emulated with NON-standard probabilistic path-loss profile. The profiled packet loss probability is %g(%%)\n', round(av_ground_to_air_per_link_packet_loss_probability*100,2));
         end % End if
%
end % End switch
%
%
%-----------Turning-on multi-cores of the HW execution platform------------
p=gcp('nocreate'); % If a pool is already active, do not create a new one
if isempty(p)
    p = parpool('local',[1,coworker_number+1]);
    % Turn-ON (coworker_number+1) cores as processes
    poolsize = p.NumWorkers;
    fprintf('The number of actually turned-ON cores is %d\n', poolsize);
else
   delete(gcp('nocreate')); % Turn-OFF the current active pool
   p = parpool('local',[1,coworker_number+1]); 
   % Turn-ON a new pool of (coworker_number+1) maximum size as local processes
   poolsize = p.NumWorkers; % Actual size of the newly created pool
   fprintf('The number of actually turned-ON cores is %d\n', poolsize);
end %End if
%
% In the case in which the underlying execution platform would support
% containers, the above parpool() may be replaced by the activation of 
% (coworker_number+1) parallel containers. The remaining part of the code
% would remain unchanged, thus supporting code portability. 
%-----------------The executing parallel pool is ready---------------------
%--------------------------------------------------------------------------
%
%-----------------Sanity check on the input data---------------------------
coworker_number = max(1, round(coworker_number));
if (coworker_number > (poolsize-1))
    fprintf('The input coworker_number must be positive, integer-valued, multiple by 3,  and\n');
    fprintf('falling in the set: {1, 2,..., %d}\n', (poolsize-1));
    error('ERROR - The FedE script aborts');
end % End if
%
if((rc_TH < 1)||(rc_TH > (coworker_number)))
    fprintf('The input value of rc_TH is out of the allowed range: [1, %d]\n', (coworker_number));
    error('ERROR - The FedE script aborts');
end % End if
%
inter_perturbation_window = max(1, round(inter_perturbation_window));
%
if ((perturbation_TH < 0))
    fprintf('The input parameter: perturbation_TH must be non negative\n ');
    error('ERROR - The FedE script aborts');
end % End if
%
if ((perturbation_strength < 0))
    error('The input parameter: perturbation_strength must be non negative. ERROR - The FedE script aborts');
end % End if
%
max_retry= max(1, max_retry);
% max_retry must be at least unit-valued
%
if (d_min > d_max)
    aux = d_max;
    d_max = d_min;
    d_min = aux;
end % End if
% d_min cannot be strictly larger than d_max
%
if(packet_loss_probability_vec > 1.0)
    fprintf('All the entries of the packet loss probability vector must fall into the  interval [0,1]\n');
    error('ERROR - The FedE script aborts');
end % End if
%
%--------------------End input data sanity check---------------------------
%--------------------------------------------------------------------------
%
%-----------------Initialize the FedE l0cal variables---------------------
datapacket_length = primal_vec_length + dual_vec_length + 5;
% Fixed length (expressed as number of fields) of a D_packet.
%
D_packet_size =  MTU_size*ceil(((q_bit_number*primal_vec_length) + (64*(dual_vec_length + 5)))/MTU_size); 
% Size in (bit) of each (possibly quantized) D_packet.
% By design, each scalar weight (e.g., primal optimization variable) of 
% the trained ML model is quantized and coded as a q_bit_number-long data.
% However, all the remaining fields of each D_packet are leaved UNQUANTIZED and
% coded as 64-bit long double data.
% It accounts also for the packet-length overhead (possibly) induced by non-unit valued MTU_size.  
%-------------------End initialization of local variables-----------------
%--------------------------------------------------------------------------
%
%
%-----------------Calling the main MPMD construct--------------------------
ticBytes(gcp);
%
spmd(coworker_number + 1)
    tic
    if (spmdIndex == spmdSize)  % Call the Orchestrator_routine                            
        [cp_or_sent_Dpacket_count, cp_or_rc_Dpacket_count,... 
         cp_or_percoworker_number_of_processed_packets_vec, cp_or_H_packet_count,... 
         cp_or_P_packet_count, cp_or_C_packet_count, cp_or_gl_clock_vec,...
         cp_or_final_wall_clock_time, cp_or_primal_dual_gl_iteration_number,...
         cp_or_obfun_vec, cp_or_squared_primal_variables_vec,...
         cp_or_squared_dual_variables_vec, cp_or_pert_loop_ind_vec,... 
         cp_or_gl_max_ob_function, cp_or_gl_min_ob_function,... 
         cp_or_maximum_points_vec, cp_or_minimum_points_vec, cp_or_local_weight_mtx, cp_or_training_testaccuracy_vec, cp_best_testaccuracy_point] =...
         Orchestrator_routine(spmdIndex, primal_vec_length, dual_vec_length,...
                              cw_max_iter_number, or_max_loop, rc_TH,...
                              inter_perturbation_window, perturbation_TH,...
                              perturbation_strength, init_primal_vec, coworker_number, alpha, classification_classes_number,dynamic_cw_number_flag, q_bit_number); 
        %
    elseif ((spmdIndex >= 1) && (spmdIndex <= (spmdSize-1))) % Call the Coworker_routine
        [cp_cw_TX_total_count, cp_cw_sent_Dpacket_count, cp_cw_lost_packets_count,...
         cp_cw_rc_Dpacket_count, cp_cw_fictiosus_pk_count, cp_cw_ghost_packet_counter,...
         cp_cw_stamptimes_vec, cp_cw_final_local_clock,...
         cp_cw_comp_energy_tot, cp_cw_net_energy_tot,...
         cp_cw_net_plus_comp_energy_tot,...
         cp_cw_local_primal_iteration_number,...
         cp_cw_local_dual_iteration_number,...
         cp_cw_squared_norm_primal_variables_vec,...
         cp_cw_squared_norm_dual_variables_vec,...
         cp_cw_send_it_ID_vec, cp_cw_best_local_accuracy, cp_cw_final_local_accuracy, cp_cw_local_accuracy_vec] = CoWorker_routine(spmdIndex,...
                       primal_vec_length, dual_vec_length, net_idle_power_vec(spmdIndex),...
                       pr_overhead, D_packet_size, net_rate_vec(spmdIndex), idle_comp_power_vec(spmdIndex),...
                       CPU_frequency_vec(spmdIndex), dyn_comp_power_coeff_vec(spmdIndex),...
                       per_iteration_CPU_cycles, max_retry, d_max, d_min,...
                       BW_vec(spmdIndex), path_loss_exponent, carrier_f,...
                       N0, packet_loss_probability_vec(spmdIndex),...
                       TX_by_RX_antenna_gains_product_vec(spmdIndex), init_primal_vec, X, Y, MB_size, dataset_distribution_type,...
                       coworker_number, epsilon, q_bit_number, mp_mdw_send_rec_delay_overhead_vec, local_solver_flag, or_max_loop, local_iter_setting_flag,...
                       UAV_H, FL_environment_flag, B, C, eta_LoS, eta_NLoS);
    else
        fprintf('Labindex is out the expected range\n');
        error('ERROR - The FedE script aborts');
    end %End if
    spmdBarrier;  
    cp_aux = toc; 
end % End spmd construct
%
FedE_running_time = cp_aux{coworker_number+1};
% Store the execution time (in (sec)) of the performed FedE run
%
tocBytes(gcp);
%----------------------End main SPMD construct-----------------------------
%
%-----------------Mining and rendering FedE output-------------------------
%
%----BUILDING UP THE VECTOR OF THE TEST ACCURACY OBTAINED DURING THE TRAINING PHASE---
gl_trainedweight_vec = 100*double(cp_or_training_testaccuracy_vec{coworker_number+1}(1:end));
% The row vector: gl_trainedweight_vec must be of size (1 x(1 + or_max_loop)). 
% Its i-th scalar entry reports the test accuracy (in percent) of the trained model
% s computed by the Orchestrator during its i-th global iteration, including
% the initial one
%
local_best_accuracy_vec = zeros(1, coworker_number);
local_final_accuracy_vec = zeros(1,coworker_number);
length_cp_cw_local_accuracy_vec = zeros(1, coworker_number);
%
for k = 1:coworker_number
    local_best_accuracy_vec(1,k) = 100*double(cp_cw_best_local_accuracy{k});
    % It is a row vector of size (1 x coworker_number) whose k-th entry reports the best test accuracy
    % in percent) of the OVERALL sequence of LOCAL ML models trained by the k-th coworker 
    %
    local_final_accuracy_vec(1,k) = 100*(cp_cw_final_local_accuracy{k});
    % It is a row vector of size (1 x coworker_number) whose k-th entry reports the test accuracy
    % (in percent) of the FINAL LOCAL ML model trained by the k-th coworker
    length_cp_cw_local_accuracy_vec(k) = length(cp_cw_local_accuracy_vec{k}(1:end));
    % Save the lengths of the returned per-coworker local accuracy sequences
end
%
if ((1 ~= size(gl_trainedweight_vec,1) )||((1+or_max_loop) ~= size(gl_trainedweight_vec,2)))
    error('Error message from FedE: the per-link packet-loss probabilities are too high. Too much Coworker-to-Orchestrator links failed prematurely. FedE run aborts without data rendering');
end % End if
%
local_accuracy_mtx = zeros(coworker_number, max(length_cp_cw_local_accuracy_vec));

% Save and equalize the lengths of the returned
% per-coworker local accuracy sequences to (or_max_loop+1)
for k=1:coworker_number 
    local_accuracy_mtx(k, 1:length_cp_cw_local_accuracy_vec(k)) = double(cp_cw_local_accuracy_vec{k}(1:end));
end % End for
%
local_accuracy_mtx = 100*local_accuracy_mtx;
% Turning the returned local accuracy in per-cent
%
%-Rendering of the final trained machine weights and interpolate the returned local accuracy sequences--
final_trained_global_weight_vec = cp_best_testaccuracy_point{coworker_number+1}(1:primal_vec_length);
% This is a row vector of size: (1 x primal_vec_length). It reports the first 
% (of the possible multiple) points of GLOBAL maxima of the average
% training accuracy vector as evaluated by the orchestrator during the
% overall training phase.
% It is the vector of the globally trained ML weights to be used in the
% test phase of the trained machine when the target is the accuracy maximization
% of the underlying model. 
% 
final_trained_local_weight_mtx = double(cp_or_local_weight_mtx{coworker_number+1});
% This matrix is of size (coworker_number x primal_vec_length). Its k-th row
% reports the final trained machine weight (row) vector as locally computed
% by the k-th coworker.
%
gl_clock_vec_aux = double(cp_or_gl_clock_vec{coworker_number+1}(1:end));
% It must be a (1 x (or_max_loop+1)) row vector
%
training_wall_clock_time = cp_or_gl_clock_vec{coworker_number+1}(end);
% The length of each row of cw_training_testaccuracy_mtx equates the length 
% of the vector cp_or_gl_clock_vec{coworker_number+1}(1:end)  
% Dummy vectors/matrices used only for the rendering of the output data
%
% Interpolate the rows of local_accuracy_mtx over the time interval
% [0, training_wall_clock_time]
if(cp_or_gl_clock_vec{coworker_number+1}(end) == Inf)
       aux_vec = cp_or_gl_clock_vec{coworker_number+1}(1:end);
       aux_vec(end) = aux_vec(end-1) + 0.1;
       cp_or_gl_clock_vec{coworker_number+1} = aux_vec; %Avoid ill behavior
end %E nd if 
%
if ((cp_or_gl_clock_vec{coworker_number+1}(end)) > 0)
        aug_x_vec1 = 0.0: 0.05: cp_or_gl_clock_vec{coworker_number+1}(end); 
        % Augmented global clock vector
else
      aug_x_vec1 = 0.0;
end % End if
%
%-------------------------------------------------------------------------
%
fignumber = 0;
%
fignumber = fignumber+1;
figure(fignumber);
%
% Plot the behavior of the test accuracy obtained by the global ML model 
% during the training phase versus global clock sequence
if(cp_or_gl_clock_vec{coworker_number+1}(end) == Inf)
       aux_vec = cp_or_gl_clock_vec{coworker_number+1};
       aux_vec(end) = aux_vec(end-1) + 0.1;
       cp_or_gl_clock_vec{coworker_number+1}= aux_vec;%Avoid ill behavior
end % End if 

augmented_accuracy_vec = gl_trainedweight_vec; 


if ((cp_or_gl_clock_vec{coworker_number+1}(end)) > 0)
        aug_x_vec = 0.0: 0.05: cp_or_gl_clock_vec{coworker_number+1}(end); %Augmented global clock vector
        int_augmented_accuracy_vec= spline(cp_or_gl_clock_vec{coworker_number+1}, augmented_accuracy_vec, aug_x_vec);
else
      aug_x_vec = 0.0;
      int_augmented_accuracy_vec = augmented_accuracy_vec;
end % End if
%
plot(aug_x_vec, min(100,max(0,int_augmented_accuracy_vec)), 'r-');
xlabel('Wall clock time (sec)'); ylabel('Accuracy (%)');
grid on;
title('FedE - Test accuracy of the global ML model in the training phase-vs.-continuos-valued wall clock time');
hold on;
[best_acc, best_acc_round] = max(int_augmented_accuracy_vec);
fprintf('The best accuracy attained by the global ML model is %.2f (%%) at wall-clock time %.1f (sec)\n', best_acc, aug_x_vec(best_acc_round(1)));
plot(aug_x_vec(best_acc_round(1)), best_acc, 'k*');
legend({sprintf('Test accuracy of the global ML model\n'),sprintf('Best accuracy\n')},'Location','SouthEast');
hold off
%
fignumber = fignumber+1;
figure(fignumber);
% Plot the behaviors of the global training loss function, squared norms of global 
% primal variables and, if present, squared norms of global dual variables
% versus global clock sequence

if (dual_vec_length >=1 ) % Dual variables are used
    augmented_cp_or_obfun_vec{coworker_number+1} = cp_or_obfun_vec{coworker_number+1};                                       
    augmented_cp_or_squared_primal_variables_vec{coworker_number+1} = cp_or_squared_primal_variables_vec{coworker_number+1}; 
    augmented_cp_or_squared_dual_variables_vec{coworker_number+1} = cp_or_squared_dual_variables_vec{coworker_number+1};     
    augmented_or_obfun_vec_aux = augmented_cp_or_obfun_vec{coworker_number+1}; 
    augmented_or_squared_primal_variables_vec_aux = augmented_cp_or_squared_primal_variables_vec{coworker_number+1}; 
    augmented_or_squared_dual_variables_vec_aux = augmented_cp_or_squared_dual_variables_vec{coworker_number+1};
    %
    if(cp_or_gl_clock_vec{coworker_number+1}(end) == Inf)
       aux_vec = cp_or_gl_clock_vec{coworker_number+1};
       aux_vec(end) = aux_vec(end-1) + 0.1;
       cp_or_gl_clock_vec{coworker_number+1}= aux_vec; % Avoid ill behavior
    end  %End if 
    %
    if ((cp_or_gl_clock_vec{coworker_number+1}(end)) > 0)
        aug_x_vec = 0.0: 0.05: cp_or_gl_clock_vec{coworker_number+1}(end); % Augmented global clock vector
        int_augmented_cp_or_obfun_vec{coworker_number+1} = spline(cp_or_gl_clock_vec{coworker_number+1}, augmented_cp_or_obfun_vec{coworker_number+1},aug_x_vec);
        int_augmented_cp_or_squared_primal_variables_vec{coworker_number+1} = spline(cp_or_gl_clock_vec{coworker_number+1}, augmented_cp_or_squared_primal_variables_vec{coworker_number+1},aug_x_vec);
        int_augmented_cp_or_squared_dual_variables_vec{coworker_number+1} = spline(cp_or_gl_clock_vec{coworker_number+1}, augmented_cp_or_squared_dual_variables_vec{coworker_number+1},aug_x_vec);
    else
      aug_x_vec = 0.0;
      int_augmented_cp_or_obfun_vec{coworker_number+1} = augmented_cp_or_obfun_vec{coworker_number+1};
      int_augmented_cp_or_squared_primal_variables_vec{coworker_number+1} = augmented_cp_or_squared_primal_variables_vec{coworker_number+1};
      int_augmented_cp_or_squared_dual_variables_vec{coworker_number+1} = augmented_cp_or_squared_dual_variables_vec{coworker_number+1};
    end % End if
    %
    subplot(4,1,1)
    plot(aug_x_vec, max(round(cp_or_gl_min_ob_function{coworker_number+1},3), min(int_augmented_cp_or_obfun_vec{coworker_number+1},round(cp_or_gl_max_ob_function{coworker_number+1},3))), 'r-');
    or_gl_min_ob_function_aux = cp_or_gl_min_ob_function{coworker_number+1}; 
    int_augmented_or_obfun_vec_aux = int_augmented_cp_or_obfun_vec{coworker_number+1};
    or_gl_max_ob_function_aux = cp_or_gl_max_ob_function{coworker_number+1};
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Average training loss -vs.- continuos-valued wall clock time'); 
    legend({sprintf('Average training loss\n')},'Location','NorthEast');
    %
    subplot(4,1,2)
    plot(aug_x_vec,max(0,(1/primal_vec_length)*int_augmented_cp_or_squared_primal_variables_vec{coworker_number+1}),'g-');
    int_augmented_or_squared_primal_variables_vec_aux = int_augmented_cp_or_squared_primal_variables_vec{coworker_number+1};
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Per-entry average squared norm of the global model weights'' vector -vs.- continuos-valued wall clock time'); 
    legend({sprintf('Per-entry average squared norm of the global model weights'' vector\n')},'Location','NorthEast');
    %
    subplot(4,1,3)
    plot(aug_x_vec,max(0,(1/dual_vec_length)*int_augmented_cp_or_squared_dual_variables_vec{coworker_number+1}),'b-');
    int_augmented_or_squared_dual_variables_vec_aux = int_augmented_cp_or_squared_dual_variables_vec{coworker_number+1};
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Per-entry average squared norm of the global dual variables'' vector -vs.- continuos-valued wall clock time'); 
    legend({sprintf('Per-entry average squared norm of the global dual variables'' vector\n')},'Location','NorthEast');
    %
   subplot(4,1,4)
if(cp_or_gl_clock_vec{coworker_number+1}(end) == Inf)
       aux_vec = cp_or_gl_clock_vec{coworker_number+1};
       aux_vec(end) = aux_vec(end-1) + 0.1;
       cp_or_gl_clock_vec{coworker_number+1}= aux_vec;%Avoid ill behavior
end % End if 

augmented_accuracy_vec = gl_trainedweight_vec; 

if(cp_or_gl_clock_vec{coworker_number+1}(end) == Inf)
       aux_vec = cp_or_gl_clock_vec{coworker_number+1};
       aux_vec(end) = aux_vec(end-1) + 0.1;
       cp_or_gl_clock_vec{coworker_number+1}= aux_vec;%Avoid ill behavior
end % End if 

if ((cp_or_gl_clock_vec{coworker_number+1}(end)) > 0)
        aug_x_vec = 0.0: 0.05: cp_or_gl_clock_vec{coworker_number+1}(end); %Augmented global clock vector
        int_augmented_accuracy_vec= spline(cp_or_gl_clock_vec{coworker_number+1}, augmented_accuracy_vec, aug_x_vec);
else
      aug_x_vec = 0.0;
      int_augmented_accuracy_vec = augmented_accuracy_vec;
end % End if
%
plot(aug_x_vec, min(100,max(0,int_augmented_accuracy_vec)), 'r-');
xlabel('Wall clock time (sec)'); ylabel('Accuracy (%)');
grid on;
title('FedE - Test accuracy of the global model in the training phase  -vs.- continuos-valued wall clock time');
hold on
plot(aug_x_vec(best_acc_round(1)), best_acc, 'k*');
legend({sprintf('Test accuracy of the global model\n'), sprintf('Best accuracy\n')},'Location','SouthEast');
hold off
%
end % End if
%
if (dual_vec_length == 0) % Dual variables are NOT used
    augmented_cp_or_obfun_vec{coworker_number+1} = cp_or_obfun_vec{coworker_number+1};
    augmented_cp_or_squared_primal_variables_vec{coworker_number+1} = cp_or_squared_primal_variables_vec{coworker_number+1};
    augmented_or_obfun_vec_aux=augmented_cp_or_obfun_vec{coworker_number+1};
    augmented_or_squared_primal_variables_vec_aux =augmented_cp_or_squared_primal_variables_vec{coworker_number+1};
    %
    if(cp_or_gl_clock_vec{coworker_number+1}(end) == Inf)
       aux_vec = cp_or_gl_clock_vec{coworker_number+1};
       aux_vec(end) = aux_vec(end-1) + 0.1;
       cp_or_gl_clock_vec{coworker_number+1}= aux_vec;% Avoid ill behavior
    end % End if 
    %
    if ((cp_or_gl_clock_vec{coworker_number+1}(end)) > 0)
        aug_x_vec = 0.0: 0.05: cp_or_gl_clock_vec{coworker_number+1}(end); % Augmented global clock vector
        int_augmented_cp_or_obfun_vec{coworker_number+1}= spline(cp_or_gl_clock_vec{coworker_number+1}, augmented_cp_or_obfun_vec{coworker_number+1},aug_x_vec);
        int_augmented_cp_or_squared_primal_variables_vec{coworker_number+1}= spline(cp_or_gl_clock_vec{coworker_number+1}, augmented_cp_or_squared_primal_variables_vec{coworker_number+1},aug_x_vec);
    else
        aug_x_vec = 0.0;
        int_augmented_cp_or_obfun_vec{coworker_number+1} = augmented_cp_or_obfun_vec{coworker_number+1};
        int_augmented_cp_or_squared_primal_variables_vec{coworker_number+1} = augmented_cp_or_squared_primal_variables_vec{coworker_number+1};
    end % End if
    %
    subplot(3,1,1)
    plot(aug_x_vec,int_augmented_cp_or_obfun_vec{coworker_number+1},'r-');
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Average training loss -vs.- continuos-valued wall clock time'); 
    legend({sprintf('Average training loss\n')},'Location','NorthEast');
    %
    subplot(3,1,2)
    plot(aug_x_vec,max(0,(1/primal_vec_length)*int_augmented_cp_or_squared_primal_variables_vec{coworker_number+1}),'g-');
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Per-entry average squared norm of the global model weights'' vector -vs.- continuos-valued wall clock time'); 
    legend({sprintf('Per-entry average squared norm of the global model weights'' vector\n')},'Location','northeast');
    %
    subplot(3,1,3)
if(cp_or_gl_clock_vec{coworker_number+1}(end) == Inf)
       aux_vec = cp_or_gl_clock_vec{coworker_number+1};
       aux_vec(end) = aux_vec(end-1) + 0.1;
       cp_or_gl_clock_vec{coworker_number+1}= aux_vec; %Avoid ill behavior
end % End if 

augmented_accuracy_vec = gl_trainedweight_vec; 

if(cp_or_gl_clock_vec{coworker_number+1}(end) == Inf)
       aux_vec = cp_or_gl_clock_vec{coworker_number+1};
       aux_vec(end) = aux_vec(end-1) + 0.1;
       cp_or_gl_clock_vec{coworker_number+1}= aux_vec ;% Avoid ill behavior
end % End if 

if ((cp_or_gl_clock_vec{coworker_number+1}(end)) > 0)
        aug_x_vec = 0.0: 0.05: cp_or_gl_clock_vec{coworker_number+1}(end); % Augmented global clock vector
        int_augmented_accuracy_vec= spline(cp_or_gl_clock_vec{coworker_number+1}, augmented_accuracy_vec, aug_x_vec);
else
      aug_x_vec = 0.0;
      int_augmented_accuracy_vec = augmented_accuracy_vec;
end % End if
%
plot(aug_x_vec, min(100,max(0,int_augmented_accuracy_vec)), 'r-');
xlabel('Wall clock time (sec)'); ylabel('Accuracy (%)');
grid on;
title('FedE - Test accuracy of the global model in the training phase  -vs.- continuos-valued wall clock time'); 
legend({sprintf('Test accuracy of the global model\n')},'Location','SouthEast');
%
end % End if
%
%
% Bar plots of the best local accuracy obtained by the sequence of the local
% ML models trained by each coworker
fignumber = fignumber+1;
figure(fignumber);
b = bar(1:1:coworker_number,local_best_accuracy_vec, 0.02,'grouped','FaceColor','flat');
for j=1:length(local_best_accuracy_vec)
    b.CData(j,:)= [(j/(length(local_best_accuracy_vec))), 0.01*(j/(length(local_best_accuracy_vec))), 0.01*(j/(length(local_best_accuracy_vec)))];
    % Set the individual per-bar colors
end
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on;xlabel('Coworker index'); ylabel('Accuracy (%)');
title('FedE - Best test accuracy attained by the sequence of the local models trained by each coworker'); 
%
% Bar plots of the final local accuracy obtained by the final local
% ML models trained by each coworker
fignumber = fignumber+1;
figure(fignumber);
b = bar(1:1:coworker_number,local_final_accuracy_vec, 0.02,'grouped','FaceColor','flat');
for j=1:length(local_final_accuracy_vec)
    b.CData(j,:)= [(j/(length(local_final_accuracy_vec))), 0.01*(j/(length(local_final_accuracy_vec))), 0.01*(j/(length(local_final_accuracy_vec)))];
    % Set the individual per-bar colors
end
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on; xlabel('Coworker index'); ylabel('Accuracy (%)');
title('FedE - Test accuracy attained by the final local model trained by each coworker'); 
%
% Plot the behaviors of the squared norms of local 
% primal variables and, if present, squared norms of local dual variables
% versus the index of the performed local primal/dual iterations 
for j = coworker_number: -floor(coworker_number/2) :1
    cw_squared_norm_primal_variables_mtx_aux(j,1:length(cp_cw_squared_norm_primal_variables_vec{j})) = cp_cw_squared_norm_primal_variables_vec{j};
    cw_squared_norm_dual_variables_mtx_aux(j,1:length(cp_cw_squared_norm_dual_variables_vec{j})) = cp_cw_squared_norm_dual_variables_vec{j};
    fignumber = fignumber+1;
    figure(fignumber);
    if (dual_vec_length >=1 ) % Local dual variables are used
        if(length(cp_cw_squared_norm_primal_variables_vec{j}) > 1)
            aug_x_vec = 1.0: 0.05: length(cp_cw_squared_norm_primal_variables_vec{j}); % Augmented global clock vector
            int_cp_cw_squared_norm_primal_variables_vec{j} = spline(1:length(cp_cw_squared_norm_primal_variables_vec{j}), cp_cw_squared_norm_primal_variables_vec{j},aug_x_vec);
            int_cp_cw_squared_norm_dual_variables_vec{j} = spline(1:length(cp_cw_squared_norm_dual_variables_vec{j}), cp_cw_squared_norm_dual_variables_vec{j},aug_x_vec);
            int_cw_squared_norm_primal_variables_mtx_aux(j,1:length(int_cp_cw_squared_norm_primal_variables_vec{j})) = int_cp_cw_squared_norm_primal_variables_vec{j};
            int_cw_squared_norm_dual_variables_mtx_aux(j,1:length(int_cp_cw_squared_norm_dual_variables_vec{j})) = int_cp_cw_squared_norm_dual_variables_vec{j};
            
        else
            aug_x_vec = 1.0;
            int_cp_cw_squared_norm_primal_variables_vec{j} = cp_cw_squared_norm_primal_variables_vec{j};
            int_cp_cw_squared_norm_dual_variables_vec{j} = cp_cw_squared_norm_dual_variables_vec{j};
            int_cw_squared_norm_primal_variables_mtx_aux(j,1:length(int_cp_cw_squared_norm_primal_variables_vec{j})) = int_cp_cw_squared_norm_primal_variables_vec{j};
            int_cw_squared_norm_dual_variables_mtx_aux(j,1:length(int_cp_cw_squared_norm_dual_variables_vec{j})) = int_cp_cw_squared_norm_dual_variables_vec{j};
            
        end % End innermost if
        subplot(3,1,1) 
        aux_y_vec = linspace(0, training_wall_clock_time, length_cp_cw_local_accuracy_vec(j));
        int_x1_aux = 0.0: 0.05: training_wall_clock_time; % Augmented global clock vector
        int_y_aux_vec = max( 0 , min(100, spline(aux_y_vec,local_accuracy_mtx(j, 1:length_cp_cw_local_accuracy_vec(j)), int_x1_aux)));
        plot(int_x1_aux(1:end), int_y_aux_vec(1:end), 'r-');
        xlabel('Wall clock time (sec)'); ylabel('Accuracy(%)');
        grid on;
        title(sprintf('FedE - Test accuracy of the local model trained by Coworker n. %d -vs.- global wall clock time\n', j)); 
        legend({sprintf('Accuracy of the local models trained by Coworker n. %d\n', j)},'Location','SouthEast'); 

        subplot(3,1,2)
        plot(aug_x_vec, max(0,(1/primal_vec_length)*int_cp_cw_squared_norm_primal_variables_vec{j}),'g-');
        xlabel('Local iteration index'); ylabel('');
        grid on;
        title(sprintf('FedE - Interpolated behaviors of the per-entry average squared norm of the local model weights'' vector -vs.- the local integer-valued iteration index - Coworker n. %d\n', j)); 
        legend({sprintf('Per-entry average squared norm of the local model weights'' vector - Coworker n. %d\n', j)},'Location','NorthEast'); 
        %
        subplot(3,1,3)
        plot(aug_x_vec, max(0,(1/dual_vec_length)*int_cp_cw_squared_norm_dual_variables_vec{j}),'b-');
        xlabel('Local iteration index'); ylabel('');
        grid on;
        title(sprintf('FedE - Interpolated behaviors of the per-entry average squared norm of the local dual variables'' vector -vs.- the local integer-valued iteration index - Coworker n. %d\n', j)); 
        legend({sprintf('Per-entry average squared norm of the local dual variables'' vector - Coworker n. %d\n',j)},'Location','NorthEast');

    end % End if
    %
    if (dual_vec_length == 0 ) % Local dual variables are NOT used
        %
        if(length(cp_cw_squared_norm_primal_variables_vec{j}) > 1)
            aug_x_vec = 1.0: 0.05: length(cp_cw_squared_norm_primal_variables_vec{j}); % Augmented global clock vector
            int_cp_cw_squared_norm_primal_variables_vec{j}= spline(1:length(cp_cw_squared_norm_primal_variables_vec{j}), cp_cw_squared_norm_primal_variables_vec{j},aug_x_vec);
            int_cw_squared_norm_primal_variables_mtx_aux(j,:)=int_cp_cw_squared_norm_primal_variables_vec{j};         
        else
           aug_x_vec = 1.0;
           int_cp_cw_squared_norm_primal_variables_vec{j} = cp_cw_squared_norm_primal_variables_vec{j};
           int_cw_squared_norm_primal_variables_mtx_aux(j,:) = int_cp_cw_squared_norm_primal_variables_vec{j};
        end %End if
        %
        subplot(2,1,1)
        aux_y_vec = linspace(0, training_wall_clock_time, length_cp_cw_local_accuracy_vec(j));
        int_x1_aux = 0.0: 0.05: training_wall_clock_time; % Augmented global clock vector
        int_y_aux_vec = max( 0 , min(100, spline(aux_y_vec, local_accuracy_mtx(j, 1:length_cp_cw_local_accuracy_vec(j)), int_x1_aux)));
        plot(int_x1_aux(1:end), int_y_aux_vec(1:end), 'r-');
        xlabel('Wall clock time (sec)'); ylabel('Accuracy(%)');
        grid on;
        title(sprintf('FedE - Test accuracy of the local model trained by Coworker n. %d -vs.- global wall clock time\n', j)); 
        legend({sprintf('Accuracy of the local models trained by Coworker n. %d\n', j)},'Location','SouthEast'); 

        subplot(2,1,2)
        plot(aug_x_vec,max(0,(1/primal_vec_length)*int_cp_cw_squared_norm_primal_variables_vec{j}),'g-');
        xlabel('Local iteration index'); ylabel('');
        grid on;
        title(sprintf('FedE - Interpolated behaviors of the per-entry average squared norm of the local model weights'' vector -vs.- local integer-valued iteration index - Coworker n. %d\n', j)); 
        legend({sprintf('Per-entry average squared norm of the local model weights'' vector - Coworker n. %d\n', j)},'Location','NorthEast');
        %
        %
    end % End if
%
end % End for
%
% Plot the local clock sequences versus the corresponding local iteration index
% sequences
for j = coworker_number: -floor(coworker_number/2) : 1
    fignumber = fignumber +1;
    figure(fignumber);
    plot(cp_cw_send_it_ID_vec{j}, cp_cw_stamptimes_vec{j}, 'rh');
    cw_send_it_ID_mtx_aux(j,1:length(cp_cw_send_it_ID_vec{j})) = cp_cw_send_it_ID_vec{j};
    cw_stamptimes_mtx_aux(j,1:length(cp_cw_stamptimes_vec{j})) = cp_cw_stamptimes_vec{j};
    xlabel('Integer-valued local iteration index'); ylabel('Continuos-valued local clock (sec)');
    grid on;
    title(sprintf('FedE - Local clock values at which coworker n. %d succesfully sends packets to the Orchestrator -vs.- corresponding local iteration indexes - The highest clock entry is the FINAL LOCAL CLOCK TIME', j));  
end % End for
%
% Plot the global clock sequence versus the corresponding global iteration index
% sequence
fignumber = fignumber +1;
figure(fignumber);
plot(0:numel(cp_or_gl_clock_vec{coworker_number+1})-1, cp_or_gl_clock_vec{coworker_number+1}, 'rh');
xlabel('Integer-valued global iteration index'); ylabel('Continuos-valued global clock (sec)');
grid on;
title(sprintf('FedE - Global clock values at which the Orchestrator receives packets from the coworkers - vs.- global iteration indexes - The highest clock entry is the FINAL EMULATED WALL CLOCK TIME'));
%
% Bar plots of the per-coworker numbers of transmit, sent, lost and received
% data-packets
fignumber = fignumber +1;
figure(fignumber);
aux_mtx1 = zeros(coworker_number,4);
%
for j= 1: coworker_number
    aux_mtx1(j,:)= [cp_cw_TX_total_count{j}, cp_cw_sent_Dpacket_count{j}, cp_cw_rc_Dpacket_count{j},cp_cw_lost_packets_count{j}];
end %End for
%
b = bar(1:1:coworker_number,aux_mtx1,0.02,'stacked');
grid on; xlabel('Coworker index'); ylabel('Number of packets');
title('FedE - Per-coworker numbers of transmit, succesfully sent, lost and received data-packets'); 
legend({sprintf('# TX packets'),sprintf('# Succesfully sent packets'),sprintf('# RX packets'),sprintf('# Lost packets')},'Location','NorthEast'); 
%
%
% Bar plots of the per-coworker numbers of fictitious and ghost
% packets
fignumber = fignumber +1;
figure(fignumber);
aux_mtx2 = zeros(coworker_number,2);
%
for j= 1: coworker_number
    aux_mtx2(j,:)=[cp_cw_fictiosus_pk_count{j},cp_cw_ghost_packet_counter{j}];
end % End for
%
b = bar(1:1:coworker_number,aux_mtx2,0.02,'stacked');
xlabel('Coworker index'); ylabel('Number of packets');
grid on;
title('FedE - Per-coworker numbers of sent fictiousus and ghost packets'); 
legend({sprintf('# fictitious packets'),sprintf('# ghost packets')},'Location','NorthEast'); 
%
%
% Bar plots of the numbers of H, P, C control packets and TX-RX data-packets
% managed by the orchestrator
fignumber = fignumber + 1;
figure(fignumber);
aux1 = cp_or_H_packet_count{coworker_number+1};
aux2 = cp_or_P_packet_count{coworker_number+1};
aux3 = cp_or_C_packet_count{coworker_number+1};
aux4 = cp_or_sent_Dpacket_count{coworker_number+1};
aux5 = cp_or_rc_Dpacket_count{coworker_number+1}; 
%
X1 = categorical({'Halt-pk', 'Pause-pk','Clock-pk','Sent-pk','Received-pk'});
Y1 = [aux1, aux2, aux3, aux4, aux5];
b = bar(X1, Y1, 0.1,'grouped','FaceColor','flat');
%
b.CData(1,:)= [1,0,0]; % Red color
b.CData(2,:)=[0,0,1];  % Blu color
b.CData(3,:)= [0,1,0]; % Green color
b.CData(4,:)=[1,0,1];  % RB color
b.CData(5,:)= [0,1,1]; % GB color
% Set the colors of the two plotted bars
%
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
% Set the label at the tip of each plotted bar
xlabel(''); ylabel('Number of packets');
grid on;
title('FedE - Numbers of packets managed by the Orchestrator'); 
%
% Bar plots of the per-coworker numbers of D_packets actually processed by
% the orchestrator
aux_vec3=zeros(1,coworker_number);
aux_vec3=cp_or_percoworker_number_of_processed_packets_vec{coworker_number+1};
fignumber = fignumber+1;
figure(fignumber);
b = bar(1:1:coworker_number,aux_vec3,0.02,'grouped','FaceColor','flat');
for j=1:length(aux_vec3)
    b.CData(j,:)= [(j/(length(aux_vec3))), 0.01*(j/(length(aux_vec3))), 0.01*(j/(length(aux_vec3)))];
    % Set the individual per-bar colors
end
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on; xlabel('Coworker index'); ylabel('Number of processed packets');
title('FedE - Per-coworker numbers of data-packets processed by the Orchestrator'); 
%
% Bar plot of the propulsion energy consumed by the emulated drone
if((FL_environment_flag == 1)&&(training_wall_clock_time< Inf)) 
% The emulated FL environment is RW-drone supported and the FL training ended without failure
fignumber = fignumber+1;
figure(fignumber);
RW_UAV_consumed_propulsion_energy = compute_RW_UAW_propulsion_energy(gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, UAV_R, UAV_angular_speed, training_wall_clock_time);
% Propulsion energy (in (Joule)) wasted by the supporting drone during the FL training
X1 = categorical({'RW-drone propulsion energy'});
Y1 = round(RW_UAV_consumed_propulsion_energy*0.001,2);
b = bar(X1, Y1, 0.1,'FaceColor','flat');
%
b.CData(1,:)= [1,0,1];
% Set the color of the plotted bar
%
grid on; 
title('FedE - Propulsion energy (kiloJoule) wasted by the flying drone hosting the orchestrator'); 
ylabel('Propulsion energy (kJ)');
end % End if
%
% Bar plot of the propulsion power consumed by the emulated drone
if((FL_environment_flag == 1)&&(training_wall_clock_time < Inf)) 
    % The emulated FL environment is drone-supported and the FL training ended without failure
    fignumber = fignumber+1;
    figure(fignumber);
    RW_UAV_consumed_propulsion_power = RW_UAV_consumed_propulsion_energy/training_wall_clock_time;
    % Propulsion power (in (W)) wasted by the supporting drone during the FL training
    X1 = categorical({'RW-drone propulsion power'});
    Y1 = round(RW_UAV_consumed_propulsion_power,2);
    b = bar(X1, Y1, 0.1,'FaceColor','flat');
    %
    b.CData(1,:)= [0,0,1];
    % Set the color of the plotted bar
    grid on; 
    title('FedE - Propulsion Power'); 
    ylabel('Propulsion power (Watt)');
end % End if
%
% Bar plots of the per-coworker computing, network and total energies
% consumed over the overall FL training wall-clock duration
aux_mtx4 = zeros(coworker_number, 2);
%
for j=1:coworker_number
aux_mtx4(j,:) = [cp_cw_comp_energy_tot{j}, cp_cw_net_energy_tot{j}];
Total_energy_vec(j) = round(sum(aux_mtx4(j,:)),2);
end %End for
fignumber = fignumber+1;
figure(fignumber);
b = bar(1:1:coworker_number,aux_mtx4,0.02,'stacked');
xtips1 = b.XEndPoints;
ytips1 = max(b.YEndPoints);
labels1 = string(Total_energy_vec);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on; xlabel('Coworker index'); ylabel('Consumed energy (Joule)');
title('FedE - Per-coworker computing, network and total energies consumed over the overall FL training wall-clock duration'); 
legend({sprintf('Computing energy'),sprintf('Network energy')},'Location','NorthEast'); 
%
% Bar plots of the per-coworker computing, network and total consumed powers
% averaged over the overall FL training wall-clock duration
fignumber = fignumber+1;
figure(fignumber);
b = bar(1:1:coworker_number,round(1000*(aux_mtx4/(cp_or_gl_clock_vec{coworker_number+1}(end))),2), 0.02,'stacked');
xtips1 = b.XEndPoints;
ytips1 = max(b.YEndPoints);
final_clock = cp_or_gl_clock_vec{coworker_number+1}(end);
labels1 = string(round(1000*(Total_energy_vec/(final_clock)),2));
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on; xlabel('Coworker index'); ylabel('Average consumed power (mW)');
title('FedE - Per-coworker computing, network and total consumed powers averaged over the FL training wall-clock duration'); 
legend({sprintf('Average computing power'),sprintf('Average network power')},'Location','NorthEast'); 
%
% Plot of the sets of the global iteration indexes at which model 
% perturbations are applied
fignumber = fignumber+1;
figure(fignumber);
perturbation_index_vec = zeros(1, or_max_loop+1);
for j = 1:numel(cp_or_pert_loop_ind_vec{coworker_number+1})
    perturbation_index_vec(cp_or_pert_loop_ind_vec{coworker_number+1}(j)) = 1;
end
%
for j=1:1:length(perturbation_index_vec)
    if(perturbation_index_vec(j) == 1)
       plot(j, perturbation_index_vec(j), 'r*');
       hold on;
    else
       plot(j, perturbation_index_vec(j), 'r*');    
       hold on
    end % End if
end % End for
xlabel('Integer-valued global iteration index'); ylabel('ON/OFF perturbation');
grid on;
title('FedE - Binary plot of the global iteration indexes at which gradient perturbation is applied'); 
legend({sprintf('1 = Perturbation-ON'),sprintf('0 = Perturbation-OFF') },'Location','NorthEast'); 
%
% Bar plots of the returned maximum/minimum values of the global 
% training loss  
fignumber=fignumber+1;
figure(fignumber);
min_str = 'Minimum';
% In the case of multiple maximum points, only the
% (1xprimal_vec_length)-dimensional coordinates of the FIRST maximum point
% are displayed
max_str = 'Maximum';
% In the case of multiple minimum points, only the
% (1xprimal_vec_length)-dimensional coordinates of the FIRST minimum point
% re displayed
X1 = categorical({max_str, min_str});
Y1 = [round(cp_or_gl_max_ob_function{coworker_number+1},3), round(cp_or_gl_min_ob_function{coworker_number+1},3)];
or_gl_max_ob_function_aux = cp_or_gl_max_ob_function{coworker_number+1};
or_gl_min_ob_function_aux = cp_or_gl_min_ob_function{coworker_number+1};
b = bar(X1, Y1, 0.1,'grouped','FaceColor','flat');
%
b.CData(1,:)= [1,0,0]; % Red color
b.CData(2,:)=[0,1,0];  % Green color
% Set the colors of the two plotted bars
%
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
%S et the label at the tip of each plotted bar
grid on; 
ylabel('Maximum and minimum values of the average training loss');
title('FedE - Average training loss maximum and minimum'); 
%
%
%----------------------End mining/rendering section------------------------
%--------------------------------------------------------------------------
%-----------------------Results saving-------------------------------------
%
save("../data/FedE_trained_MLP1_workspace.mat");
fprintf('The final trained weights are returned row-wise by the variable: final_trained_global_weight_vec and permanently saved in the file: FedE_trained_MLP1_workspace.mat\n');
%
%--------------------------------------------------------------------------
fprintf('The FedE script ends succesfully\n');
%
%End of the FedE script
