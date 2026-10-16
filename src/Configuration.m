% Script for the configuration of the considered architecture in terms of:
% - training and test sets
% - number of parameters
% - initialization of the parameter vector



% case 'SVM'
train_set = 'MNIST_SVM_parity_Train';
test_set  = 'MNIST_SVM_parity_Test';
%l = 28 * 28;    % l = 784
%w = 0.1*rand(l, 1) - 0.05;


% case 'MLP1'
%train_set = 'FMNIST_MLP_Train';
%test_set  = 'FMNIST_MLP_Test'; 
%H = [512, 256, 128];
%M = [28*28, H, 10];
%l = get_num_parameters(M);    % l = 567434;
%w = rand(l, 1) - 0.5;


% case 'MLP2'
%train_set = 'TwS_MLP_Train';
%test_set  = 'TwS_MLP_Test'; 
%H = [12, 8, 50, 12, 50];
%M = [1034, H, 3];
%l = get_num_parameters(M);    % l = 14389;
% w = rand(l, 1) - 0.5;
%w = init_weights(M);
		
		
		


%% Function used to retrieve the number of parameters of a neural network

function n_el = get_num_parameters(M)
% Evaluate the number of parameters of an MLP network
    n_el = 0;
    L = length(M) - 1;
    for i = 1:L
        n_el = n_el + (M(i)+1)*M(i+1);
    end
end


%% Function used to initialize the MLP2 weights

function ww = init_weights(M)
% Initialize the MLP weights and biases    
    ww = [];
    for j = 1:length(M)-1
        sig = sqrt(2/(M(j) + M(j+1)));
        ww = [ww; sig*randn(M(j)*M(j+1), 1); zeros(M(j+1), 1)];
    end    
end
