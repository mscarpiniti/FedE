function out_acc = Test_model_ser(in_weight_vec)
% The service receives a weight vector: in_weight_vec and returns the
% corresponding test accuracy: out_acc under the currently trained model and associated
% test dataset. 
% The service is implemented as a MATLAB function, whose code can be customized by the user.
% The default implementation covers the SVM, MLP1 and MLP2 models under
% the parity-MNIST, FMNIST and TwS test datasets.
%
%---ONLY A SINGLE MODEL AND SINGLE TEST DATASET MUST BE SELECTED AT TIME---
%
%
persistent X
persistent Y


%--------------Case of SVM under the parity-MNIST test dataset-----------------
% Load the parity_MNIST test dataset
% load("MNIST_SVM_parity_Test.mat", "X", "Y");
% Convert data in double
%X = double(X);
%Y = double(Y);
%[test_feature_number, feature_length] = size(X);
%[test_feature_number1, class_number] = size(Y);

%if(test_feature_number ~= test_feature_number1)
   %error('Error message from Test_model_ser: under SVM, the row numbers of the matrices of the test features and corresponding labels do NOT match. The service call aborts');
%end

%if(class_number >= 2)
   %error('Error message from Test_model_ser: under SVM, the number of the columns of the loaded matrix of the test labels must be unit valued. The service call aborts');
%end
% Sanity check under SVM-----------------------------------
%[r,c]=size(in_weight_vec);
%if(r < c)
  % w = (in_weight_vec)';%w must be a column vector
%else
  % w = (in_weight_vec);
%end %End if

%if(length(w) ~= feature_length)
    %error('Error message from Test_model_ser: under SVM, the size of the loaded trained weight vector does NOT match the size of the loaded test features. The service call aborts');
%end
% Evaluate the SVM output --------------------------------------
%y_pred = predict_SVM(X, w);
% Compute the confusion matrix (CM) under SVM -----------------------------
%CM = confusionmat(Y, y_pred);
% Evaluate the returned accuracy under SVM
%out_acc = sum(diag(CM))/sum(CM(:));
%--------------------End SVM test------------------------------------------
%
%
%--------------Case of MLP1 under the FMNIST test dataset------------------
%Load the test dataset
load("../data/FMNIST_MLP_Test.mat", "X", "Y");
% Convert data in double
X = double(X);
Y = double(Y);
%
X = X/255;%Only under MLP1
%
[test_feature_number, feature_length] = size(X);
[test_feature_number1, class_number] = size(Y);

if(test_feature_number ~= test_feature_number1)
   error('Error message from Test_model_ser: under MLP1,the row numbers of the matrices of the test features and corresponding labels do NOT match. The service call aborts');
end

if(class_number  ~= 10)
   error('Error message from Test_model_ser: under MLP1, the number of the columns of the loaded matrix of the test labels does NOT match the expected one. The service call aborts');
end

% Sanity check under MLP1
[r,c] = size(in_weight_vec);
if(r < c)
   w = (in_weight_vec)'; %w must be a column vector
else
   w = (in_weight_vec);
end %End if

if(length(w) ~= 1462538)            
    error('Error message from Test_model_ser: under MLP1, the size of the loaded trained weight vector does NOT match its expected size. The service call aborts');
end
%
% Set the MLP1 hidden layers
H = 2*[512, 256, 128]; 

% Evaluate the MLP1 output -----------------------------------------------
y_pred = predict_MLP(X, Y, H, w);

% One-hot decode for evaluating the MLP1 accuracy--------------------------
n = size(Y, 2) - 1;
Y = onehotdecode(Y.', 0:n, 1).';
y_pred = onehotdecode(y_pred.', 0:n, 1).';
% Compute the confusion matrix (CM) under MLP1
CM = confusionmat(Y, y_pred);
% Evaluate the returned MLP1 accuracy
out_acc = sum(diag(CM))/sum(CM(:));
%--------------------End MLP1 test-----------------------------------------
%
%
%--------------Case of MLP2 under the TwS test dataset--------------------
% Load the test dataset
%load("TwS_MLP_Test_Balanced.mat", "X", "Y");
% Convert data in double
%X = double(X);
%Y = double(Y);
%[test_feature_number, feature_length] = size(X);
%[test_feature_number1, class_number] = size(Y);

%if(test_feature_number ~= test_feature_number1)
   %error('Error message from Test_model_ser: under MLP2, the row numbers of the matrices of the test features and corresponding labels do NOT match. The service call aborts');
%end

%if(class_number  ~= 3)
  % error('Error message from Test_model_ser: under MLP2, the number of the columns of the loaded matrix of the test labels does NOT match the expected one. The service call aborts');
%end
% Sanity check under MLP2--------------------------------------------
%[r,c] = size(in_weight_vec);
%if(r < c)
   %w = (in_weight_vec)';%w MUST be a column vector
%else
  % w = (in_weight_vec);
%end %End if

%if(length(w) ~= 32167)%MLP2_new  %14389 MLP2 by Michele
    %error('Error message from Test_model_ser: under MLP2, the size of the loaded trained weight vector does NOT match its expected size. The service call aborts');
%end
% Set the numbers of neurons per-hidden layer of MLP2
%H = 2*[12, 8, 50, 12, 50];%MLP2_new
%H = [12, 8, 50, 12, 50] Michele's original MLP2 version;
% Evaluate the MLP2 output --------------------------------------
%y_pred = predict_MLP(X, Y, H, w);
% One-hot decode for evaluating MLP2 test accuracy------------------------
%n = size(Y, 2) - 1;
%Y = onehotdecode(Y.', 0:n, 1).';
%y_pred = onehotdecode(y_pred.', 0:n, 1).';
% Compute the MLP2 confusion matrix (CM)
%CM = confusionmat(Y, y_pred);
% Evaluate the returned MLP2 accuracy
%out_acc = sum(diag(CM))/sum(CM(:));
%-----------------End MLP2 test--------------------------------------------
%
%
end %End service code