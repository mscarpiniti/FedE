%
% Script for testing the MLP1.
clear; close all; clc; fprintf('\n');
%
% Load the test dataset
load("../data/FMNIST_MLP_Test.mat", "X", "Y");

% Convert data in double
X = double(X);
Y = double(Y);
%
% If MLP1, normalize data to [0,1]
X = X/255;
%
[test_feature_number, feature_length] = size(X);
[test_feature_number1, class_number] = size(Y);
%
if(test_feature_number ~= test_feature_number1)
   error('Error message from test_MLP1: the row numbers of the matrices of the test features and corresponding labels do NOT match. Script aborts');
end

if(class_number  ~= 10)
   error('Error message from test_MLP1: the number of the columns of the loaded matrix of the test labels does NOT match the expected one. Script aborts');
end
%
% Load the already trained MLP1 weight vector
load("FedE_trained_MLP1_workspace.mat", "final_trained_global_weight_vec");
final_trained_global_weight_vec = double(final_trained_global_weight_vec);
[r,c] = size(final_trained_global_weight_vec);
%
%-----------------Sanity check--------------------------------------------
if(r < c)
   w = (final_trained_global_weight_vec)';%w MUST be a column vector
else
   w = (final_trained_global_weight_vec);
end %End if

if(length(w) ~= 1462538)       %(length(w) ~= 567434)
    error('Error message from test_MLP1: the size of the loaded trained weight vector does NOT match its expected size. Script aborts');
end
%---------------------------------------------------------------------------

% Set the MLP hidden layers
H = 2*[512, 256, 128];

% Evaluate the network output --------------------------------------
y_pred = predict_MLP(X, Y, H, w);

% One-hot decode for evaluating performance ------------------------
n = size(Y, 2) - 1;
Y = onehotdecode(Y.', 0:n, 1).';
y_pred = onehotdecode(y_pred.', 0:n, 1).';

% Compute the confusion matrix (CM)
CM = confusionmat(Y, y_pred);

% Evaluate the overall accuracy
acc = sum(diag(CM))/sum(CM(:));

% Evaluate the per-class precision
pre = diag(CM)./sum(CM, 1).';

% Evaluate the per-class recall
rec = diag(CM)./sum(CM, 2);

% Evaluate the per-class f1-score
f1_score = 2*pre.*rec ./ (pre + rec);

% Pack all metrics into a matrix
METRICS = [pre.'; rec.'; f1_score.'];
save("../data/MLP1_saved_test_performance.mat", "METRICS");

% Print all metrics ------------------------------------------------
fprintf('Results for MLP1\n');
fprintf('Accuracy: %5.2f\n', 100*acc);
fprintf('Macro precision: %5.2f\n', mean(pre));
fprintf('Macro recall: %5.2f\n', mean(rec));
fprintf('Macro F1-score: %5.2f\n', 2*mean(pre)*mean(rec)/(mean(pre)+mean(rec)));
fprintf('\n');
fprintf('PRECISION   RECALL   F1-SCORE\n');
fprintf('%7.3f %10.3f %9.3f\n', METRICS);

% Plot confusion matrix
labels = 0:9;
cm = confusionchart(CM, labels);
cm.Title = 'MLP1';

