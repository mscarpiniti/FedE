% Script for testing the SVM.
clear; close all; clc; fprintf('\n');
%
% Load the test dataset
load("../data/MNIST_SVM_parity_Test.mat", "X", "Y");

% Convert data in double
X = double(X);
Y = double(Y);

[test_feature_number, feature_length] = size(X);
[test_feature_number1, class_number] = size(Y);

if(test_feature_number ~= test_feature_number1)
   error('Error message from test_SVM: the row numbers of the matrices of the test features and corresponding labels do NOT match. Script aborts');
end

if(class_number >= 2)
   error('Error message from test_SVM: the number of the columns of the loaded matrix of the test labels must be unit valued. Script aborts');
end
%
% Load the best trained SVM weights
load("FedE_trained_SVM_workspace.mat", "final_trained_global_weight_vec");
final_trained_global_weight_vec = double(final_trained_global_weight_vec);
[r,c] = size(final_trained_global_weight_vec);
%
%-----------------Sanity check--------------------------------------------
if(r < c)
   w = (final_trained_global_weight_vec)';
else
   w = (final_trained_global_weight_vec);
end %End if

if(length(w) ~= feature_length)
    error('Error message from test_SVM: the size of the loaded trained weight vector does NOT match the size of the loaded test features. Script aborts');
end

%----------------------------------------------------------------------
%
%------------------Evaluate the SVM output --------------------------------
y_pred = predict_SVM(X, w);


%--------------Compute the confusion matrix (CM) --------------------------
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
save("../data/SVM_saved_test_performance.mat", "METRICS");
%
% Print all test metrics ------------------------------------------------
fprintf('Results for SVM\n');
fprintf('Accuracy: %5.2f\n', 100*acc);
fprintf('Macro precision: %5.2f\n', mean(pre));
fprintf('Macro recall: %5.2f\n', mean(rec));
fprintf('Macro F1-score: %5.2f\n', 2*mean(pre)*mean(rec)/(mean(pre)+mean(rec)));
fprintf('\n');
fprintf('PRECISION   RECALL   F1-SCORE\n');
fprintf('%7.3f %10.3f %9.3f\n', METRICS);
%
% Plot confusion matrix
cm = confusionchart(CM);
cm.Title = 'SVM';
