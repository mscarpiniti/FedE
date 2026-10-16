% This function evaluate the output of a SVM.
%
% USAGE: y_pred = predict_SVM(X_test, w)
%
% where:
%  - X_test: is the test set feature matrix (N_examples x N_features)
%  - w: is the optimal weight vector of SVM (N_features x 1)

function y_pred = predict_SVM(X_test, w)
% Evaluate the classifier output
y_pred = sign(X_test * w);

% Remove possible but very limited zeros
y_pred(y_pred == 0) = 1;

end % End function