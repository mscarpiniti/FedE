% This function evaluate the output of an MLP.
%
% USAGE: y_pred = predict_MLP(X_test, Y_test, H, w)
%
% where:
%  - X_test: is the test set feature matrix (N_examples x N_features)
%  - Y_test: is the test set target matrix (N_examples x N_classes)
%  - H: is a row vector collecting the numbers of neurons in each hidden layers (H = [512, 256,
%  128], ecc.)
%  - w: is the trained COLUMN weight vector of the MLP (N_parameters x 1)

function y_pred = predict_MLP(X_test, Y_test, H, w)
% Evaluate the classifier output vector of size (No x 1)

Ni = size(X_test, 2);    % Number of inputs = size of the test input features
No = size(Y_test, 2);    % Number of outputs = number of classes to be detected
M = [Ni, H, No];         % Numbers of neurons in the  input, hidden and output layers
L = length(M) - 1;       % Number of output-plus-hidden layers

W = cell(L, 1);
b = cell(L, 1);
g = cell(L+1, 1);

pt = 1;
for i = 1:L
    W{i} = reshape(w(pt:pt+M(i+1)*M(i)-1), M(i+1), M(i));
    b{i} = w(pt+M(i+1)*M(i):pt+M(i+1)*M(i)+M(i+1)-1);
    pt = pt + M(i+1)*M(i) + M(i+1);
end

g{1} = X_test.';
for i = 1:L
    g{i+1} = max(W{i} * g{i} + b{i}, 0);  % ReLU
    % g{i+1} = 1./(1 + exp(-(W{i} * g{i} + b{i})));  % Sigmoid
end
g{L+1} = W{L} * g{L} + b{L};

y_pred = softmax(g{L+1});
y_pred(y_pred < max(y_pred, [], 1)) = 0;
y_pred(y_pred == max(y_pred, [], 1)) = 1;

y_pred = y_pred.';

end %End function
        
