function [DF, F, returnedID, MB] = MLP2_gradient_fun(w, xb, db, inputID)

% Function dedicated to compute the gradient of the cost function and the
% cost fucntion value.
% In this example we implement the MLP with an arbitrary number of hidden 
% layers.
%
% Usage: [DF, F, H, MB] = MLP2_gradient(w, xb, db)
% 
% where:
% - w: is the current weight vector.
%      It must be a COLUMN vector of size (primal_vec_length x 1);
% - xb: is the new input batch.
%       It must be a matrix of size (MB x feature_size);
% - db: is the new label matrix.
%       It must be a matrix of size (MB x number_of_classes). Each row
%       must be an UNIT vector. The i-th row of db is the one-hot encoded
%       label associated to the feature in the i-th row ogf xb;
% - DF: is the returned gradient vector. It is a COLUMN vector of size
%       (primal_vec_length x 1);
% - F: is the returned SCALAR loss fucntion value. The default setting is regularized cross-entropy;
% - H: (optional) is the number of hidden neurons.
% - inputID: is the identifier of the calling coworker
% - returnedID: echoes inputID 
% - MB: minibatch_size
% December 2023 - M. Scarpiniti, E. Baccarelli - DIET Department
%
C = 0.00001;
%Regularizing coefficient of the adopted cross-entropy training loss function 
%
H = 2*[12, 8, 50, 12, 50];  
% Number of per-hidden layer neurons of MLP2

%----------Initial check------------------------------------------------
[r,c] = size(w);
if (r < c)
   w = (w)'; %Convert the w input to a column vector
end

%-------------------------------------------------------------------------
returnedID = inputID;
%-------------------------------------------------------------------------


persistent W b
persistent h dh Wgrad bgrad


x = xb';
d = db';

% Initialization ----------------------------------------------------------
Ni = size(x, 1);    % Number of inputs = size of each training input feature
No = size(d, 1);    % Number of outputs = number of classes to be decided
M = [Ni, H, No];    % Neurons in layers
L = length(M) - 1;  % Number of layers barring the input one
MB = size(xb, 1);   % Mini-batch size

if isempty(W)
    W = cell(L, 1);
    b = cell(L, 1);

    Wgrad = cell(L, 1);
    bgrad = cell(L, 1);

    h  = cell(L+1, 1);
    dh = cell(L+1, 1);
end


% Pack vector to matrices -------------------------------------------------
pt = 1;
for i = 1:L
   W{i} = reshape(w(pt:pt+M(i+1)*M(i)-1), M(i+1), M(i));
   b{i} = w(pt+M(i+1)*M(i):pt+M(i+1)*M(i)+M(i+1)-1);
   pt = pt + M(i+1)*M(i) + M(i+1);
end

% Forward phase -----------------------------------------------------------
h{1} = x;  % Input
for i = 1:L-1
    h{i+1} = max(W{i} * h{i} + b{i}, 0);  % ReLU
    % h{i+1} = 1./(1 + exp(-(W{i} * h{i} + b{i})));  % Sigmoid
end
h{L+1} = W{L} * h{L} + b{L};
y = softmax(h{L+1});

% Back-Propagation algorithm ----------------------------------------------
dh{L+1} = y - d;
for i = L:-1:1
    Wgrad{i} = dh{i+1} * h{i}.';
    bgrad{i} = sum(dh{i+1}, 2);
    dh{i} = (W{i}.' * dh{i+1}) .* sign(h{i});  % ReLU
    % dh{i} = (W{i}.' * dh{i+1}) .* (h{i} .* (1 - h{i}));  % Sigmoid
end

% Unpack to column-wise gradient vector -----------------------------------
DF = [];
for i = 1:L
    DF = [DF; Wgrad{i}(:); bgrad{i}(:)];
end
DF = (DF/MB)+ (C*(w/length(w)))/MB; %OK
%Regularized DF

% Regularized cross-entropy training function------------------------------------------------------------
F = (-sum((1/No)*sum(log(y+(10^-8)) .* d))/MB)+(C*(1/MB)*((0.5*((w')*w)/(length(w)))));%OK

end% End function 
