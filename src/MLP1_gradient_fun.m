function [DF, F, returnedID, MB] = MLP1_gradient_fun(w, xb, db, inputID)

% Function dedicated to compute the gradient of the cost function and the
% cost function value.
% In this example, we implement the MLP with an arbitrary number of hidden 
% layers.
%
% Usage: [DF, F, H, MB] = MLP1_gradient(w, xb, db)
% 
% where:
% - w: is the current weight vector. 
%      It must be a COLUMN vector of size (primal_vec_length x 1); 
% - xb: is the new input batch. 
%       It must be a matrix of size (MB x feature_size);
% - db: is the matrix of the desired labels. It must be a matrix of size
%       (MB x number_of_classes);
% - DF: is the returned gradient vector. It is a COLUMN vector of size 
%       (primal_vec_length x 1);
% - F: is the SCALAR loss function value. The default setting is cross-entropy;
% - H: (optional) is the number of hidden neurons.
% - inputID: is the identifier of the calling coworker
% - returnedID: echoes inputID
% - MB: minibatch_size
% Decemnber 2023 - M. Scarpiniti, E.Baccarelli - DIET Department



H = 2*[512, 256, 128];  
% Number of per-hidden layer neurons
% of the MLP to be trained


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

% xb is a matrix of size (MB x feature_vec_size)
% db is a unit-vector of size (1 x number_of_classes)
x = xb';
d = db';

% Initialization ----------------------------------------------------------
Ni = size(x, 1);    % Number of inputs = size of each training feature vector
No = size(d, 1);    % Number of outputs = number of classes to be decided
M = [Ni, H, No];    % Neurons in layers
L = length(M) - 1;  % Number of layers barring the input one
MB = size(xb, 1);   % Mini-batch size == Number of rows of xb

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
DF = DF/MB;


% Scalar loss function------------------------------------------------------------
F = -sum((1/No)*sum(log(y+(10^-10)) .* d))/MB;%Cross entropy

end% End function 
