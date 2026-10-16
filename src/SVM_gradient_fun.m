function [DF, F, returnedID, MB] = SVM_gradient_fun(w, xb, db, inputID)

% Function dedicated to compute the gradient of the cost function and the
% cost function value.
% In this example we implement the linear SVM with Hinge training loss function.
%
% Usage: [DF, F, returnedID] = SVM_gradient_fun(w, xb, db, inputID)
% 
% where:
% - w: is the current weight vector;
% - xb: is the current feature (e.g., observation) vector;
% - db: is the SCALAR +/-1 desired label;
% - DF: is the computed gradient vector;
% - F: is the scalar loss function value. The default value is the Hinge loss
% -inputID: is the ID of the calling coworker
% - returnedID: echoes inputID
% - MB:minibatch_size

% July 2021 - M. Scarpiniti - DIET Department

%----------Initial check------------------------------------------------
[r,c] = size(w);
if (r < c)
   w = (w)'; %Convert the w input to a column vector
end

[r,c] = size(xb);
if (r < c)
   xb = (xb)'; %Convert the xb input to a column vector
end


[r,c] = size(db);
if (r < c)
   db = (db)'; %Convert the db input to a column vector
end
%-------------------------------------------------------------------------
returnedID = inputID;
%
[MB, l] = size(xb);
%
C = 0.001*((MB-1)/MB);
%Regularization parameter

y= (1/MB)*db*w'*xb;
%Scalar output of the linear binary SVM

e = 1-y;
%Scalar error

DF = ((1/MB)*C*w) -((1/MB)*sign(e)*db*xb); 
%DF is a COLUMN vector. It is the gradient vector of the regularized Hinge
%loss function

F = max(0,e);
%Current Hinge loss value
end %End function
