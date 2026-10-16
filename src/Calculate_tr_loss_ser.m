function [out_obfun_value, out_local_weight_mtx] = Calculate_tr_loss_ser(in_obfun_value_past, in_gl_primal_vec, in_it, in_rc_datapacket_mtx, in_updating_map_vec,...
                                                                        in_dual_vec_dim, in_local_weight_mtx)
%
% The service:
%
%  [out_obfun_value, out_local_weight_mtx] = Calculate_tr_loss_ser(in_obfun_value_past, in_gl_primal_vec, in_it, in_rc_datapacket_mtx,...
%                                                                 in_updating_map_vec, in_dual_vec_dim, in_local_weight_mtx)
%
% updates the current GLOBAL average value of the training loss function and the matrix of the per-coworker local weight vectors  
% at the Orchestrator on the basis of the updated LOCAL average values of
% the training loss functions currently returned by coworkers and the associated weight vectors.  
% The returned out_obfun_value value is
% a scalar, while the returned out_local_weight_mtx is a matrix of size: (coworker_number x length(in_gl_primal_vec)).
% The service is implementeed as a call to a MATLAB function.
% All the input vector parameters of the function are ROW vectors.
%
% in_obfun_value_past reports the last value of the global loss function
% available at the Orchestrator.
%
% in_gl_primal_vec is the most recent weight vector of the trained machine 
% that is available at the orchestrator.
%
%i n_updating_map_vec reports the indexes of the
% coworkers that have sent the latest (e.g., most recent) local updates to
% the orchestrator. This means, in turn, that the most recently updated rows of:
% in_rc_datapacket_mtx are the ones whose indexes correspond to the unit-valued
% entries of: in_updating_map_vec. 
%
% in_it is the current number of the global iteration carried out by the Orchestrator.
%
% The actual analytical form of the global loss function updating relationship 
% performed by the function: calculate_tr_loss_ser can be customized by the user on the basis of the 
% actually trained ML model.   
% By default, calculate_tr_loss_ser implements a moving average updating, in which the 
% enforcing term is an average of the rows of: in_rc_datapacket_mtx whose
% indexes correspond to the unit_valued entries of: in_updating_map_vec,
% e.g., an average of the most fresh local loss function values that the
% orchestrator has received from the coworkers. 
%
% By default, out_obfun_value reports:
% (i) the current average cross-entropy training loss under MLP models; 
% (ii) the current average Hinge training loss under SVM models; and, 
% (iii) the current average squared training error under the regressor
% model. 

[r, in_primal_vec_dim] = size(in_gl_primal_vec);
if (r ~= 1)
    error('The first input parameter passed to the Calculate_tr_loss_ser is NOT a ROW vector. The service call aborts');
end

%
cw_number = length(in_updating_map_vec);
[r,c] = size(in_rc_datapacket_mtx);
if((r ~= cw_number)||(c ~= (in_primal_vec_dim + in_dual_vec_dim + 5)))
   error('The size of the third input matrix parameter passed to the Calculate_tr_loss_ser is WRONG. The service call aborts');
end
%
%
%--------This part of the code can be CUSTOMIZED by the user on the basis
%--of the actual updating of the global loss function to be performed---
out_aux = zeros(1, 1);
out_local_weight_mtx = in_local_weight_mtx;
for j=1:cw_number
    out_local_weight_mtx(j,1:in_primal_vec_dim) = in_rc_datapacket_mtx(j,1:in_primal_vec_dim); 
    if (in_updating_map_vec(j) == 1) %The j-th coworker sent a fresh packet
       out_aux = out_aux + in_rc_datapacket_mtx(j,end-1); 
    end %End if
end %End for
%
out_aux = out_aux/max(1,sum(in_updating_map_vec));
%Compute the average of the accumulated fresh loss-function entries of in_rc_datapacket_mtx 
%while avoiding divisions by zero. 
%

mixer = 0.7;

if ((mixer >=1) || (mixer < 0))
    fprintf('Error message from the Calculate_tr_loss_ser\n');
    fprintf('The mixer constant must fall in the SEMI-open interval [0,1)\n');
    error('The service call aborts and the FedE run exits');
end
%
if (in_it == 1)
    out_obfun_value = out_aux;
else
   out_obfun_value = ((mixer*in_obfun_value_past) + ((1-mixer)*out_aux)); 
   %Perform a moving average updating of the local loss function values
   %returned by coworkers
end
%
end%End function
