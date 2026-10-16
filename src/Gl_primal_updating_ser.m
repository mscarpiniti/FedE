function out_gl_primal_vec = Gl_primal_updating_ser(in_gl_primal_vec, in_gl_dual_vec, in_rc_datapacket_mtx, in_updating_map_vec, alpha)
%
% The service:
%
% out_gl_primal_vec = Gl_primal_updating_ser(in_gl_primal_vec, in_gl_dual_vec,in_rc_datapacket_mtx, in_updating_map_vec, alpha)
%
% updates the vector of the global primal variables at the orchestrator on
% the basis of the updated local primal and dual vectors that have been
% sent by the coworkers.
% The service is implemented as a MATLAB function.
% All the input and output vector parameters defining the service API are row vectors.
%
% in_updating_map_vec reports the indexes of the
% coworkers that have sent the latest (e.g., most recent) local updates to
% the orchestrator. 
%
% The actual analytical form of the updating relationship performed by this service 
% can be customized by the user on the basis of the actually implemented FL scheme.   
% By default, the service implements a moving average updating, in which the
% current global model is updated as a convex linear combination of the
% previous global model and the age-weighted average of the last 
% (e.g., most fresh) local model weights that the orchestrator has 
% received from the coworkers. 
%
[r, in_primal_vec_dim] = size(in_gl_primal_vec);
if (r ~= 1)
    error('The first input parameter passed to Gl_primal_updating_ser is not a row vector. The service call aborts');
end
%
if (~isempty(in_gl_dual_vec))
    [r, in_dual_vec_dim] = size(in_gl_dual_vec);
    if (r ~= 1)
    error('The second input parameter passed to Gl_primal_updating_ser is not a row vector. The sedrvice call aborts');
    end %End innermost if
else
   in_dual_vec_dim = 0; 
end %End outermost if
%
cw_number = length(in_updating_map_vec);
[r,c] = size(in_rc_datapacket_mtx);
if((r ~= cw_number)||(c ~= (in_primal_vec_dim + in_dual_vec_dim + 5)))
   error('The size of the third input matrix parameter passed to Gl_primal_updating_ser is wrong. The service call aborts');
end
%
%
%--------This part of the code can be customized by the user on the basis
%--of the actual updating of the global primal variables to be performed---
%
local_clock_vec = zeros(1, cw_number);
%The non-zero entries of local_clock_vec must store the current local clock values of the
%coworkers who participate to the current aggregation step.
for j=1:cw_number
    if (in_updating_map_vec(j) == 1) 
       %The j-th coworker sent a fresh packet
       local_clock_vec(1,j) =  in_rc_datapacket_mtx(j, end);    
    end %End if
end %End for
%
current_global_clock = max(local_clock_vec);
%By design, the current value of the global clock is the maximum of the
%local clock values of the coworkers who participate to the current
%aggregation step
%
age_weight_vec = zeros(1, cw_number);
%The non-zero entries of age_weight_vec must store the age-depending weights of the
%coworkers who participate to the current aggregation step
%
for j=1:cw_number
    if (in_updating_map_vec(j) == 1) 
       %The j-th coworker sent a fresh packet
       age_weight_vec(1,j) =  exp(-1.0*(current_global_clock - local_clock_vec(1,j)));      
       %The expression of the age-depending weight can be customized by the
       %user. An alternative setting may be: age_weight_vec(1,j) =  exp(-(current_global_clock - local_clock_vec(1,j)));
    end %End if
end %End for
%
out_vec_aux = zeros(1, in_primal_vec_dim);
for j=1:cw_number
    if (in_updating_map_vec(j) == 1) 
       %The j-th coworker sent a fresh packet
       out_vec_aux = out_vec_aux + (age_weight_vec(1,j)*in_rc_datapacket_mtx(j, 1:in_primal_vec_dim));
       %The local model provided by coworker#j is scaled down by the age-depending dumping coefficient age_weight_vec(1,j)    
    end %End if
end %End for
%
%
out_vec_aux = out_vec_aux/max(eps, sum(age_weight_vec));
%Compute the average of the accumulated fresh entries of in_rc_datapacket_mtx 
%while avoiding divisions by zero
%
%The basic Fed-Averaging corresponds to the case of alpha == 0; 
if ((alpha < 0) || (alpha >= 1))
    fprintf('Error message from Gl_primal_updating_ser\n');
    fprintf('The setting of the forgetting factor: alhpa is WRONG\n');
    fprintf('alhpa must fall in the semi-open interval: [0, 1)\n');
    error('FedE run aborts');  
end
%
out_gl_primal_vec = (alpha*in_gl_primal_vec) + ((1-alpha)*out_vec_aux);
%Perform a moving average updating of the returned gl_primal_vec
%------End of the part to be customized by the user----------------------
%
end %End seervice code