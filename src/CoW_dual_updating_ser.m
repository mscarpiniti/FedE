function [out_cw_dual_vec, out_ID, out_first_momentum_dual_gradient_vec, out_second_momentum_dual_gradient_vec] = CoW_dual_updating_ser(in_ID,...
    in_cw_primal_vec, in_cw_dual_vec, in_gl_last_primal_vec, in_gl_last_dual_vec, in_iter, in_first_momentum_dual_gradient_vec, in_second_momentum_dual_gradient_vec, epsilon, in_local_solver_flag)
%
%T he service:
%
% [out_cw_dual_vec, out_ID, out_first_momentum_dual_gradient_vec, out_second_momentum_dual_gradient_vec] = ...
%                             CoW_dual_updating_ser(in_ID, in_cw_primal_vec, in_cw_dual_vec, ...
%                                             in_gl_last_primal_vec, in_gl_last_dual_vec, in_iter, ...
%                                             in_first_momentum_dual_gradient_vec, in_second_momentum_dual_gradient_vec, epsilon, in_local_solver_flag)
%
% performs a single local dual iteration by exploiting the passed input parameters:
% in_cw_primal_vec, in_cw_dual_vec, in_gl_last_primal_vec,
% in_gl_last_dual_vec,in_iter, in_first_momentum_dual_gradient_vec,
% in_second_momentum_dual_gradient_vec,
% and returns to the calling coworker:
% (i) the updated local dual vector: out_cw_dual_vec;
% (ii) the echo of the coworker ID: out_ID; and
% (iii) the average first and second momenta of the dual Lagrangian gradient
% vector.
% It is implemented as a MATLAB function.
% If length(in_cw_dual_vec) == length(in_gl_last_dual_vec) == 0,
% then, 
% out_cw_dual_vec = out_first_momentum_dual_gradient_vec = out_second_momentum_dual_gradient_vec = [] 
% are returned.
%
% All the input and output vector parameters are row vectors.
% The input parameter: in_iter passes the (integer-valued and positive) index
% of the local iteration currently executed by the calling coworker.
%
% The code of the service can be customized by the user on the basis of
% the actually implemented local dual solver.
%
% The default implementation generates: out_cw_dual_vec according to the
% implemented consensus algorithm, in which the local primal variable vector 
% of each coworker is subject to a single consensus-type scalar constraint, 
% so as to enforce the alignment of the local and global vectors of the primal
% variables.
%
% The implemented default local dual solvers are SG and Adam.

%--Setting of the regularizing parameters used by the SG dual local solver-
a_0 = 0.01; 
a_1 = 1.0;
%
%
%-Setting of the regularizing parameters used by the Adam dual local solver-
eta = 0.01;
rho1= 0.9; 
rho2= 0.99;
delta= 10^-8;
%
%---------------End setting of the local variables----------------------------
%
%-----------------Sanity check-------------------------------------------
gl_primal_vec_length = length(in_gl_last_primal_vec);
cw_primal_vec_length = length(in_cw_primal_vec);
gl_dual_vec_length = length(in_gl_last_dual_vec);
cw_dual_vec_length = length(in_cw_dual_vec);
if((gl_primal_vec_length ~= cw_primal_vec_length)||(gl_dual_vec_length ~= cw_dual_vec_length))
    error('ERROR from the CoW_dual_updating_ser: the lengths of the passed local/global primal/dual vectors do NOT agree. The service call aborts');
end %End if
%
%
if(gl_dual_vec_length == 0)
    fprintf('WARNING: the implemented FL optimization is NOT using local/global DUAL variables\n');
    out_cw_dual_vec = [];                      % All the returned
    out_first_momentum_dual_gradient_vec = []; % vector parameters
    out_second_momentum_dual_gradient_vec = [];% are empty
    return; %Early exit the service
end % End if
%
if (epsilon <= 0)
    fprintf('Error message from CoW_dual_updating_ser: the setting of the epsilon parameter is wrong\n');
    fprintf('The epsilon parameter must be STRICTLY POSITIVE\n');
    error('The service call aborts and the FedE runn exits\n');
end
%
%-------------------End sanity check---------------------------------------
%
out_ID = in_ID;
%
%------This part of code is executed ONLY if gl_dual_vec_length >= 1-------
%
%-------------Generate the ROW vector: cw_dual_Lagrangian_grad_vec-------- 
%This part of the code can be customized by the user on the basis of the
%actual expression assumed by the Lagrangian function of the actually
%handled optimization problem. 
%Formally, cw_dual_Lagrangian_grad_vec is the ROW vector of the gradients
%of the Lagrangian function performed with respect to the local DUAL
%variables available at the coworker.  
%It must be a ROW vector of size: (1 x cw_dual_vec_length).  
%Its default implementation corresponds to the case of the Imperfect
%Consensus Optimization problem subject only to the imperfect consensus
%constraints with tolerance threshold equal to epsilon. 

cw_dual_Lagrangian_grad_vec = (((in_cw_primal_vec - in_gl_last_primal_vec)*...
    (in_cw_primal_vec - in_gl_last_primal_vec)') - epsilon)*ones(1,cw_dual_vec_length); 

%
%-------------Generate the row vector: out_cw_dual_vec------------------- 
%This part of the code can be customized by the user on the basis of the 
%actually implemented local dual solver.
%The default implementation generates out_cw_dual_vec as a
%ascent SG or Adam dual iteration. 

switch in_local_solver_flag

    case 0 %SGD is run as dual local solver
         step_size2_vec = (a_0*ones(1,length(cw_dual_Lagrangian_grad_vec)))./(sqrt((a_1*ones(1,length(cw_dual_Lagrangian_grad_vec))) + ((cw_dual_Lagrangian_grad_vec)^2)));
         %Computation of the adaptive time-varying step-size vector for the SG
         %
         %Updating the local dual variables under the ascent SG dual local solver
         out_cw_dual_vec = max(0, in_cw_dual_vec + (step_size2_vec .* cw_dual_Lagrangian_grad_vec));
         out_first_momentum_dual_gradient_vec = in_first_momentum_dual_gradient_vec; %It is not used under SG
         out_second_momentum_dual_gradient_vec = in_second_momentum_dual_gradient_vec; %It is not used under SG


    case 1 %Adam is run as dual local solver
         %-Generate the Adam's first and second momentum dual gradient iterations-
         biased_first_momentum_gradient_vec = (rho1*in_first_momentum_dual_gradient_vec) + ((1.0-rho1)*cw_dual_Lagrangian_grad_vec);
         biased_second_momentum_gradient_vec = (rho2*in_second_momentum_dual_gradient_vec) + ((1.0-rho2)*(cw_dual_Lagrangian_grad_vec.^2));
         out_first_momentum_dual_gradient_vec = (1.0*biased_first_momentum_gradient_vec)/(1.0-((rho1)^(in_iter+0))); 
         out_second_momentum_dual_gradient_vec = (1.0*biased_second_momentum_gradient_vec)/(1.0-((rho2)^(in_iter+0))); 
         %
         %-Updating the local dual variables under the ascent Adam dual local solver-
         out_cw_dual_vec = max(0, in_cw_dual_vec + (eta*((out_first_momentum_dual_gradient_vec)./(sqrt(out_second_momentum_dual_gradient_vec) + (delta*ones(1,cw_dual_vec_length))))));

    otherwise
       error('Error message from CoW_dual_updating_ser: the passed value of local-solver_flag is not allowed: FedE run aborts');
end %End switch
%
end %End function 
%
%