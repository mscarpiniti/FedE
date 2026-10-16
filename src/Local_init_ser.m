function [out_ID, out_primal_local_vec, out_dual_local_vec] = Local_init_ser(in_ID, in_gl_primal_vec, in_gl_dual_vec, in_Pert_flag)

% The service:
%
%    [out_ID, out_primal_local_vec, out_dual_local_vec] = Local_init_ser(in_ID, in_gl_primal_vec, in_gl_dual_vec, in_Pert_flag)
%
% is called by the CoWorker_routine on a per coworker-basis before entering each batch of local
% iterations. It allows the customizable initialization of the local primal 
% and dual vectors: out_primal_local_vec, out_dual_local_vec
% on the basis of the corresponding global primal and dual vectors: 
% in_gl_primal_vec, in_gl_dual_vec lastly
% broadcasted by the orchestrator in both cases in which: 
% (i) a perturbation action has been not performed by the orchestrator 
% in the last global iteration, e.g.,in_Pert_flag == 0, and, 
% (ii) a perturbation action has been performed by the orchestrator 
% in the last global iteration, e.g.,in_Pert_flag == 1.
% In the default implementation, the service code directly copies the input global
% vectors into the corresponding otuput local ones in both cases of in_Pert_flag == 0 
% and in_Pert_flag == 1.
% The service is implemented as a MATLAB function.
%
out_ID = in_ID;

if(in_Pert_flag == 0) 
%The last global iteration performed by the orchestrator did not embraced a perturbation action
       out_primal_local_vec = in_gl_primal_vec; %Do NOT customize
       out_dual_local_vec = in_gl_dual_vec;     %these statements
elseif(in_Pert_flag == 1) 
%The last global iteration performed by the orchestrator embraced a perturbation action
      out_primal_local_vec = in_gl_primal_vec; %These statements are customizable
      out_dual_local_vec = in_gl_dual_vec;     %by the user 
else
   error('Error message from Local_init_ser: the passed Per_flag value is not binary. The service call aborts');
end %End if
%

end %End service code