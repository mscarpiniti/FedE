function out_binary_flag = Eval_pert_cond_ser(in_obfun_last,...
    in_obfun_start, in_global_primal_vec, in_last_perturbated_primal_vec,...
    in_it, in_last_perturbation_loop_index, in_perturbation_TH)
%
% The service:
%
% out_binary_flag = Eval_pert_cond_ser(in_obfun_last,...
%    in_obfun_start, in_global_primal_vec, in_last_perturbated_primal_vec,...
%    in_it, in_last_perturbation_loop_index, in_perturbation_TH)
%
% evaluates the condition that fires the application of a (single) 
% perturbation action on each scalar entry of the current vector of the global 
% primal variables at the orcherstrator side. 
% The service is implemented as a MATLAB function.
%
% In principle, the perturbation condition is considered to be met 
% when the norm of the gradient vector of the global objective function 
% done w.r.t. the global primal variables is below a (small enough) 
% activation threshold. This would require the availability of the analytical
% expression of the gradient vector, which, in turn, is application-depending. 
%
% In order to avoid the explicit use of the analytical expression of the 
% gradient vector of the used application-depending global objective function, 
% we propose to replace it by the so_called: "Norm of the Memory-Gradient".
% This last:
% (i) allows to account for the behavior of the global training loss function 
% over a large-enough customizable global iteration window, e.g., it is a gradient 
% with "memory", in which the memory span is given as the input parameter:
% inter_perturbation_window;
%
% and,
%
% (ii) can be numerically evaluated without requiring the explicit analytical
% expressions of the gradient of the used global training loss function, e.g., 
% it does not require to change the code of this service when the used global 
% training loss function changes.
%
% The returned out_binary_flag is unit (resp., zero)-valued when the
% perturbation condition is met (resp., is not met).
%
in_it = in_it;
%At the present, it is not used
in_last_perturbation_loop_index = in_last_perturbation_loop_index;
%At the present, it is not used
%
%Calculate the proposed "norm of the gradient with memory" 
memory_grad_norm = (abs(in_obfun_start - in_obfun_last))/...
    (1+sqrt((in_last_perturbated_primal_vec - in_global_primal_vec)*(in_last_perturbated_primal_vec - in_global_primal_vec)'));
%
%
%Evaluate the condition for applying the perturbation action
if (memory_grad_norm <= ((in_perturbation_TH)*(0.5*(abs(in_obfun_start) + abs(in_obfun_last)))))
  out_binary_flag = 1; %Perturbation is applied
else
  out_binary_flag = 0; %Perturbation is not applied
end %End if
%
end %End service