
function rc_TH_out = Set_CoW_number_ser(it_in, rc_TH_input_in, inter_perturbation_window_in, or_max_loop_in, classification_classes_number_in, coworker_number_in) 

% The service: 
%
% rc_TH_out = Set_CoW_number_ser(squared_primal_variables_vec_in, squared_dual_variables_vec_in, obfun_vec_in, it_in, rc_TH_input_in, inter_perturbation_window_in, or_max_loop_in, classification_classes_number_in,...
%                          coworker_number_in)
%
% reports in output the number: rc_TH_out of coworkers that must participate
% to the (it_in)-th global iteration to be carried out by the orchestrator. 
% The actual strategy for the setting of rc_TH_out can be customized by the user 
% on the basis of the provided input API, that is,,
% (i) current vector: squared_primal_variables_vec_in of the available global squared primal
% variables
% (ii) current vector: squared_dual_variables_vec_in of the available global squared
% dual variables
% (iii) current vector: obfun_vec_in of the available global objective function values
% (iv) index: it_in of the current global iteration to be carried out by the
% orchestrator 
% (v) input value: rc_TH_input_in of the number of participating coworkers  
% (vi) number of global iterations: inter_perturbation_window_in between
% two adjacent perturbation actions performed by the orchestrator 
% (vii) total number: or_max_loop_in of global iterations to be carried out
% by the orchestrator 
% (viii) number of decision classes: classification_classes_number_in
% involved by the used training data sets, and, 
% (ix) total number: coworker_number_in of available coworkers. 
%
% The service is implemented as a MATLAB function.
%
%
if(inter_perturbation_window_in == Inf)
   dummy = or_max_loop_in + 1;
else
   dummy = inter_perturbation_window_in;
end
%
%
%-Code segment to be customized by the user on the basis of the actual
%-strategy to be implemented for the dynamic setting of the number of 
%-coworkers that must participate to the (it_in)-th global iteration carried 
%-out by the orchestrator-------------------------------------------------
%

rc_TH_out = randi([1, coworker_number_in],1,1);
%It implements the default time-varying strategy for the setting of the number of
%the coworkers which must participate to the (it_in)-th global iteration carried out by the 
%orchestrator
%
rc_TH_out = max(1, min(coworker_number_in, rc_TH_out)); 
%It guarantees that the returned value of rc_TH_out always falls in the feasible set: 
%{1, 2,.., coworker_number_in}.
%
end %End service code