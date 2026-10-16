function [out_iter_number, out_ID] = Set_iter_ser(in_ID,...
    in_gl_last_primal_vec, in_gl_last_dual_vec, in_cw_primal_vec,...
    in_cw_dual_vec, in_max_iteration_number, in_cw_rc_Dpacket_count, in_or_max_iteration, epsilon, in_local_iter_setting_flag)
%
% The service:
%
% [out_iter_number, out_ID] = Set_iter_ser(in_ID,in_gl_last_primal_vec, in_gl_last_dual_vec,in_cw_primal_vec,in_cw_dual_vec, in_max_iteration_number,in_cw_rc_Dpacket_count, ...
%                                      in_or_max_it, epsilon, in_local_iter_setting_flag)
%
% returns the number: out_iter_number of consecutive local iterations to be
% performed by the calling coworker on the last D_packet received from the
% orchestrator. It is guaranteed that the returned out_iter_number falls
% in the integer set: {1, 2, ...,in_max_iteration_number}.
% The service is iplemented as a MATLAB function and all the input vector parameters are row vectors.
% The code for the evaluation of out_iter_number can be customized by the user.
% By default, the implemented strategy is to increase the number of local
% iterations: out_iter_number performed by each coworker for increasing values
% of the number of global aggregations: in_cw_rc_Dpacket_count performed by the orchestrator. 
% This strategy relies on the consideration that the average aggregation rate would be high at the 
% training start in order to allow fast alignment of the per-coworker local 
% models, and, then, it would decrease near the training end, e.g., 
% when the per-coworker local models are already aligned enough.   
%
out_ID = in_ID;
gl_primal_vec_length = length(in_gl_last_primal_vec);
cw_primal_vec_length = length(in_cw_primal_vec);
gl_dual_vec_length = length(in_gl_last_dual_vec);
cw_dual_vec_length = length(in_cw_dual_vec);
if((gl_primal_vec_length ~= cw_primal_vec_length)||(gl_dual_vec_length ~= cw_dual_vec_length))
    error('Error from Set_iter_ser function: the lengths of the passed local/global primal/dual vectors do NOT agree. The service call aborts');
end %End if
%
if(gl_dual_vec_length == 0)
    fprintf('Warning: the implemented FL optimization is NOT using local/global dual variables\n');
end % End if
%
%--------------------Customized generation of out_iter_number--------------
%This part of the code can be customized by the user in agreement with the
%actual algorithm that is designed to set the number of consecutive
%local iterations to be carried out by the calling coworker for processing 
%each normal D_packet received from the orchestrator
%
switch in_local_iter_setting_flag 
    case 0 %Default static strategy
         out_iter_number = in_max_iteration_number;
    %Static updating strategy: all coworkers always perform batches of in_max_iteration_number local iterations
    %
    case 1 %Default time-varying strategy
         out_iter_number = randi([1,in_max_iteration_number], 1, 1);
         %Random selection of out_iter_number
         %
         %out_iter_number = min(in_max_iteration_number, 2 + floor(in_cw_rc_Dpacket_count/(in_or_max_iteration/max(1,(in_max_iteration_number-1)))));    
         %Time-varying INCREASING strategy.
         %out_iter_number is progressively increased with the number: in_cw_rc_Dpacket_count  
         %of packets received by the calling coworker
    %
        %out_iter_number = max(2, in_max_iteration_number - floor(in_cw_rc_Dpacket_count/(in_or_max_iteration/max(1,(in_max_iteration_number-1)))));    
        %Time-varying DECREASING strategy. out_iter_number is progressively
        %decreased with the number: in_cw_rc_Dpacket_count of the
        %global aggregations performed by the orchestrator. 
        %A minimum of a single local iteration is guaranteed
    %
    case 2 %Default per-coworker adaptive strategy
         out_iter_number = min(in_max_iteration_number, 2 + floor(epsilon/(sum(((in_gl_last_primal_vec - in_cw_primal_vec).^2)))));     
    %Per-coworker adaptive strategy. According to the AFAFed approach, the
    %number of local iterations performed by each coworker is incresed (resp., decreased)
    %when the squared gap between the current global model:
    %in_gl_last_primal_vec and the current per-coworker local model:
    %in_cw_primal_vec is small (resp., high).
    otherwise
        error('Error message from Set_iter_ser: the passed value of local_iter_setting_flag is invalid. The service call aborts');
end %End switch
%--------------------------------------------------------------------------
%
%
end %End service code