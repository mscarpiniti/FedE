function [out_cw_local_loss, out_cw_primal_vec, out_ID, out_first_momentum_primal_gradient_vec, out_second_momentum_primal_gradient_vec] = CoW_primal_updating_ser(in_ID, in_cw_primal_vec, in_cw_dual_vec, in_gl_last_primal_vec, in_gl_last_dual_vec, in_iter, in_gl_loss_value, in_first_momentum_primal_gradient_vec, ...
    in_second_momentum_primal_gradient_vec,in_X, in_Y, in_MB_size,in_dataset_distribution_type, in_coworker_number, in_local_solver_flag)

% The service:
%
% [out_cw_local_loss, out_cw_primal_vec, out_ID, out_first_momentum_primal_gradient_vec,out_second_momentum_primal_gradient_vec] = CoW_primal_updating_ser(in_ID,...
%                 in_cw_primal_vec, in_cw_dual_vec, in_gl_last_primal_vec,...
%                 in_gl_last_dual_vec, in_iter, in_gl_loss_value,in_first_momentum_primal_gradient_vec,in_second_momentum_primal_gradient_vec, in_X, in_Y,...
%                 in_dataset_distribution_type, in_coworker_number, in_local_solver_flag)
%
% performs a single local primal iteration by exploiting the passed input parameters:
% in_cw_primal_vec, in_cw_dual_vec, in_gl_last_primal_vec,
% in_gl_last_dual_vec, in_iter, in_gl_loss_value,out_second_momentum_primal_gradient_vec
% and returns to the calling coworker:
% (i) the updated local primal vector: out_cw_primal_vec;
% (ii) the last computed value of the corresponding scalar-valued local loss
% function: out_cw_local_loss; and,
% (iii) the echo of the coworker ID: out_ID
%
% The service is implemented as a MATLAB function.
% All the input and output vector parameters are row vectors.
%
% The input parameter: in_gl_loss_value passes the last
% average value of the global training loss function that has been computed by the
% orchestrator and broadcasted to all coworkers.
%
% The input parameter: in_iter passes the (integer-valued and positive) index
% of the local iteration currently performed by the calling coworker.
%
% The code of the function can be customized by the user on the basis of
% the actually implemented primal local solver algorithm.
%
out_ID = in_ID;
%
%-----------------Sanity check--------------------------------------------
gl_primal_vec_length = length(in_gl_last_primal_vec);
cw_primal_vec_length = length(in_cw_primal_vec);
gl_dual_vec_length = length(in_gl_last_dual_vec);
cw_dual_vec_length = length(in_cw_dual_vec);
if((gl_primal_vec_length ~= cw_primal_vec_length)||(gl_dual_vec_length ~= cw_dual_vec_length))
    error('ERROR from CoW_primal_updating_ser: lengths of the passed local/global primal/dual vectors do NOT agree. The service call aborts');
end % End if
%
if(gl_dual_vec_length == 0)
    fprintf('Warning: the implemented FL optimization is NOT using local/global dual variables\n');
end % End if
%
[r,c]=size(in_gl_loss_value);
if((r ~= 1)||(c ~= 1))
  error('Error from CoW_primal_updating_ser: the passed in_gl_loss_value input parameter is NOT scalar. The service call aborts');  
end
%
if(in_iter == 0)
    in_iter = in_iter+1;
end
%
%-------------------End sanity check---------------------------------------
%
%------------------Set local variables-------------------------------------
%
%----Setting of the regularizing parameters used by the SGD local solver--
%
a_0 = 0.01; 
a_1 = 1.5;
%
%---Setting of the regularizing parameters used by Adam local solver--------
eta = 0.0005; 
rho1 = 0.9;
rho2 = 0.999; 
delta = (10)^(-8);
%-----------------------End set local constants---------------------------
%
%----------------Generate the row gradient vector--------------------------
%This part of the code can be customized by the user on the basis of 
%the actually trained ML model by calling a further suitable library function
%that performs the (typically, backward-propagation and minibatch-based) 
%computation of the stochastic gradient of the adopted global training loss function.  
%The returned local loss value must be scalar.
%The returned cw_primal_grad_vec must be a row vector of size:
%(1 x cw_primal_vec_length).
%Formally speaking, cw_primal_grad_vec is the vector gradient of the 
%adopted global objective function performed with respect to the local 
%PRIMAL variables of the coworker (e.g., the weights of the trained ML model
%that are locally available at the coworker) and averaged over the local
%training data available at the calling coworker.
%
%----On-the-fly per-coworker data loading and IID/1-NONIID data partitioning-----------
%A data-loading service is implemented by cw_primal_fun_new routine, in
%order to perform per-coworker data loading and data partitioning according to the required IID/1-NONIID dataset
%distribution type in on-the-fly fashion.
%For this purpose, each coworker is assumed to be equipped with a local Data Loader 
%that fetches data on-the-fly during the training from the underlying local storage. 
%In principle, the Data Loader module is a local daemon that runs as
%a process or thread. 
%
%-------------------Data loading-------------------------------------------
[number_of_features, number_of_classes] = size(in_Y);
[number_of_features, feature_size] = size(in_X);
%The input parameters in_X and in_Y store the full training feature set and
%the corresponding labels in double format 

%-On-the-fly generation of IID/1-NONIID per-coworker training datasets----%
current_minibatch_mtx = [];
desired_label_mtx = [];

switch in_dataset_distribution_type
    %
    case 0 %Generate IID per-coworker training datasets
         current_minibatch_mtx = [];
         desired_label_mtx = [];
         for i=1:1:in_MB_size
             feature_index = randi(number_of_features);
             current_minibatch_mtx = [current_minibatch_mtx; in_X(feature_index,:)];
             desired_label_mtx = [desired_label_mtx; in_Y(feature_index,:)];
         end %End for in the i-index
   %
    case 1 %Generate 1-NONIID per-coworker training datasets
         if (number_of_classes > in_coworker_number) %Sanity check
             error('Error from cw_primal_fun_new: under 1-NONIID training datasets, the number of coworkers must be NOT lower than the number of decision classes. FedE run aborts');
         end %End if
         %
         switch number_of_classes
             case {1,2} % Generate 1-NONIID per-coworker training datasets under binary classifiers 
                  if(mod(in_ID,2) == 0) % The calling coworker is even
                     current_minibatch_mtx = [];
                     desired_label_mtx = [];
                     for i=1:1:in_MB_size
                         true = 0;
                         while (true == 0)
                             feature_index = randi(number_of_features);
                             if (in_Y(feature_index) == +1) %Even coworkers are trained only on (+1)-labeled training features
                                 current_minibatch_mtx = [current_minibatch_mtx; in_X(feature_index,:)];
                                 desired_label_mtx = [desired_label_mtx; in_Y(feature_index,:)];
                                 true = 1;
                             else
                                 true = 0;
                             end %End if
                         end %End while
                     end %End for in the i-index
                  end %End if on even coworkers
                  %
                  if(mod(in_ID+1,2) == 0) %The calling coworker is ODD
                     current_minibatch_mtx = [];
                     desired_label_mtx = [];
                     for i=1:1:in_MB_size
                         true = 0;
                         while (true == 0)
                             feature_index = randi(number_of_features);
                             if (in_Y(feature_index) == -1) %Odd coworkers are trained only on (-1)-labeled training features
                                 current_minibatch_mtx = [current_minibatch_mtx; in_X(feature_index,:)];
                                 desired_label_mtx = [desired_label_mtx; in_Y(feature_index,:)];
                                 true = 1;
                             else
                                 true = 0;
                             end %End if
                         end%End while
                     end %End for in the i-index
                  end%End if on ODD coworkers
               %End case number_of_classes <= 2
               %
           otherwise %Generate 1-NONIID per-coworker training datasets under multi-class classifiers 
                   current_minibatch_mtx = [];
                   desired_label_mtx = [];
                   for i=1:1:in_MB_size
                       true = 0;
                       while (true == 0)
                              feature_index = randi(number_of_features);
                              if((mod(in_ID, number_of_classes) + 1) == find(in_Y(feature_index,:)))
                                 current_minibatch_mtx = [current_minibatch_mtx; in_X(feature_index,:)];
                                 desired_label_mtx = [desired_label_mtx; in_Y(feature_index,:)];
                                 true = 1;
                              else
                                  true=0;
                              end %End if
                       end %End while
                   end %End for in the i-index
         end %End innermost switch on number_of_classes
    %
    otherwise %Error in the passed in_dataset_distribution_type's value
      error('Error message from cw_primal_fun_new: the passed value of dataset_distribution_type is NOT allowed. FedE run aborts');    
end %End main switch on in_dataset_distribution_type
%
%
%
[r,c] = size(current_minibatch_mtx);
if((r~=in_MB_size)||(c~=feature_size))
  error('Message from cw_primal_fun_new: the size of the built up current_minibatch_mtx is WRONG. FedE aborts');
end %End if
%
[r,c] = size(desired_label_mtx);
if((r~=in_MB_size)||(c~=number_of_classes))
  error('Message from cw_primal_fun_new: the size of the built up desired_label_mtx is WRONG. FedE aborts');
end %End if
%------------End generation of the per-coworker training datasets----------
%
%
%---------------------------Call the gradient function--------------------%
%[cw_primal_grad_vec, out_cw_local_loss, backID, minibatch_size] = SVM_gradient_fun(in_cw_primal_vec', current_minibatch_mtx, desired_label_mtx, in_ID);
[cw_primal_grad_vec, out_cw_local_loss, backID, minibatch_size] = MLP1_gradient_fun(in_cw_primal_vec', current_minibatch_mtx, desired_label_mtx, in_ID);
%[cw_primal_grad_vec, out_cw_local_loss, backID, minibatch_size] = MLP2_gradient_fun(in_cw_primal_vec', current_minibatch_mtx, desired_label_mtx, in_ID);
%--------------------------------------------------------------------------
%
%
if (backID ~= in_ID)
    fprintf('Error message from CoW_primal_updating_ser: the returned coworker''s ID does not match the input  coworker''s ID\n');
    error('FedE run aborts');
end
%
[r,c]= size(cw_primal_grad_vec);
if (r > c)
    cw_primal_grad_vec = (cw_primal_grad_vec)'; %Turn the returned gradient to a row vector
    %The returned gradient vector must be a ROW vector
end

[r,c]= size(cw_primal_grad_vec);
if((r~=1)||(c~=cw_primal_vec_length))
    fprintf('Error message from CoW_primal_updating_ser\n');
    fprintf('The size of the returned gradient vector:cw_primal_grad_vec is wrong\n');
    fprintf('The expected size of:cw_primal_grad_vec is: (1 x %d)\n', cw_primal_vec_length);
    error('FedE aborts');
end
%--------------------------------------------------------------------------
%
%-------Generate the ROW vector: cw_primal_Lagrangian_function_grad_vec----
if (cw_dual_vec_length >= 1)
      cw_primal_Lagrangian_function_grad_vec = cw_primal_grad_vec + (2*in_cw_dual_vec(1)*(in_cw_primal_vec - in_gl_last_primal_vec));
      %This is the gradient of the underlying Lagrangian function in the 
      %case of the Imperfect Consensus Optimization problem subject to 
      %only the imperfect consensus constraints.
      %In general, the computation of
      %cw_primal_Lagrangian_function_grad_vec can be customized
      %by the user on the basis af the actual expression of the Lagrangian function of the
      %actually handled distributed optimization problem. 
elseif (cw_dual_vec_length == 0)
      cw_primal_Lagrangian_function_grad_vec = cw_primal_grad_vec; 
end
%
%
%---Run an iteration of the selected local solver to update the local
%--------------------primal vector variables------------------------------

switch in_local_solver_flag

    case 0 %SGD is used a primal local solver
         step_size1_vec = (a_0*ones(1,length(cw_primal_Lagrangian_function_grad_vec)))./(sqrt((a_1*ones(1,length(cw_primal_Lagrangian_function_grad_vec))) + ((cw_primal_Lagrangian_function_grad_vec).^2))); 
         %Evaluation of the component-wise adaptive step-size 
         out_cw_primal_vec = in_cw_primal_vec - (step_size1_vec .* cw_primal_Lagrangian_function_grad_vec);
         out_first_momentum_primal_gradient_vec = in_first_momentum_primal_gradient_vec; 
         out_second_momentum_primal_gradient_vec = in_second_momentum_primal_gradient_vec;
         %SG primal iteration

    case 1 %Adam is used as primal local solver
         %-Generate the Adam's first and second momentum gradient iterations------- 
         if (in_iter == 1)
            biased_first_momentum_gradient_vec = cw_primal_Lagrangian_function_grad_vec;
            biased_second_momentum_gradient_vec = (cw_primal_Lagrangian_function_grad_vec.^2); 
         else
            biased_first_momentum_gradient_vec = (rho1*in_first_momentum_primal_gradient_vec) + ((1-rho1)*cw_primal_Lagrangian_function_grad_vec);
            biased_second_momentum_gradient_vec = (rho2*in_second_momentum_primal_gradient_vec) + ((1-rho2)*(cw_primal_Lagrangian_function_grad_vec.^2)); 
         end %End if

         out_first_momentum_primal_gradient_vec = (1.0*biased_first_momentum_gradient_vec)/(1.0-((rho1)^(in_iter+0))); 
         out_second_momentum_primal_gradient_vec = (1.0*biased_second_momentum_gradient_vec)/(1.0-((rho2)^(in_iter+0)));
         %
         %Updating of the local primal variables under the Adam local solver
         out_cw_primal_vec = in_cw_primal_vec - (eta*((out_first_momentum_primal_gradient_vec./(sqrt(out_second_momentum_primal_gradient_vec) + (delta*ones(1,cw_primal_vec_length))))));
    otherwise
         error('Error message from CoW_primal_updating_ser: the passed value of local_solver_flag is not allowed. FedE run aborts');

end %End switch
%
end %End function 
%

