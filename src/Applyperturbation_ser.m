function out_gl_primal_vec = Applyperturbation_ser(in_gl_primal_vec, in_perturbation_strength,...
    in_obf_last, in_obf_start, in_init_primal_vec)
%
% The service:
%  
% out_gl_primal_vec = Applyperturbation_ser(in_gl_primal_vec, 
%                      in_perturbation_strength,in_obf_last, in_obf_start, in_init_primal_vec)
%
% applies an i.i.d. zero-mean Gaussian perturbation to each scalar entry
% of the input global primal vector: in_gl_primal_vec or init vec: in_init_primal_vec.
% The goal is to attempt to allow the global primal variables to escape local 
% minima and/or saddle points of the used trained loss function.
% The service is implemented as a MATLAB function. 
% The per-entry variance of the applied normal perturbation is 
% (i) scaled up proportionally to in_perturbation_strength; and,
% (ii)scaled up for decreasing values of the absolute difference:
% |in_obf_start - in_obf_last|.
%
% All the input and output vector parameters of the service call must be row 
% vectors. 
%
[r,c]=size(in_gl_primal_vec);

if ((r ~= 1) || (isempty(in_gl_primal_vec))) 
   error('The first input parameter passed to the: Applyperturbation_ser MUST be a NON-empty ROW vector. The service call aborts');
end % end if
%
%Apply perturbation to each scalar entries of in_init_primal_vec
aux = (in_init_primal_vec*(in_init_primal_vec)')/length(in_init_primal_vec);
%
%The perturbation standard deviation scales up for decreasing: |in_obf_last - in_obf_start|,
%e.g., the applied perturbation strength increases with the flatness of
%of the attraction basin of the local minimum around which the global primal
%vector is currently trapped.
%The additive unit term guarantees that, at non-vanishing in_perturbation_strength,
%perturbation_st_dev is NOT zero even in the case of vanishing: in_gl_primal_vec.^2 
%
perturbation_st_dev = +sqrt(in_perturbation_strength*aux);

%Apply random perturbation
out_gl_primal_vec = ((max(0.90, min(1.10,(rand(1,length(in_init_primal_vec))+0.5)))).*in_init_primal_vec) + (perturbation_st_dev*randn(1,length(in_init_primal_vec)));

end %End function