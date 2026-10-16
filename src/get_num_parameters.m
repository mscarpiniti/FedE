function n_el = get_num_parameters(M)
% Evaluate the number of parameters of an MLP model
    n_el = 0;
    L = length(M) - 1;
    for i = 1:L
        n_el = n_el + (M(i)+1)*M(i+1);
    end
end
