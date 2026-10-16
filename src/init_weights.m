function ww = init_weights(M)
% Initialize the MLP2 weights and biases    
    ww = [];%Column vector
    for j = 1:length(M)-1
        sig = sqrt(2/(M(j) + M(j+1)));
        ww = [ww; sig*randn(M(j)*M(j+1), 1); zeros(M(j+1), 1)];
    end    
end