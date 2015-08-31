function nn = updateUW(nn, x, y)

    n = nn.n;
    m = size(x, 1);
    
    x = [x];

    nn.a{1} = x;
    
    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = 0;
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = 0;
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
        nn.a{i} = [nn.a{i}];
    end
    
    %update U
    nn.W{n - 1} = 0;
    A = nn.a{n-1}' * nn.a{n-1};
    D = inv(A);
    B = A \ nn.a{n-1}';
    C = B * y;
    nn.W{n - 1} = C;
    nn.W{n - 1} = nn.W{n-1}';
    
    switch nn.output 
        case 'sigm'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %update W
    H = nn.a{n-1}/(nn.a{n-1}'*nn.a{n-1});
    E = 2 * x';
    G = (nn.a{n-1}.*(1-nn.a{n-1}')');
    J = (H*(nn.a{n-1}'*y)*(y'*H) - y*(y'*H));
    F = G.*J;
    T = E * F;
    nn.vW{1} = T';
    for i = 1 : (nn.n - 2)
        dW = 0.01 * nn.vW{i};
        nn.W{i} = nn.W{i} - dW;
    end
    
    %error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end