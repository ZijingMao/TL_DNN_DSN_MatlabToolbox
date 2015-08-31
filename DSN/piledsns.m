function [ nn ] = piledsns( nn )
%PILEDSNS Summary of this function goes here
%   Detailed explanation goes here
nn.size(1) = nn.size(1)+nn.size(nn.n);

for i = 2 : nn.n-1   
    % weights and weight momentum
    newFeatureWeight = (rand(nn.size(i), nn.size(nn.n)) - 0.5) * 2 * 4 * ...
        sqrt(6 / (nn.size(i) + nn.size(i - 1)));
    arr = nn.W{i-1}(:, 1);
    nn.W{i-1}(:, 1) = [];
    %nn.W{i - 1} = cat(2, newFeatureWeight, nn.W{i - 1});
    nn.W{i-1} = [arr newFeatureWeight nn.W{i-1}];
    nn.vW{i - 1} = zeros(size(nn.W{i - 1}));

    % average activations (for use with sparsity)
    nn.p{i} = zeros(1, nn.size(i));
end

end

