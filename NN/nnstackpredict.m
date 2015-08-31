function [labels, a] = nnstackpredict(nn, x, opt, k)
    nn.testing = 1;
    A = zeros(size(x,1), nn.size(end));
    nn = stacknnff(nn, x, A, opt.stack, k);
    nn.testing = 0;
    
    [~, i] = max(nn.a{end},[],2);
    labels = i;
    a = nn.a{end};
end
