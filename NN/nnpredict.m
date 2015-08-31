function [labels, a] = nnpredict(nn, x, opt)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)), opt.stack);
    nn.testing = 0;
    
    [~, i] = max(nn.a{end},[],2);
    labels = i;
    a = nn.a{end};
end
