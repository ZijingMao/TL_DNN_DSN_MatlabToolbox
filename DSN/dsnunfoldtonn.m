function [ nn ] = dsnunfoldtonn( dsn, outputsize )
%dsnUNFOLDTONN Unfolds a dsn to a NN
%   dsnunfoldtonn(dsn, outputsize ) returns the unfolded dsn with a final
%   layer of size outputsize added.
    if(exist('outputsize','var'))
        size = [dsn.sizes outputsize];
    else
        size = [dsn.sizes];
    end
    nn = nnsetup(size);
    for i = 1 : numel(dsn.rbm)
        nn.W{i} = [dsn.rbm{i}.c dsn.rbm{i}.W];
    end
end

