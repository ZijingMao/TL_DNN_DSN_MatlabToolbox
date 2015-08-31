function  test_NN(train_x, test_x, train_y, test_y)
%TEST_NN Summary of this function goes here
%   Detailed explanation goes here

A = max(max(train_x));
A = [A;max(max(test_x))];
B = min(min(train_x));
B = [B;min(min(test_x))];
range = [min(B);max(A)];
train_x	= (train_x-range(1))/(-range(1)+range(2));
test_x = (test_x-range(1))/(-range(1)+range(2));

[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

rand('state',0)
nn = nnsetup([20 18 2]);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
opts.stack = 0;
[nn, L] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y, opts);

end

