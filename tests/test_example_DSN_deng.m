function [er] = test_example_DSN(train_x, test_x, train_y, test_y)
% load mnist_uint8;
% 
% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);

A = max(max(train_x));
A = [A;max(max(test_x))];
B = min(min(train_x));
B = [B;min(min(test_x))];
range = [min(B);max(A)];
train_x	= (train_x-range(1))/(-range(1)+range(2));
test_x = (test_x-range(1))/(-range(1)+range(2));

[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%%  ex2 train a 100-100 hidden unit dsn and use its weights to initialize a NN
rand('state',0)
%train dsn
dsn.sizes = [20];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   0.05;
opts.stack     =   0;
dsn = dsnsetup(dsn, train_x, opts);
dsn = dsntrain(dsn, train_x, opts);

%unfold dsn to nn
nn = dsnunfoldtonn(dsn, 2);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = 100;
nn = finetuning(nn, train_x, train_y, opts);
%figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights
for i = 1 : 1
    nn = piledsns( nn );
    opts.stack     =   i;
    
    %update new training set
    %train_x = [nn.label train_x];
    
    nn = finetuning(nn, train_x, train_y, opts);
end

%fix length before testing
%len = size(nn.W{1}, 2);
%nn.W{1} = nn.W{1}(:, 10*opts.stack+1:len);

[er, bad] = nntest(nn, test_x, test_y, opts);

assert(er < 0.10, 'Too big error');
