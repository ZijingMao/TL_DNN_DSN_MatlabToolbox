function testonDSN_Li
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex2 train a 100-100 hidden unit dsn and use its weights to initialize a NN
rand('state',0)
%train dsn
dsn.sizes = [100];
opts.numepochs =   1;
opts.batchsize = 200;
opts.momentum  =   0;
opts.alpha     =   1;
opts.stack     =   0;
dsn = dsnsetup(dsn, train_x, opts);
dsn = dsntrain(dsn, train_x, opts);

%unfold dsn to nn
nn = dsnunfoldtonn(dsn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = 200;
nn = dsnupdate(nn, train_x, train_y, opts);
%figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights
for i = 1 : 16
    nn = piledsns( nn );
    opts.stack     =   i;
    train_x = [nn.label train_x];
    nn = dsnupdate(nn, train_x, train_y, opts);
end

len = size(nn.W{1}, 2);
nn.W{1} = nn.W{1}(:, 10*opts.stack+1:len);
[er, bad] = nntest(nn, test_x, test_y, opts);

assert(er < 0.10, 'Too big error');
