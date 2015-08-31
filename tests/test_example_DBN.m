function [Az, pred, expected, nn] =  test_example_DBN(train_x, test_x, train_y, test_y, hidden1, alpha, iterations, batchsize)

% load mnist_uint8;
% 
% train_x = double(train_x)/255;
% test_x  = double(test_x)/255;
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

%%  ex1 train a 100 hidden unit RBM and visualize its weights
% rand('state',0)
% dbn.sizes = [100];
% opts.numepochs =   1;
% opts.batchsize = 100;
% opts.momentum  =   0;
% opts.alpha     =   1;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [100, hidden1];
opts.numepochs =   1;
opts.batchsize = batchsize;
opts.momentum  =   0;
opts.alpha     =   alpha;
opts.stack     =   0;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = batchsize;
nn = nntrain(nn, train_x, train_y, opts);

[er(1), bad, pred, expected] = nntest(nn, test_x, test_y, opts);
[Az(1),~,~,~]=rocarea(pred,test_y(:, 1));
% disp(['Az ' num2str(Az(1))]);
%disp(['err ' num2str(er(1))]);
for i = 1 : iterations-1
   nn = nntrain(nn, train_x, train_y, opts);
   [er(i+1), bad, pred, expected] = nntest(nn, test_x, test_y, opts);
   [Az(i+1),~,~,~]=rocarea(pred,test_y(:, 1));
    % disp(['Az ' num2str(Az(i+1))]);
    %disp(['err ' num2str(er(i+1))]);
end

