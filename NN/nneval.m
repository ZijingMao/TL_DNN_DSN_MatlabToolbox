function [loss] = nneval(nn, loss, train_x, train_y, stack, val_x, val_y)
%NNEVAL evaluates performance of neural network
% Returns a updated loss struct
assert(nargin == 5 || nargin == 7, 'Wrong number of arguments');

% training performance
nn                    = nnff(nn, train_x, train_y, stack);
loss.train.e(end + 1) = nn.L;

% validation performance
if nargin == 7
    nn                    = nnff(nn, val_x, val_y, stack);
    loss.val.e(end + 1)   = nn.L;
end

%calc misclassification rate if softmax
if strcmp(nn.output,'softmax')
    [er_train, ~]               = nntest(nn, train_x, train_y);
    loss.train.e_frac(end+1)    = er_train;
    
    if nargin == 7
        [er_val, ~]             = nntest(nn, val_x, val_y);
        loss.val.e_frac(end+1)  = er_val;
    end
end

end
