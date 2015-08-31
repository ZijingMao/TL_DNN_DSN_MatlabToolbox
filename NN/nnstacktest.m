function [er, bad, a, b] = nnstacktest(nn, x, y, opt, k)
    [labels, a] = nnstackpredict(nn, x, opt, k);
    [~, expected] = max(y,[],2);
    bad = find(labels ~= expected);
    er = numel(bad) / size(x, 1);
    b = expected;
end