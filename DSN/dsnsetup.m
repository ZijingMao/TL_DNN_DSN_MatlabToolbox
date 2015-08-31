function [ dsn ] = dsnsetup( dsn, x, opts )


    n = size(x, 2);
    dsn.sizes = [n, dsn.sizes];

    for u = 1 : numel(dsn.sizes) - 1
        dsn.rbm{u}.alpha    = opts.alpha;
        dsn.rbm{u}.momentum = opts.momentum;

        dsn.rbm{u}.W  = zeros(dsn.sizes(u + 1), dsn.sizes(u));
        dsn.rbm{u}.vW = zeros(dsn.sizes(u + 1), dsn.sizes(u));

        dsn.rbm{u}.b  = zeros(dsn.sizes(u), 1);
        dsn.rbm{u}.vb = zeros(dsn.sizes(u), 1);

        dsn.rbm{u}.c  = zeros(dsn.sizes(u + 1), 1);
        dsn.rbm{u}.vc = zeros(dsn.sizes(u + 1), 1);
    end


end

