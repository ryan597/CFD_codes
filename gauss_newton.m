function [beta, MSE, xvals, fvals] = gauss_newton( beta0, fun)

    x0 = beta0 ;
% termination tolerance
    tol = 1e-6;

% maximum number of allowed iterations
    maxiter = 1000;

% minimum allowed perturbation
    dxmin = 1e-8;

% step size ( 0.33 causes instability, 0.2 quite accurate)
    alpha = 0.01;
    
% initialize gradient norm, optimization vector, iteration counter, perturbation
    gnorm = inf; x = x0; dx = inf;
    niter=2;
    xvals = [];
    fvals = [];
    while and(gnorm>=tol, and(niter <= maxiter, dx >= dxmin))
    %----------------
        dh = 1e-4;
        
    % evaluate the objective function at the left and right points
        y1 = fun(x - dh);
        y2 = fun(x + dh);
    
    % calculate the slope (rise/run) for dimension i
        Jacob_f = (y2 - y1) / (2*dh);
    %-------------------
        Jacob_t = transpose(Jacob_f);
        g = -Jacob_t * fun(x) ./ (  Jacob_t * Jacob_f );
        gnorm = norm(g);
    
    % take step:
        beta  = x + alpha*g;
        dx    = norm(alpha*g);
        x     = beta;
        niter = niter + 1;
        if mod(niter, 10)==0
            xvals = [xvals, x];
            fvals = [fvals, fun(x)];
            display(niter)
        end
    end
    MSE = fun(beta);
end