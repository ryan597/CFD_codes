function [xopt, MSE, xvals, fvals] = lev_marq(x0, fun)
% Initialization point
x = x0;
% Tolerences and parameters
epsilon1 = 1e-6;
epsilon2 = 1e-8;
maxiter = 1000;
tau=1e-2;
niter=0;
v = 2.0;
% Initialization of gradients and function values
J = (fun(x+1e-3) - fun(x-1e-3))/(2*1e-3);
F = fun(x);
g = J'*F;
A = J*J';
mu = tau*max(max(A));
% Loop exit condition
found = (norm(g, Inf)<=epsilon1);
% Plotting contours
%Z = @(x1, x2) x1.^2+x1*x2+3*x2.^2;
%figure(1); clf; 
%fcontour(Z, [-5 5 -5 5]); 
%axis equal; hold on
% Enter loop until iterations reach max or tolerence reached
xvals = [x];
fvals = [F];
while ~found && niter<maxiter
    if mod(niter, 10)==0
        display(niter)
    end
    niter = niter+1;
    h = -((A+mu*eye(size(A)))\g')'; 
    %h = -g *inv(A +mu*eye(2));
    % Gradient is smaller than tolerence -> exit loop
    if norm(h) <= epsilon2
        if gradcount>3
            found= true;
        end
        gradcount = gradcount+1;
    else
        gradcount=0;
        xnew = x + h;
        % rho is the gain parameter (F(x)-F(xnew))/(L(0)-L(h))
        rho  = (fun(x)-fun(xnew))/(0.5*h*(mu*h-g)');
        if ~isfinite(xnew)
            display(['Number of iterations: ' num2str(niter)])
            error('x is inf or NaN')
        end
        % Acceptable steps i.e. when F(xnew)<F(x)
        if rho>0
            x = xnew;
            J = (fun(x+1e-3) - fun(x-1e-3))/(2*1e-3);; 
            F = fun(x);
            A = J*J';     
            g = J'*F;
            mu = mu*max(1.0/3, 1.0-(2.0*rho-1.0)^3);
            v  = 2.0;
            found = (norm(g, Inf)<=epsilon1);
            
            xvals = [xvals, x];
            fvals = [fvals, F];
        else
            mu = mu*v;
            v = 2.0*v;
        end
    end
end
xopt = x;
MSE  = fun(xopt);
end