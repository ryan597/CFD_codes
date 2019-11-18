%% Lev-Marq optimation of omega

Sample_Size  = 50;
omega        = pi;
aspect_ratio = 2;

[xx,yy,C,C_true,res_it]=test_poisson_jacobiMC(Sample_Size,omega,aspect_ratio);

% Random sampling points
[X, Y]= meshgrid(xx, yy);
ran   = randsample(size(X,1)*size(X,2),50,true);

Xvec = reshape(X,size(X,1)*size(X,2),1);
Xvec = Xvec(ran);
Yvec = reshape(Y,size(Y,1)*size(Y,2),1);
Yvec = Yvec(ran);
% Adding random noise from normal distribution
sigma = max(Yvec)*0.1;
Cvec = reshape(C_true,size(C_true, 1)*size(C_true, 2),1);
Zi = Cvec(ran) + sigma.*normrnd(0, 1, size(Cvec(ran)));
% Give an initial guess for omega
guess_omega = rand(1)*2*pi;

xopt = LevMarq(guess_omega, Zi, Sample_Size, aspect_ratio, ran);

function Cost = cost(omega, Zi, Sample_Size, aspect_ratio, ran)
[~,~,ci]   = test_poisson_jacobiMC(Sample_Size, omega, aspect_ratio);
ci = reshape(ci, size(ci,1)*size(ci, 2), 1);
ci = ci(ran);
Cost = sum((Zi-ci).^2);
end

function [xopt, fopt, niter, gnorm, dx] = LevMarq(x0, Zi, Sample_Size, aspect_ratio, ran)
% Tolerences and parameters
epsilon1 = 1e-6;
epsilon2 = 1e-6;
maxiter = 1000;
tau=1e-6;
niter=0;
v = 2.0;
x = x0;
% Initialization of gradients and function values
F = cost(x0, Zi, Sample_Size, aspect_ratio, ran);
J = num_grad(@cost, x, 1e-6, Zi, Sample_Size, aspect_ratio, ran);
g = J'*F;
A = J*J';
mu = tau*max(max(A));
% Loop exit condition
found = (norm(g, Inf)<=epsilon1);
% Enter loop until iterations reach max or tolerence reached
while ~found && niter<maxiter
    niter = niter+1;
    h = -((A+mu*eye(size(A,1)))\g')'; 
    %h = -g *inv(A +mu*eye(2));
    % Gradient is smaller than tolerence -> exit loop
    if norm(h) <= epsilon2*(norm(x)+epsilon2)
        found= true;
    else
        xnew = x + h;
        % rho is the gain parameter (F(x)-F(xnew))/(L(0)-L(h))
        rho  = (cost(x, Zi, Sample_Size, aspect_ratio, ran)-cost(xnew, Zi, Sample_Size, aspect_ratio, ran))/(0.5*h*(mu*h-g)');
        if ~isfinite(xnew)
            display(['Number of iterations: ' num2str(niter)])
            error('x is inf or NaN')
        end
        % Acceptable steps i.e. when F(xnew)<F(x)
        if rho>0
            x = xnew;
            J = num_grad(cost, guess_omega, 1e-6, Zi, Sample_Size, aspect_ratio, ran);
            F = cost(x, Zi, Sample_Size, aspect_ratio, ran);
            A = J*J';     
            g = J'*F;
            mu = mu*max(1.0/3, 1.0-(2.0*rho-1.0)^3);
            v  = 2.0;
            found = (norm(g, Inf)<=epsilon1);
        else
            mu = mu*v;
            v = 2.0*v;
        end
    end
end
xopt = x;
fopt = cost(xopt, Zi, Sample_Size, aspect_ratio, ran);
niter= niter-1;
gnorm= norm(g);
dx   = norm(x-xnew);
display(["Optimum omega: ", xopt, ", Min Cost: ", fopt])
end

