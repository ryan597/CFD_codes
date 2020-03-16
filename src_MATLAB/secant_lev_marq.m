function [xopt, MSE] = secant_lev_marq(x0, fun)


delta = 0.01;
epsilon1 = 1e-6;
epsilon2 = 1e-6;
maxiter  = 1e3;
niter=0;

B0 = (fun(x0+delta)-fun(x0-delta))/(2*delta);

x = x0;
B = B0;
j = 0;
n = length(x);
g = B'*f(x);
mu = max(max(B*B'));
v  = 2;
found = (norm(g, Inf)<=epsilon1);

while ~found && niter<maxiter
   niter = niter+1;
   h = -((B*B'+mu*eye(2))\g)';
   if norm(h)<=epsilon2*(norm(x)+epsilon2)
       found=true;
   else
       if norm(h, 1)<0.8*norm(h)
           if x==0
               eta = delta^2;
           else
               eta = delta*norm(x, 1);
           end
           xnew = x;
           xnew(j) = x(j) + eta;
           ho = (xnew-x)';
           u  = (f(xnew)-f(x)-B*ho)/(ho'*ho);
           B = B + (u*ho')';
       end 
       xnew = x + h;
       u = (f(xnew)-f(x)-B*h)/(h'*h);
       B = B + (u*h')';
       if f(xnew)<f(x)
           x = xnew;
           rho = (fun(x)-fun(xnew))/(0.5*h*(mu*h-g')');
           mu  = mu*max(1/3, 1-(2*rho-1)^3);
           v = 2;
       else
           mu = mu*v;
           v = v*2;
       end
       g = B'*f(x);
       found = (norm(g, Inf)<=epsilon1);
   end 
end 
end
