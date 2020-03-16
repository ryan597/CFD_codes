function [C]=nls_poisson_jacobi(omega,X)

grid_size  = 2.*length(unique(X(:,1)));
aspect_ratio = 2;

[xx,yy,C] = test_poisson_jacobiMC(grid_size,omega,aspect_ratio);
grad      = gradient(C,xx(2)-xx(1),yy(2)-yy(1));

[Xvals,Yvals] = meshgrid(xx,yy);

Xvals = reshape(Xvals,size(Xvals,1)*size(Xvals,2),1);
Yvals = reshape(Yvals,size(Yvals,1)*size(Yvals,2),1);
C     = reshape(C,size(C,1)*size(C,2),1);
grad  = reshape(grad,size(grad,1)*size(grad,2),1);

[~,Locb] = ismember(X,[Xvals,Yvals],'rows');

C = C(Locb);
