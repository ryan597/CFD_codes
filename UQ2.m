%% Solve the Poisson equation
Sample_Size  = 200; 
omega        = pi;
aspect_ratio = 2;

[xx,yy,C,C_true,res_it]=test_poisson_jacobiMC(Sample_Size,omega,aspect_ratio);

[X,Y] = meshgrid(xx,yy);

surf(X,Y,C');
%surf(X,Y,(C-C_true)');

%Calculate the Jacobian


%% Set up the data (regular spacing) 
ind_x = 1:4:length(xx);
ind_y = 1:4:length(yy);

[Xm,Ym] = meshgrid(xx(ind_x),yy(ind_y));

plot(X,Y,'ko')
hold on;
plot(Xm,Ym,'ro')

Xvec = reshape(Xm,size(Xm,1)*size(Xm,2),1);
Yvec = reshape(Ym,size(Ym,1)*size(Ym,2),1);
Cvec = reshape(C(ind_x,ind_y),size(ind_x,2)*size(ind_y,2),1);

plot3(Xvec,Yvec,Cvec,'o')

sigma = .01.*max(Cvec);

Ydata = Cvec + sigma.*normrnd(0,1,size(Yvec));

plot3(Xvec,Yvec,Cvec,'ko')
hold on;
plot3(Xvec,Yvec,Ydata,'ro')

%Non-linear least squares (implements Levenberg-Marquardt nonlinear least
%squares algorithm)
beta0   = 0.5;
options = statset('Display','iter');
Xval = [Xvec,Yvec];
[beta,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(Xval,Ydata,@nls_poisson_jacobi,beta0,options);

%Here the Numerical Jacobian and Hessians.
% %Numerical gradient
% Jtest = jacobian_f(beta,@nls_poisson_jacobi,Xval,nls_poisson_jacobi(beta,Xval));
% %or (jacobian_f better for more than one beta) 
% Jtest2 = num_grad(@nls_poisson_jacobi, beta , 1e-4, Xval);
% 
% %Numerical Hessian
% H = num_hess(@nls_poisson_jacobi, beta , 1e-4, Xval);

%Estimated regression coefficients
beta;

%Residuals for the fitted model
figure(2)
subplot(2,1,1)
surf(X,Y,C');
est_C = nls_poisson_jacobi(beta,[Xvec,Yvec]);
subplot(2,1,2)
scatter3(Xvec,Yvec,est_C,[],est_C)

figure(3)
plot(R)

%Estimate the 95% CI for C
[ypred,delta] = nlpredci(@nls_poisson_jacobi,Xval,beta,R,'Covar',CovB,...
                         'MSE',MSE,'SimOpt','on');
scatter3(Xvec,Yvec,ypred)
hold on
scatter3(Xvec,Yvec,ypred+delta,'ro')
scatter3(Xvec,Yvec,ypred-delta,'ro')

%Estimated variance-covariance matrix for the fitted coefficients
ci = nlparci(beta,R,'Jacobian',J);
%or
sdbeta = sqrt(CovB);
[beta - 1.96*sdbeta, beta, beta + 1.96*sdbeta]

%Estimate of the error variance term
MSE

%What happening the optimisation surface
residual = @(omega,X,ydata)(ydata-nls_poisson_jacobi(omega,X));

dx = linspace(-5,5,100);

for i = 1:100
SSE(i) = sqrt(sum(residual(omega+dx(i),Xval,Ydata).^2));
end
figure(4)
plot(omega+dx,SSE)
hold on;
plot(omega,sqrt(sum(residual(omega,Xval,Ydata).^2)),'go')
plot(beta0,sqrt(sum(residual(beta0,Xval,Ydata).^2)),'ro')
plot(beta,sqrt(sum(residual(beta,Xval,Ydata).^2)),'ko')


%% Set up the data (random spacing) 

ran = randsample(size(X,1)*size(X,2),50,true);

Xvec = reshape(X,size(X,1)*size(X,2),1);
Yvec = reshape(Y,size(Y,1)*size(Y,2),1);

plot(X,Y,'ko')
hold on;
plot(Xvec(ran),Yvec(ran),'ro')
