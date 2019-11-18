% Markov  Chain  Monte  Carlo  is  based on the idea that rather than 
% compute a probability density, say p(μ_{jy}), 
% we would  be  just  as  happy  to  have  a  large  random  sample  from 
% p(μjy) as to know the precise form of the density.  Intuitively, if the 
% sample were large enough,  we  could  approximate  the  form  of  the  
% probability  density  using kernel  density  estimators  or  histograms.  
% In  addition,  we  could  compute accurate measures of central tendency 
% and dispersion for the density, using the mean and standard deviation of 
% the large sample.

% The non-linear least squares model can be written as 
% y=f(X|omega) + epsilon
% epsilon ~ N(0,sigma^2)
% The parameters to be estimated are omega and sigma.

% We assume our prior for the parameter omega is independent from that for
% the parameter sigma

%Set up the the likelihood
fn.like =@(data,time,pars,fn) sum(log(normpdf(nls_poisson_jacobi(pars(1),time),data,sqrt(pars(2)))));
%Set up the the Prior (normal omega and log uniform sigma)
fn.prior=@(pars,path,prior_pars) sum(log(normpdf(pars(1)))+log(1./pars(2)));

niter=10;
pars=zeros(2,niter); % omega and sigma
pars(:,1)= [1.44,.01];
prior_pars=[.01,.01];
stepvar=[.25,.25]';
accepts=0;

Xval = [Xvec,Yvec];

loglike = fn.like(Ydata,Xval,pars(:,1),fn); 
path    = nls_poisson_jacobi(pars(1,1),Xval);
log_alpha_denom = loglike+fn.prior(pars(:,1),path,prior_pars);

niter = 1000;
for  iter=2:niter    
    % Draw from proposal distribution using the previous value (iter-1)
    X=normrnd(pars(:,iter-1),stepvar);    
    % 
    loglike = fn.like(Ydata,Xval,X,fn);  
    [~,~,path]    = test_poisson_jacobiMC(grid_size, X(1,1), aspect_ratio);
    log_alpha_numer = loglike+fn.prior(X,path,prior_pars);
    
    % make a decision
    if(log(rand)<=min(log_alpha_numer - log_alpha_denom, 0))      
        log_alpha_denom=log_alpha_numer;        
        accepts=accepts+1;        
        pars(:,iter)=X;
    else
        pars(:,iter)=pars(:,iter-1);
    end
    
end

histogram(pars(1,:))
omega_hat = mean(pars(1,:));
sd_omega_hat = std(pars(1,:));

path    = nls_poisson_jacobi(omega_hat,Xval);
scatter3(Xvec,Yvec,path,[],path)

accepts/iter 
