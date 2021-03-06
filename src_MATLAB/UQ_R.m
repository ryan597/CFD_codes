%% Function/sampling parameters
grid_size    = 50;         % grid points in y
aspect_ratio = 2;          % grid points in x is aspect_ratio*grid_size-1
omega        = pi;         % True omega
sigma        = 0.05;        % Noise varience parameter
method       = "random";  % Sampling method can be uniform or random
samples      = 20;          % grid spacing for uniform (eg 4 for every 4th grid point) or number of samples for each direction (eg 20 gives 20x20 samples)
%beta0        = 3.5;        % beta0 is the initial guess
algorithm    = "LM";        % GN=Gauss-Newton, LM=LevMarq
runs         = 1;         % number of times to start with random beta0
%% Analysis

[Xval, Ydata, ind_x, ind_y] = objectiveSampling(grid_size, aspect_ratio, omega, sigma, method, samples);

% Simulated anneling for beta
[beta_SA, beta_array, SA_vals] = simulatedAnnel(Ydata, grid_size, aspect_ratio, ind_x, ind_y);
beta0 = mean(beta_array(500:1000));
% Use result from Sim Annel to start optimisation
[beta, MSE, beta0, xvals, fvals] = guessBeta(runs, algorithm, beta0, Ydata, grid_size, aspect_ratio, ind_x, ind_y);

display([beta, MSE, beta0])
% Recalculate full solutions for plotting
[~,~,C]        = test_poisson_SOR_MC(grid_size, omega, aspect_ratio);
[xx,yy,best_C] = test_poisson_SOR_MC(grid_size, beta, aspect_ratio);


[X, Y] = meshgrid(xx, yy);
% Plotting the true solution and optimised solution
figure(1)
subplot(2, 1, 1)
surf(X, Y, C' + 0.01*normrnd(0, 1, (size(C'))));
title(sprintf("True solution - \\omega= \\pi, \\sigma=%s", num2str(sigma)))
subplot(2, 1, 2)
surf(X, Y, best_C')
hold on 
plot3(X(ind_y, ind_x), Y(ind_y,ind_x), best_C(ind_x, ind_y)', 'ro', "markerfacecolor", "r")
title(sprintf("Best solution %s - \\omega=%s, samples=%s", [algorithm, num2str(beta),num2str(samples)]))
hold off
%saveas(1, sprintf("Poisson_%s_samples_%s_sigma_%s_beta0_%s.jpg", [algorithm, samples, sigma, beta0]))


%[t, cost] = detailCost(Ydata, grid_size, aspect_ratio, ind_x, ind_y);
figure(2)
subplot(2, 1, 1)
semilogy(t, cost)
hold on
semilogy(beta_array, SA_vals, 'ro')
semilogy(xvals, fvals, 'go', 'LineWidth', 2)
title("SA and LM - Semilog plot of cost function vs \\omega")
xlabel("\omega")
ylabel("Log(MSE)")
semilogy(beta, MSE, 'ko', 'LineWidth', 2)
hold off
subplot(2, 1, 2)
plot(t, cost)
hold on 
plot(beta_array, SA_vals, 'ro')
plot(xvals, fvals, 'go', 'LineWidth', 2)
title("SA and LM - Plot of cost function vs \\omega")
xlabel("\omega")
ylabel("Cost - RMSE")
plot(beta, MSE, 'ko', 'LineWidth', 2)
hold off
%saveas(2, sprintf("CostDescent_%s_samples_%s_sigma_%s_beta0_%s.jpg", [algorithm, samples, sigma, beta0]))


figure(3)
j=0;
for i = 0:1000
plot(t, (exp(-cost)/max(exp(-cost))).^(i/100), 'LineWidth', 3)
title("Simulated Annealing of Cost Function")
xlabel("\omega")
ylabel("Normalised exp(-cost)")
pause(0.01)

if ismember(i, [0, 1, 2, 5, 10, 100, 500, 1000])
    saveas(3, sprintf("Simulated_Annel_cost_%s.jpg", num2str(j)))
    j = j+1;
end
end


%% Function definitions
function [beta, MSE, best_beta0, xvals, fvals] = guessBeta(runs, algorithm, beta0, Ydata, grid_size, aspect_ratio, ind_x, ind_y)
    MSE = inf;
    for i=1:runs
        if runs==1
            beta0 = beta0 + normrnd(0, 0.3, 1);
        else
            beta0 = rand(1)*5;
        end
        [new_beta, new_MSE, new_xvals, new_fvals] = optimization(algorithm, beta0, Ydata, grid_size, aspect_ratio, ind_x, ind_y);
        if new_MSE<MSE
            MSE = new_MSE;
            beta= new_beta;
            best_beta0=beta0;
            xvals = new_xvals;
            fvals = new_fvals;
        end
    end
end

function [Xval, Ydata, ind_x, ind_y] = objectiveSampling(grid_size, aspect_ratio, omega, sigma, method, samples)
% Solve Poissons equation
[xx,yy,C]=test_poisson_SOR_MC(grid_size,omega,aspect_ratio);

if (method=="uniform")
    % If method is uniform then samples corresponds to the step size
    % Generate the sampling indices
    ind_x = 1:samples:length(xx);
    ind_y = 1:samples:length(yy);
elseif (method=="random")
    % If method is random then samples corresponds to the number of points
    ind_x = randsample(length(xx), samples, true)';
    ind_y = randsample(length(yy), samples, true)';
end

% Meshgrid of the sampled points
[Xm,Ym] = meshgrid(xx(ind_x),yy(ind_y));
% Reshape into vectors for taking derivatives
Xvec = reshape(Xm,size(Xm,1)*size(Xm,2),1);
Yvec = reshape(Ym,size(Ym,1)*size(Ym,2),1);
Xval = [Xvec,Yvec];
Cvec = reshape(C(ind_x,ind_y),size(ind_x,2)*size(ind_y,2),1);
% Adding Gaussian noise to the true solution
sigma = sigma*max(Cvec);
Ydata = Cvec + sigma.*normrnd(0,1,size(Yvec));
end

function [beta, MSE, xvals, fvals] = optimization(algorithm, beta0, Ydata, grid_size, aspect_ratio, ind_x, ind_y)
    % Return the optimized parameter beta after implementating chosen
    % algorithim
    function MSE = fun(t)
        % Generate the new predictions of the model for the new t parameter and
        % return in vector form
        [~,~,C] = test_poisson_SOR_MC(grid_size, t, aspect_ratio);
        Cvec = reshape(C(ind_x,ind_y),size(ind_x,2)*size(ind_y,2),1);
        MSE = norm(Ydata-Cvec);
    end
    switch algorithm
        case {"GN"}
            [beta, MSE, xvals, fvals] = gauss_newton(beta0,  @fun);
        case {"LM"}
            [beta, MSE, xvals, fvals] = lev_marq(beta0, @fun);
    end
end

function [t, cost] = detailCost(Ydata, grid_size, aspect_ratio, ind_x, ind_y)
    % Used for plotting the cost function and the steps taken along it
    step = 0.001;
    t = 0:step:5;                   % range to plot cost function over
    cost = zeros(1, length(t));     % vector to store the cost for each value
    for i=1:length(t)               % loop over range and calculate cost
        [~,~,z] = test_poisson_SOR_MC(grid_size, i*step, aspect_ratio);
        z  = reshape(z(ind_x,ind_y),size(ind_x,2)*size(ind_y,2),1);
        cost(1, i) = norm(Ydata-z); % evaluated at the indices of sampled points 
    end                             % ind_x and ind_y
end


function [beta, beta_array, SA_vals] = simulatedAnnel(Ydata, grid_size, aspect_ratio, ind_x, ind_y)
    % Initialize beta
    beta = rand(1)*2*pi;
    beta_array = [];
    SA_vals = [];
    Told = 5;
    max_iter = 1000;
    function MSE = fun(t)
        % Calculate the cost at t
        [~,~,C] = test_poisson_SOR_MC(grid_size, t, aspect_ratio);
        Cvec = reshape(C(ind_x,ind_y),size(ind_x,2)*size(ind_y,2),1);
        MSE = (norm(Ydata-Cvec));
    end
    fun_old = fun(beta);
    
    % Iterate through increasing temp
    for i = 1:max_iter
        T = log(Told+1);
        % Draw from proposal
        beta_proposal = normrnd(beta, T, 1);
        % Acceptance step
        fun_prop = fun(beta_proposal);
        alpha = exp((fun_old-fun_prop)/T);
        
        if (rand(1)<alpha)
            beta_array = [beta_array, beta_proposal];
            fun_old = fun_prop;
            SA_vals = [SA_vals, fun_prop];
            beta = beta_proposal;
        else
            beta_array = [beta_array, beta];
            SA_vals = [SA_vals, fun_old];
        end
        Told = T;
    end
end
