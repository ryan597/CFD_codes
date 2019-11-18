function euler_vorticity()

% Fix parameters
Nx = 32;
Ny = 32;
dt = 1e-3;
tfinal = 10.0;
n_timesteps = floor(tfinal/dt);

% Grid spacing
dx = 2.0*pi/Nx;
dy = 2.0*pi/Ny;

% Discretized grid
xx = [0:Nx]*dx;
yy = [0:Ny]*dy;

% Initial conditions on the gird points
% w and gamma are defined on the meshgrid(xx,yy)
[w0, gamma0] = initial_vorticity(xx, yy);

% Wavenumbers
kx = ones(1, Ny)' * [-Nx/2:Nx/2-1];
ky = [-Ny/2:Ny/2-1]' * ones(1, Ny);

% Make complex so 1j isnt needed later
kx = 1i*kx;
ky = 1i*ky;

% dealias?

disp("Entering time loop...");
% Solve the stretching and vorticity equations in each timestep
%for iteration_time = 0:n_timesteps
%   if mod(iteration_time, 100)==0
%       seconds = round(iteration_time*dt, 4);
%       disp(["Time: ", seconds, "s"]);
%       % plotting 
%       set(pcolormesh(Y, X, w), "edgecolor", "None");
%       colormap "hot";
%       colorbar
%       drawnow
%   end
%end
gamma = ode45(@fun, [0,tfinal], gamma0);
for i =0:tfinal
    pcolormesh(Y, X, gamma(i))
    drawnow
disp("Simulation finished.")
end


function dgdt = fun(t,g)
dgdt = 1/(2*pi)^2*(sum(sum(fftshift(fft2(g.^2))))) ...
    -g.^2 - (ux*(ifft2(ifftshift(kx*fftshift(fft2(g))))) ...
    + uy*(ifft2(ifftshift(ky*fftshift(fft2(g))))));
end

function [w0, gamma0] = initial_vorticity(xx, yy)
[X, Y] = meshgrid(xx, yy);
w0 = -sin(X)-cos(X)*cos(Y);
gamma0 = sin(X)*sin(Y)-cos(X);
end