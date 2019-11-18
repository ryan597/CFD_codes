% pseudo spectral method for 3D Euler flow equation

% space grid
Nx = 256;
Ny = 256;
Lx = 2*pi;
Ly = 2*pi;
dx = Lx / Nx;
dy = Ly / Ny;
x = (0:Nx-1)*dx;
y = (0:Ny-1)*dy;

% time grid
dt = 0.001;
T = 1.1;
t = 0:dt:T;
Nt = length(t);

% frequency grid
kx = (2*pi/Lx) * (-Nx/2:Nx/2-1);
ky = (2*pi/Ly) * (-Ny/2:Ny/2-1);
k = sqrt(kx'.^2 + ky.^2);

% initial condition
w = -sin(x') - cos(x').*cos(y); 
g = sin(x').*sin(y) - cos(y);
w_hat = fftshift(fft2(w));
g_hat = fftshift(fft2(g));

% configure plot
figure
set(pcolor(x, y, w), 'edgecolor', 'none')
colormap hot
axis equal
xlim([x(1), x(end)])
ylim([y(1), y(end)])
colorbar
hold on

for i = 2:Nt
    
    % velocity field
    u_hat = 1j*(kx'.*g_hat+ky.*w_hat)./k.^2;
    v_hat = 1j*(ky.*g_hat-kx'.*w_hat)./k.^2;
    
    % no zero mode
    u_hat(Nx/2+1, Ny/2+1) = 0;
    v_hat(Nx/2+1, Ny/2+1) = 0;
    
    % inverse transform
    u = real(ifft2(ifftshift(u_hat)));
    v = real(ifft2(ifftshift(v_hat)));
    
    % average gamma^2 
%     avg_g2 = sum(abs(g_hat).^2, 'all');
    avg_g2 = 1/(4*pi^2)*sum(g.^2, 'all')*dx*dy;
    
    % differentiation in fourier space
    wx = real(ifft2(ifftshift(1j*kx'.*w_hat)));     % dw/dx
    wy = real(ifft2(ifftshift(1j*ky.*w_hat)));      % dw/dy
    gx = real(ifft2(ifftshift(1j*kx'.*g_hat)));     % dg/dx
    gy = real(ifft2(ifftshift(1j*ky.*g_hat)));      % dg/dy
    
    % increment step
    w = dt*(g.*w - u.*wx - v.*wy) + w;
    g = dt*(2*avg_g2 - g.^2 - u.*gx - v.*gy) + g;
    
    % fourier transform
    w_hat = fftshift(fft2(w));
    g_hat = fftshift(fft2(g));
    
    % 2/3 rule dealiasing 
    w_hat(k > 0.66*Nx/2) = 0;
    g_hat(k > 0.66*Nx/2) = 0;
    
    % show
    if mod(i, 100) == 0
        t(i)
        set(pcolor(x, y, w), 'edgecolor', 'none')
        colormap 'hot'
        drawnow
    end
    
end

