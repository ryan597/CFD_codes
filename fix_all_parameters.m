function [Nx,Ny,Lx,Ly,A0,dx,dy,kx0,ky0,dt,t_final,n_timesteps]=fix_all_parameters()

aspect_ratio=2;

Ny=101;
Nx=aspect_ratio*(Ny-1)+1;

Ly=1.d0;
Lx=aspect_ratio*Ly;

A0=10.d0;

dx=Lx/(Nx-1);
dy=Ly/(Ny-1);

kx0=2*pi/Lx;
ky0=pi/Ly;

dt=1.d-4;
t_final=0.4d0;
n_timesteps=floor(t_final/dt);

end