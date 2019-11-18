function [xx,yy,C,time_vec,norm_decay]=test_diffusion_SOR()

% Numerical method to solve 
% (d/dt)C=[D_{xx}+D_{yy}]C+s(x,y),
% subject to periodic boundary conditions in the x-direction,
% and Neuman boundary conditions at y=0 and y=L_y.
  
% *************************************************************************
% Simulation parameters

[Nx,Ny,~,~,A0,dx,dy,kx0,ky0,dt,~,n_timesteps]=fix_all_parameters();
w = 1.5;
% max_iteration for diffusion:
max_iteration=30;

% *************************************************************************
% Poisson solution:

 [~,~,C_poiss,~,~]=test_poisson_SOR();
        
% *************************************************************************
% Initialise source and concentration fields.
% Initialize source and concentration fields.

s_source=zeros(Nx,Ny);
C=zeros(Nx,Ny);

xx=0*(1:Nx);
yy=0*(1:Ny);

kx=kx0;
ky=3*ky0;
  
for i=1:Nx
    for j=1:Ny
      x_val=(i-1)*dx;
      y_val=(j-1)*dy;
      
      xx(i)=x_val;
      yy(j)=y_val;
      
      s_source(i,j)=A0*cos(kx*x_val)*cos(ky*y_val);
      C(i,j)=cos(kx0*x_val)*cos(ky0*y_val)+cos(2*kx0*x_val)*cos(ky0*y_val)+cos(kx0*x_val)*cos(4*ky0*y_val);
    end
end

% *************************************************************************  Enter into time loop now
% Enter into time loop now

norm_decay=0*(1:n_timesteps);
time_vec=(1:n_timesteps)*dt;

for iteration_time=1:n_timesteps

    RHS= get_RHS(C,dt,dx,dy,Nx,Ny);
    RHS=RHS+dt*s_source;

    for SOR_iteration=1:max_iteration
        C_old=C;
        
        C=do_SOR_C(C_old,RHS,dt,dx,dy,Nx,Ny, w);

        % Implement Neumann conditions at y=0,y=L_y.
        C(:,1)=C(:,2);
        C(:,Ny)=C(:,Ny-1);
    end

    err1= get_diff(C,C_old,Nx,Ny);
    if(mod(iteration_time,100)==0)
        display(strcat(num2str(iteration_time),': Residual is ...', num2str(err1)))
        [~,myhandle]=contourf(xx,yy,C');
        set(myhandle,'edgecolor','none');
        colorbar
        drawnow
    end
    
    norm_decay(iteration_time)=sqrt(sum(sum(C-C_poiss).^2)/(Nx*Ny));
    
end

end

% *************************************************************************  Enter into time loop now
% *************************************************************************  Enter into time loop now

function RHS=get_RHS(C,dt,dx,dy,Nx,Ny)

ax=dt/(dx*dx);
ay=dt/(dy*dy);

Diffusion=zeros(Nx,Ny);

for j=2:Ny-1
    for i=1:Nx

        if(i==1)
            im1=Nx-1;
        else
            im1=i-1;
        end 

        if(i==Nx)
            ip1=2;
        else
            ip1=i+1;
        end 

        Diffusion(i,j)=ax*( C(ip1,j)+C(im1,j)-2.d0*C(i,j))+ay*( C(i,j+1)+C(i,j-1)-2.d0*C(i,j) );
    end 
end 

RHS=C+0.5d0*Diffusion;

end

% *************************************************************************  Enter into time loop now

function C=do_SOR_C(C_old,RHS,dt,dx,dy,Nx,Ny,w)

ax=dt/(dx*dx);
ay=dt/(dy*dy);
diag_val=1+ax+ay;

C=zeros(Nx,Ny);

for j=2:Ny-1
    for i=1:Nx

    if(i==1)
        im1=Nx-1;
    else
        im1=i-1;
    end 

    if(i==Nx)
        ip1=2;
    else
        ip1=i+1;
    end

    C(i,j)= (1-w)*C_old(i,j) + (w/diag_val)*(ax/2.d0)*(C_old(ip1,j)+C(im1,j))+ ...
        (w/diag_val)*(ay/2.d0)*(C_old(i,j+1)+C(i,j-1))+(w/diag_val)*RHS(i,j);
    
    end 
end 

end

% *************************************************************************  Enter into time loop now

function err1=get_diff(C,C_old,~,~)

err1=max(max(abs(C-C_old)));

end