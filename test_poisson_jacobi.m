function [xx,yy,C,C_true,res_it]=test_poisson_jacobi()

% Numerical method to solve 
% [D_{xx}+D_{yy}]C+s(x,y)=0,
% subject to periodic boundary conditions in the x-direction,
% and Neuman boundary conditions at y=0 and y=L_y.
  
[Nx,Ny,~,~,A0,dx,dy,kx0,ky0]=fix_all_parameters();
iteration_max=5000;

dx2=dx*dx;
dy2=dy*dy;

s_source=zeros(Nx,Ny);

% Initialise source

kx=kx0;
ky=3*ky0;
  
for i=1:Nx
    for j=1:Ny
      x_val=(i-1)*dx;
      y_val=(j-1)*dy;
      s_source(i,j)=A0*cos(kx*x_val)*cos(ky*y_val);
    end
end

% Compute analytic solution ***********************************************


xx=0*(1:Nx);
yy=0*(1:Ny);
C_true=zeros(Nx,Ny);

for i=1:Nx
    for j=1:Ny
      xx(i)=(i-1)*dx;
      yy(j)=(j-1)*dy;
      
      C_true(i,j)=(A0/(kx*kx+ky*ky))*cos(kx*xx(i))*cos(ky*yy(j));
      
    end
end

% Iteration step **********************************************************
% Initial guess for C:
C=zeros(Nx,Ny);

res_it=0*(1:iteration_max);

for iteration=1:iteration_max
    
    C_old=C;
    
    for i=1:Nx
        
        % Periodic BCs here.
        if(i==1)
            im1(i)=Nx-1;
        else
            im1(i)=i-1;
        end

        if(i==Nx)
            ip1(i)=2;
        else
            ip1(i)=i+1;
        end
        
        C(:, 2:Ny-1) = 1/(dx2)*(C_old(ip1, 2:Ny-1)+C_old(im1, 2:Ny-1)) +...
                1/dy2 * (C_old(:, 3:Ny)+C_old(:,1:Ny-2)) + s_source(:, 2:Ny-1)
            
        %for j=2:Ny-1

            %diagonal=(2.d0/dx2)+(2.d0/dy2);
            %tempval=(1.d0/dx2)*(C_old(ip1,j)+C_old(im1,j))+(1.d0/dy2)*(C_old(i,j+1)+C_old(i,j-1))+s_source(i,j);
            %C(i,j)=tempval/diagonal;
            
        %end
    end
    
        
    % Implement Dirichlet conditions at y=0,y=L_y.
    C(:,1)=C(:,2);
    C(:,Ny)=C(:,Ny-1);
    
    res_it(iteration)=max(max(abs(C-C_old)));
    
end
plot(1:5000, res_it)
end


