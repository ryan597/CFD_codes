
%iter_v = 1:600;
%wv = 1:600;
%for i=1:600
%    wv(i) = 1.4 + i*0.001;
%    [~,~,~,~,~, iter] = test_poisson_SO(wv(i));
%    iter_v(i)= iter;
%end

%plot(wv, iter_v, 'LineWidth', 3)


function [xx,yy,C,C_true,res_it, iter]=test_poisson_SOR()

% Numerical method to solve 
% [D_{xx}+D_{yy}]C+s(x,y)=0,
% subject to periodic boundary conditions in the x-direction,
% and Neuman boundary conditions at y=0 and y=L_y.
w = 1.82;  
[Nx,Ny,~,~,A0,dx,dy,kx0,ky0]=fix_all_parameters();
iteration_max=1000;

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
    
    C_old = C;
    
    for i=1:Nx
        
        % Periodic BCs here.
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

        for j=2:Ny-1

            diagonal=(2.d0/dx2)+(2.d0/dy2);
            
            C(i,j) = (1-w)*C(i,j) + (w*1.d0/dx2)*(C(ip1,j)+C(im1,j))/diagonal+ ...
                ((w*1.d0/dy2)*(C(i,j+1)+C(i,j-1))+ w * s_source(i,j))/diagonal;
            
        end
    end
        
    % Implement Dirichlet conditions at y=0,y=L_y.
    C(:,1)=C(:,2);
    C(:,Ny)=C(:,Ny-1);
    
    res_it(iteration)=max(max(abs(C-C_old)));
    if(res_it(iteration)<1.d-4)
        iter = iteration;
        %break
    end
    %contourf(C)
semilogy(1:iteration_max, res_it, 'LineWidth', 3)
end
end
