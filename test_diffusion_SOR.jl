include("fix_all_parameters.jl")
include("poisson_SOR.jl")
using PyPlot


function diffusion_SOR(w)
    Nx, Ny, ~, ~, A0, dx, dy, kx0, ky0, dt, ~, n_timesteps = fix_all_params()
    max_iteration = 30
    
    C_poiss, ~, ~ = test_poisson_SOR()

    s_source = zeros(Nx, Ny)
    C = zeros(Nx, Ny)

    xx = zeros(Nx)
    yy = zeros(Ny)

    kx = kx0
    ky = 3*ky0

    @fastmath @inbounds for i in 1:Nx
        @inbounds for j in 1:Ny
            x_val = (i-1)*dx
            y_val = (j-1)*dy

            xx[i] = x_val
            yy[j] = y_val

            s_source[i, j] = A0 * cos(kx*x_val)*cos(ky*y_val)
            C[i, j] = cos(kx0*x_val)*cos(ky0*y_val)+cos(2*kx0*x_val)*cos(ky0*y_val)+
                        cos(kx0*x_val)*cos(4*ky0*y_val)
        end
    end

    norm_decay = zeros(n_timesteps)
    time_vec = dt*[1:n_timesteps]

    C_old = zeros(Nx, Ny)

    @fastmath @inbounds for iteration_time in 1:n_timesteps

        RHS = get_RHS(C, dt, dx, dy, Nx, Ny)
        RHS = RHS + dt*s_source

        @inbounds for SOR_iteration in 1:max_iteration
            C_old = C

            C = do_SOR_C(C_old, RHS, dt, dx, dy, Nx, Ny, w)

            C[:, 1] = C[:, 2]
            C[:, Ny] = C[:, Ny-1]
        end

        err1 = get_diff(C, C_old, Nx, Ny)

        if mod(iteration_time, 100)==0
            contourf(yy, xx, C)
            title("Diffusion solution - SOR Method")
            PyPlot.pause(0.005)
        end

        norm_decay[iteration_time] = sqrt(sum(sum(C-C_poiss, dims=1).^2))
    end
end


function get_RHS(C, dt, dx, dy, Nx, Ny)

    ax = dt/(dx*dx)
    ay = dt/(dy*dy)

    Diffusion  = zeros(Nx, Ny)

    im1 = 0
    ip1 = 0
    @fastmath @inbounds for j in 2:Ny-1
        @inbounds for i in 1:Nx

            if i==1
                im1 = Nx-1
            else
                im1 = i-1
            end

            if i == Nx
                ip1 = 2
            else 
                ip1 = i+1
            end 

            Diffusion[i, j] = ax*( C[ip1, j]+C[im1, j] - 2*C[i, j]) +
                    ay*( C[i, j+1] + C[i, j-1] - 2* C[i, j])
        end
    end 

    RHS = C + 0.5*Diffusion
    return RHS
end 


function do_SOR_C(C_old, RHS, dt, dx, dy, Nx, Ny, w)

    ax = dt/(dx*dx)
    ay = dt/(dy*dy)
    diag_val = 1 + ax + ay

    C = C_old

    im1 = 0
    ip1 = 0
    @fastmath @inbounds for j in 2:Ny-1
        @inbounds for i in 1:Nx

            if i==1
                im1=Nx-1
            else 
                im1=i-1
            end

            if i ==Nx
                ip1 = 2
            else
                ip1 = i+1
            end

            temp_val = (ax/2)*(C[ip1, j]+C[im1, j]) + (ay/2)*(C[i, j+1] + C[i, j-1])
            pre = (1-w)*C[i,j]
            C[i, j] = pre + (w*temp_val)/diag_val + (w/diag_val)*RHS[i, j]
        end
    end

    return C
end

function get_diff(C, C_old, Nx, Ny)
    err1 = maximum(abs.(C-C_old))
    return err1
end

#@time diffusion_SOR(w = 1.8)