include("fix_all_parameters.jl")
using PyPlot

function test_poisson_jacobi()
    
    Nx, Ny, ~, ~, A0, dx, dy, kx0, ky0 = fix_all_params()
    iteration_max = 5000

    dx2 = dx*dx
    dy2 = dy*dy

    s_source = zeros(Nx, Ny)

    kx = kx0
    ky = 3*ky0

    for i in 1:Nx
        for j in 1:Ny
            x_val = (i-1) * dx
            y_val = (j-1) * dy
            s_source[i, j] = A0*cos(kx*x_val) * cos(ky*y_val)
        end
    end 

    xx = zeros(Nx)
    yy = zeros(Ny)
    C_true = zeros(Nx, Ny)

    for i in 1:Nx
        for j in 1:Ny
            xx[i] = (i-1) * dx
            yy[j] = (j-1) * dy

            C_true[i,j] = (A0/(kx*kx + ky*ky)) * cos(kx*xx[i])*cos(ky*yy[j])
        end
    end

    
    C = zeros(Nx, Ny)

    res_it = zeros(iteration_max)

    for iteration in 1:iteration_max
        
        C_old = copy(C)

        for i in 1:Nx
            if i==1
                im1 = Nx-1
            else
                im1 = i-1
            end

            if i==Nx
                ip1 = 2
            else
                ip1 = i+1
            end

            for j=2:Ny-1
                diagonal = (2/dx2) + (2/dy2)
                temp_val = (1/dx2)*(C_old[ip1, j]+C_old[im1, j]) + 
                    (1/dy2)*(C_old[i, j+1] + C_old[i, j-1]) + s_source[i, j]
                C[i, j] = temp_val/diagonal

            end
        end

        C[:, 1] = C[:, 2]
        C[:, Ny] = C[:, Ny-1]
        res_it[iteration] = maximum(abs.(C-C_old))
    end
    return C, xx, yy, res_it
end

# Disable plotting when not in use (PyPlot import is slow)
"""
function jacobi_plot()
    Z, xx, yy= test_poisson_jacobi()

    contourf(yy, xx, Z)
    title("Poisson solution - Jacobi Method")
    PyPlot.show()
end
"""

#@time test_poisson_jacobi()
#jacobi_plot()
