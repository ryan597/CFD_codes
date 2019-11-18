include("fix_all_parameters.jl")
#using PyPlot

function test_poisson_SOR(w=1.8)
    
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
    last_iter=5000
    @fastmath for iteration in 1:iteration_max

        C_old = copy(C)
        
        im1 = 0
        ip1 = 0
        @inbounds for i in 1:Nx
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

            @inbounds for j=2:Ny-1
                diagonal = (2/dx2) + (2/dy2)
                C[i, j] = (1-w)*C[i,j] + (w/dx2)*(C[ip1, j]+C[im1, j])/diagonal + 
                    (w/dy2)*(C[i, j+1] + C[i, j-1])/diagonal + (w/diagonal)*s_source[i, j]
        

            end
        end
        

        C[:, 1] = C[:, 2]
        C[:, Ny] = C[:, Ny-1]

        
        res_it[iteration] = maximum(abs.(C-C_old))
        if res_it[iteration] < 1e-5
            last_iter = iteration
            break
        end
    end
    return C, xx, yy, last_iter
end

# PyPlot import significantly slows program,
# Disabled for when not in use
"""
function SOR_plot()
    Z, xx, yy = test_poisson_SOR()

    contourf(yy, xx, Z)
    title("Poisson solution - SOR Method")
    PyPlot.show()
end
"""

#@time test_poisson_SOR()
#SOR_plot()


# Determine the best w for SOR
"""
last_iteration = 5000*ones(61)
for i in 1:60
    w =  1.4 + (i-1)*0.01
    ~,~,~, last_iteration[i] = test_poisson_SOR(w)

    plot(w, last_iteration[i], marker="o")
    PyPlot.pause(0.0005)
end
PyPlot.show()
"""