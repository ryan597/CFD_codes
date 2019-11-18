using FFTW
using Dierckx
using PyPlot

# Tidy up and comment function doc string
function vorticity_solve()
    # Function parameters (Move to fix_params?)
    nu_damping  = 0.05
    Nx, Ny      = 256, 256
    dt, t_final = 1e-3, 100.0
    n_timesteps = Int(floor(t_final/dt))

    # Grid spacing is multiple of 2pi for easy FT
    dx = 2*pi/Nx
    dy = 2*pi/Ny

    nu = 5.9e-30
    nuP = nu
    A0, R = 1, 0.9
    kmax, kmin = 9, 7

    # Hyperdiffusivity
    p_hyper = 8

    # Create the discretized grid
    xx = [0:Nx-1...]*dx
    yy = [0:Ny-1...]*dy

    w, theta = initial_vorticity(Nx, Ny, dx, dy)

    # Matrices of wavenumbers with shapes (Ny, Nx)
    kx = ones(1, Ny+1)' *([-Nx/2:Nx/2...])'
    ky = ([-Ny/2: Ny/2...]) * ones(1, Nx+1)

    # Forcing
    k_mod = sqrt.(kx.^2 .+ ky.^2)
    Mx_scale = (k_mod .< kmax) .* (k_mod .> kmin) # Filter
    forcing_hat = Mx_scale.*(ones(Ny,Nx))
    A0_num = A0*Nx*Ny/sqrt(pi*(kmax^2 - kmin^2))*(sqrt(2))

    kx = im*kx
    ky = im*ky
    # ∇^2 in Fourier space
    k2_visc = kx.^2 + ky.^2
    k_hyperdiff = -((-1.0)^p_hyper)*k2_visc.^p_hyper
    k2_poisson  = copy(k2_visc)
    k2_poisson[Int(Nx/2), Int(Nx/2)] = 1      # fixed Laplacian in Fourier space for Poisson

    dealias = (abs.(kx).<(2.0/3.0)*(Nx/2.0)) .* (abs.(ky).<(2.0/3.0)*(Ny/2.0))

    w_hat = fftshift(fft(w))
    println(sum(w_hat))
    theta_hat = fftshift(fft(theta))

    # Initial particle position
    x0, y0 = 0.5*Nx*dx, 0.5*Ny*dy

    println("Entering time loop... \n")
    # Update vorticity through each time step
    for iteration_time in 1:n_timesteps
        break
        w_hat, theta_hat, forcing_hat = update_vorticity!(w_hat, theta_hat, kx, ky, k2_poisson, k_hyperdiff, dealias, dt, Nx, Ny, Mx_scale, R, A0_num, nu, nuP, nu_damping, forcing_hat)
        #theta_hat = copy(theta_hat_new)
        #psi_hat = -w_hat./k2_poisson

        # Particle tracking
#        u = real.(ifft(ky.*psi_hat))
#        v = real.(ifft(-kx.*psi_hat))
#        x0, y0 = update_particle_state!(xx, yy, u, v, dt, x0, y0)

        if mod(iteration_time, 100)==0
            println("Time ", iteration_time*dt, "s  -  ", iteration_time/n_timesteps *100, "% ...")
            w = real.(ifft(ifftshift(w_hat)))
            pcolor(xx, yy, w)
            PyPlot.colorbar()
            title("2D Navier Stokes - Vorticity")
            PyPlot.pause(0.0005)
            PyPlot.clf()
        end
    end
    #PyPlot.show()
end

# Clean function parameters
function update_vorticity!(w_hat, theta_hat, kx, ky, k2_poisson, k_hyperdiff, dealias, dt, Nx, Ny, Mx_scale, R, A0_num, nu, nuP, nu_damping, forcing_hat)
    """
    Poissons equation is solved in Fourier space.
    In Fourier space the derivatives are multiplications of k2
    We perform multiplications in real space and derivative in fourier space.
    Compute the streamfunction, get velocity and gradient of vorticity
    """
    psi_hat = -w_hat./k2_poisson    # Solve Poissons eqn
    u   = real.(ifft(ifftshift( ky.*psi_hat))) # u = D_y psi
    v   = real.(ifft(ifftshift(-kx.*psi_hat))) # v = - D_x psi
    w_x = real.(ifft(ifftshift( kx.*w_hat)))   # x derivative of vorticity
    w_y = real.(ifft(ifftshift( ky.*w_hat)))   # y derivative of vorticity
    theta_x = real.(ifft(ifftshift( kx.*theta_hat))) # x derivative of vorticity
    theta_y = real.(ifft(ifftshift( ky.*theta_hat))) # y derivative of vorticity

    convect     = u.*w_x + v.*w_y
    convect_hat = fftshift(fft(convect))

    convect_theta     = u.*theta_x + v.*theta_y
    convect_theta_hat = fftshift(fft(convect_theta))

    # Spherical dealiasing
    convect_hat = dealias.*convect_hat
    convect_theta_hat = dealias.*convect_theta_hat
    Mx_random       = rand(Ny, Nx)
    random_part_hat = Mx_scale.*exp.(2.0*pi*im*Mx_random)
    forcing_hat     = sqrt(1.0-R*R)*A0_num*random_part_hat + R*forcing_hat
    damping_hat     = -nu_damping*w_hat

    # Solution at next step
    w_hat_new = ((1.0 .+(0.5)*dt*nu*k_hyperdiff)./(1.0 .-(0.5)*dt*nu*k_hyperdiff)).*w_hat + dt*((convect_hat+forcing_hat+damping_hat)./(1.0 .-(0.5)*dt*nu*k_hyperdiff))

    theta_hat_new = ((1.0 .+(0.5)*dt*nuP*k_hyperdiff)./(1.0 .-(0.5)*dt*nuP*k_hyperdiff)).*theta_hat + dt*((convect_theta_hat)./(1.0 .-(0.5)*dt*nuP*k_hyperdiff))

    return w_hat_new, theta_hat_new, forcing_hat

end

function update_particle_state!(xx, yy, u, v, dt, x0, y0)
    u0 = Spline2D(xx, yy, u)(x0, y0)
    v0 = Spline2D(xx, yy, v)(x0, y0)

    x0_new = x0+u0*dt
    y0_new = y0+v0*dt

    return x0_new, y0_new
end

function initial_vorticity(Nx, Ny, dx, dy)
    """
    Inputs : Number of grid points in the x and y directions (Nx, Ny) and
             the respective grid spcaings (dx, dy)

    Expressions for the vorticities can be changed to give a prefered initialization
    as long as the matrix sizes remain defined over the discretized grid.

    Returns : The vorticities w and theta defined over the discretized grid
              defined by the number of grid points (Nx, Ny) and spacing (dx, dy).
              Vorticities will have shape (Nx, Ny).
    """
    println("Generating initial fields...")
    xx = [0:Nx-1...]*dx
    yy = [0:Ny-1...]*dy
    # x0i and y0i for i in {1,2,3} are phase shift parameters
    x01 = (0.75)*pi
    x02 = (5.0/4.0)*pi
    x03 = x02

    y01 = pi
    y02 = pi
    y03 = y02+(pi/(2.0*sqrt(2.0)))

    r0 = (1.0/pi)^2
    # ii, jj are meshgrids to vectorise the vorticity field (w) initialization
    ii, jj = meshgrid([1:Nx...], [1:Ny...])
    println("w vorticity...")
    w = exp.(-((ii*dx .-x01).^2+(jj*dy .-y01).^2)/r0) .+ exp.(-((ii*dx .-x02).^2+(jj*dy .-y02).^2)/r0) .-
        0.5.*exp.(-((ii*dx .-x03).^2+(jj*dy .-y03).^2)/r0)
    w = w*(1.0/(pi*r0))
    println("θ vorticity...")
    theta = zeros(Nx, Ny)
    @inbounds for i in 1:Nx
        @simd for j in 1:Ny
            rad   = (xx[i]-pi).^2 + (yy[j]-pi).^2
            theta[i,j] = (3.0/4.0)*exp(-rad)
        end
    end
    println("Initialization complete. \n")
    return w, theta
end

function meshgrid(x, y)
    """
    Inputs : Two 1 dimensional arrays (x, y)

    Generates the meshgrid from input vectors x and y.

    Returns : Matrices X and Y with shapes (length(x), length(y)), where the
              X matrix has x repeated over its columns and Y matrix has y
              repeated over its rows.

    Example : x = [1 2 3], y = [4 5]
              Then meshgrid(x,y) returns
              X = [1 1; 2 2; 3 3]
              Y = [4 5; 4 5; 4 5]
    """
    nx = length(x)
    ny = length(y)
    X = zeros(nx,ny)
    Y = zeros(nx,ny)
    @inbounds for i in 1:nx
        @inbounds @simd for j in 1:ny
            X[i, j] = x[i]
            Y[i, j] = y[j]
        end
    end
    return X, Y
end

vorticity_solve()
