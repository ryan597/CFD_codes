import numpy as np
import matplotlib.pyplot as plt

def vorticity_solve():
    # function parameters
    Nx, Ny = 256, 256
    dt, t_final = 1e-3, 100.0
    n_timesteps = int(np.floor(t_final/dt))

    # Grid spacing
    dx = 2.*np.pi/Nx
    dy = 2.*np.pi/Ny

    nu = 5.9e-30
    nu_damping = 0.05
    A0 = 1.0
    R = 0.9
    kmax, kmin = 9, 7

    # Hyperdiffusivity
    p_hyper = 8

    # Discretized grid
    xx = np.arange(0, Nx)*dx
    yy = np.arange(0, Ny)*dy

    w = initial_vorticity(Nx, Ny, dx, dy)

    # Matrices of wavenumbers
    kx = np.ones((1, Ny)).T * (np.arange(-Nx/2, Nx/2))
    ky = np.reshape(np.arange(-Ny/2, Ny/2), (1, Ny)).T * np.ones((1, Nx))

    # forcing
    k_mod = np.sqrt(kx**2 + ky**2)
    Mx_scale = (k_mod<=kmax) * (k_mod>=kmin)
    forcing_hat = Mx_scale * np.ones((Ny, Nx))
    A0_num = A0*Nx*Ny/(np.sqrt(np.pi*(kmax**2 - kmin**2)))*(np.sqrt(2.0)/1)

    kx = 1j*kx
    ky = 1j*ky

    # Spherical dealiasing
    dealias = (np.abs(kx) < (2.0/3.0)*(Nx/2.0)) * (np.abs(ky)<(2.0/3.0)*(Ny/2.0))
    # Laplacian in fourier space
    k2_visc = kx**2 + ky**2
    k2_poisson = k2_visc
    k2_poisson[int(Nx/2),int(Nx/2)] = 1
    k_hyperdiff = -((-1)**p_hyper)*k2_visc**p_hyper

    w_hat = np.fft.fftshift(np.fft.fft2(w))

    print("Entering time loop... \n")
    # Update vorticity through each time step
    for iteration_time in range(0, n_timesteps):
        if np.mod(iteration_time, 100)==0:            
            seconds = np.round(iteration_time*dt, 4)
            print("Time: {0}s".format(seconds))
            w = np.real(np.fft.ifft2(np.fft.ifftshift(w_hat)))
            plt.imshow(w.T, cmap='hot')
            #plt.pcolormesh(yy, xx, w, cmap="hot")
            if iteration_time==0:
                plt.colorbar()
            plt.title("2D Navier Stokes - Vorticity")
            #plt.pause(1e-7)
            plt.savefig("W_images/vorticity_{0}".format(int(iteration_time/100)))
            # Numerical check for imcompressibility
            """
            if np.mod(iteration_time, 100)==0:
            psi_hat = -w_hat/k2_poisson                 # Solve Poisson's eqn
            u       = np.real(np.fft.ifft2(np.fft.ifftshift( ky*psi_hat)))# u = D_y psi
            v       = np.real(np.fft.ifft2(np.fft.ifftshift(-kx*psi_hat)))# v =-D_x psi
            u_hat   = np.fft.fftshift(np.fft.fft2(u))
            v_hat   = np.fft.fftshift(np.fft.fft2(v))
            u_x     = np.real(np.fft.ifft2(np.fft.ifftshift( kx*u_hat )))
            v_y     = np.real(np.fft.ifft2(np.fft.ifftshift( ky*v_hat )))
            grad_U[int(iteration_time/100)]  = np.abs(u_x + v_y).sum()
            """
        w_hat, forcing_hat = update_vorticity(w_hat, Nx, Ny, dt, kx, ky,
            k2_poisson, k_hyperdiff, dealias, Mx_scale, A0_num, R, nu,
            nu_damping, forcing_hat)
    print("Simulation finished. Showing final plot...")
    # Plot final w
    plt.pcolormesh(yy, xx, w, cmap="hot")
    plt.title("2D Navier Stokes - Vorticity")
    plt.colorbar()
    plt.show()


def update_vorticity(w_hat, Nx, Ny, dt, kx, ky, k2_poisson, k_hyperdiff, dealias,
        Mx_scale, A0_num, R, nu, nu_damping, forcing_hat):
    """
    Poissons equation is solved in Fourier space.
    In Fourier space the derivatives are multiplications of k2
    We perform multiplications in real space and derivative in fourier space.
    Compute the streamfunction, get velocity and gradient of vorticity
    """
    psi_hat = -w_hat/k2_poisson                 # Solve Poisson's eqn
    u       = np.real(np.fft.ifft2(np.fft.ifftshift( ky*psi_hat)))# u = D_y psi
    v       = np.real(np.fft.ifft2(np.fft.ifftshift(-kx*psi_hat)))# v =-D_x psi
    w_x     = np.real(np.fft.ifft2(np.fft.ifftshift( kx*w_hat)))  # x derivative of vorticity
    w_y     = np.real(np.fft.ifft2(np.fft.ifftshift( ky*w_hat)))  # y derivative of vorticity

    conv     = np.add(np.multiply(u,w_x), np.multiply(v,w_y))
    conv_hat = np.fft.fftshift(np.fft.fft2(conv))
    conv_hat = conv_hat * dealias              # Spherical dealiasing

    #Mx_random       = np.random.rand(Ny, Nx)
    #random_part_hat = Mx_scale * np.exp(2*np.pi*1j*Mx_random)
    forcing_hat     = 0#np.sqrt(1-R*R)*A0_num*random_part_hat + R*forcing_hat
    damping_hat     = -nu_damping*w_hat

    # Solution at next step
    w_hat_new = ((1+0.5*dt*nu*k_hyperdiff)/(1-0.5*dt*nu*k_hyperdiff))*w_hat \
        + dt*((conv_hat+forcing_hat+damping_hat)/(1-0.5*dt*nu*k_hyperdiff))

    return w_hat_new, forcing_hat

def initial_vorticity(Nx, Ny, dx, dy):
    """
    Inputs : Number of grid points in the x and y directions (Nx, Ny) and
             the respective grid spcaings (dx, dy)

    Expressions for the vorticities can be changed to give a prefered initialization
    as long as the matrix sizes remain defined over the discretized grid, ie. defined
    by the ii and jj meshgrids.

    Returns : The vorticities w and theta defined over the discretized grid
              defined by the number of grid points (Nx, Ny) and spacing (dx, dy).
              Vorticities will have shape (Nx, Ny).
    """
    print("Generating initial fields...")
    # Defining the Gaussian parameters
    x01 = 0.75*np.pi
    x02 = (5.0/4)*np.pi
    x03 = (5.0/4)*np.pi

    y01 = np.pi
    y02 = np.pi
    y03 = np.pi + (np.pi/(2.0*np.sqrt(2.0)))

    r0  = (1.0/np.pi)**2
    # Meshgrids for the inintialization of the vorticity field
    jj, ii = np.meshgrid(np.arange(1, Nx+1), np.arange(1, Ny+1))
    print("w vorticity")
    w = np.exp(-((ii*dx-x01)**2+(jj*dy-y01)**2)/r0)     \
        + np.exp(-((ii*dx-x02)**2+(jj*dy-y02)**2)/r0)   \
        -0.5*np.exp(-((ii*dx-x03)**2+(jj*dy-y03)**2)/r0)
    w = w*(1.0/(np.pi*r0))
    # theta vorticity...

    print("Initialization complete.\n")
    return w


#def update_data_file(seconds, w):
#    with open('vorticity_data.csv', mode='w') as data_file:
#        vorticity_write = csv.writer(data_file, delimiter=',',
#            quotechar='"', quoting=csv.QUOTE_MINIMAL)
#        vorticity_write.writerow([seconds, w])


vorticity_solve()
