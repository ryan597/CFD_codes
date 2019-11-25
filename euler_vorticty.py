import numpy as np
import matplotlib.pyplot as plt

def update_vorticity(g, w, u, v, dt, dx, dy, kx, ky):
    # Add the +g and +w for forward euler, then call this func instead of 
    # vorticity_rk4 for faster computation
    gnew = dt*(2*g2_avg(g, dx, dy)- g**2 - convect(g, u, v, kx, ky)) #+g
    wnew = dt*(g*w - convect(w, u, v, kx, ky)) #+w
    return gnew, wnew

def vorticity_rk4(g, w, u, v, dt, dx, dy, kx, ky):
    kg1, kw1 = update_vorticity(g,w,u,v,dt,dx,dy,kx,ky)
    kg2, kw2 = update_vorticity(g+kg1/2,w+kw1/2,u,v,dt,dx,dy,kx,ky)
    kg3, kw3 = update_vorticity(g+kg2/2,w+kw2/2,u,v,dt,dx,dy,kx,ky)
    kg4, kw4 = update_vorticity(g+kg3,w+kw3,u,v,dt,dx,dy,kx,ky)

    g_new = g + 1/6*(kg1+2.*kg2+2.*kg3+kg4)    
    w_new = w + 1/6*(kw1+2*kw2+2*kw3+kw4)

    return g_new, w_new

def update_velocities(g_hat, w_hat, kx, ky, k2):

    uhat = 1j*(kx*g_hat+ky*w_hat)/(k2)
    vhat = 1j*(ky*g_hat-kx*w_hat)/(k2)
    # Assume no zero mode
    Ny, Nx = np.shape(k2)
    uhat[int(Ny/2),int(Nx/2)]=0
    vhat[int(Ny/2),int(Nx/2)]=0

    u = np.real(np.fft.ifft2(np.fft.ifftshift(uhat)))
    v = np.real(np.fft.ifft2(np.fft.ifftshift(vhat)))

    return u, v

def g2_avg(g, dx, dy):
    avg = 1.0/(2*np.pi)**2 * np.sum(g**2)*dx*dy
    return avg

def convect(C, u, v, kx, ky):
    C_hat = np.fft.fftshift(np.fft.fft2(C))
    C_x = np.real(np.fft.ifft2(np.fft.ifftshift(1j*kx*C_hat)))
    C_y = np.real(np.fft.ifft2(np.fft.ifftshift(1j*ky*C_hat)))
    convec_term = (u*C_x + v*C_y)

    return convec_term

def initial_conditions(xx, yy):
    X, Y = np.meshgrid(xx, yy)
    w0 = -np.sin(X) - np.cos(X)*np.cos(Y)
    g0 = np.sin(X)*np.sin(Y) - np.cos(Y)
    return w0, g0

def blowup_test(g):
    # infinity norm
    norm_g = np.max(np.sum(np.abs(g), axis=1))
    if norm_g >= 2**32 -1:
        blowup = True
    else:
        blowup = False
    return blowup

if __name__=="__main__":

    # Grid specifications
    Nx, Ny = 128, 128 
    dt, tfinal = 0.01, 1.5
    n_timesteps = int(np.floor(tfinal/dt))
    print("~~ Euler Vorticity Solver ~~ \n")
    print(f"##### \nParameters: \nGrid points = {Nx}x{Ny}")
    print(f"final time = {tfinal}s")
    # Grid spacing
    dx = 2.*np.pi/Nx
    dy = 2.*np.pi/Ny

    # Discretized grid
    xx = np.arange(0, Nx)*dx
    yy = np.arange(0, Ny)*dy
    print("Setting intial conditions... \n")
    w, g  = initial_conditions(xx, yy)
    w_hat = np.fft.fftshift(np.fft.fft2(w))
    g_hat = np.fft.fftshift(np.fft.fft2(g))

    # Matrices of wavesnumbers
    kx = np.ones((1, Ny)).T * (np.arange(-Nx/2, Nx/2))
    ky = np.reshape(np.arange(-Ny/2, Ny/2), (1, Ny)).T * np.ones((1, Nx))

    k2 = kx**2+ky**2
    k2[int(Nx/2),int(Nx/2)]=1

    dealias = (np.abs(kx) < (2.0/3.0)*(Nx/2.0)) * (np.abs(ky)<(2.0/3.0)*(Ny/2.0))

    print("Entering time loop... \n")
    # Update the vorticity and stretching terms in each timestep
    for iteration_time in range(0, n_timesteps):
        if np.mod(iteration_time, 1)==0:
            seconds = np.round(iteration_time*dt,4)
            print(f"Time: {seconds}s")
            plt.pcolormesh(yy, xx, w.T, cmap="hot")
            plt.colorbar()
            plt.clim(vmin=-2, vmax=2)
            plt.title("2D Euler - Vorticity")
            plt.pause(1e-8)
            plt.clf()
            # Implement numerical consistancy checks...

        u, v = update_velocities(g_hat, w_hat, kx, ky, k2)
        g, w = vorticity_rk4(g, w, u, v, dt, dx, dy, kx, ky)
        blowup = blowup_test(g)
        if blowup:
            print("Solution has blownup. \n")
            print("Exiting loop.")
            plt.pcolormesh(yy, xx, w.T, cmap='hot')
            plt.title("2D Euler - Vorticity")
            plt.colorbar()
            plt.show()
            break
        # dealias to remove
        w_hat = np.fft.fftshift(np.fft.fft2(w))*dealias
        g_hat = np.fft.fftshift(np.fft.fft2(g))*dealias

    print("Simulation finished. \n Showing final plot... \n")
    plt.pcolormesh(yy, xx, w, cmap="hot")
    plt.title("2D Euler - Vorticity")
    plt.colorbar()
    plt.clim(vmin=-2, vmax=2)
    plt.show()