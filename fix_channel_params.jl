function fix_channel_params()

    aspect_ratio = 2

    Ny = 21
    Nz = aspect_ratio*(Ny-1)+1
    Nx = aspect_ratio*(Nz-1)+1

    Ly = 1
    Lz = aspect_ratio*Ly
    Lx = aspect_ratio*Lz

    dx = Lx/(Nx-1)
    dy = Ly/(Ny-1)
    dz = Lz/(Nz-1)

    dt = 0.0001
    t_final = 0.4
    n_timesteps = Int(floor(t_final/dt))

    
    return Nx, Ny, Nz, Lx, Ly, Lz, dx, dy, dz, dt, t_final, n_timesteps
end
