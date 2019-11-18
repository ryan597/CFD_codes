function fix_all_params()

    aspect_ratio::Int32=2
    
    Ny::Int32=101
    Nx::Int32=aspect_ratio*(Ny-1)+1

    Ly::Int32=1
    Lx::Int32=aspect_ratio*Ly

    A0::Int32=10

    dx::Float32=Lx/(Nx-1)
    dy::Float32=Ly/(Ny-1)

    kx0::Float64=2*pi/Lx
    ky0::Float64=pi/Ly

    dt::Float32 = 0.0001
    t_final::Float32 = 0.4
    n_timesteps::Int32 = Int(floor(t_final/dt))

    return Nx, Ny, Lx, Ly, A0, dx, dy, kx0, ky0, dt, t_final, n_timesteps
end

