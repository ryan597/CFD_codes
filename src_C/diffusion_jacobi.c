#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define PI 3.14159265358979323846


void getRHS(const int Nx, const int Ny, double C[][Ny], double RHS[][Ny], double s_source[][Ny],
            const double dt, const double ax, const double ay);

void do_Jacobi(const int Nx, const int Ny, const int n_timesteps, double C[][Ny], double RHS[][Ny], double s_source[][Ny],
                   const double dt, const double  ax, const double ay, const double w, const int max_iteration);

int main()
{
    // timing of program
    double begin = omp_get_wtime();

    // Parameters
    const int Nx = 101;
    const int Ny = 101;
    const int aspect_ratio = 2;
    const int Ly = 1;
    const int Lx = aspect_ratio * Ly;
    const double A0 = 1.0;
    const double dx = Lx / (Nx + 1);
    const double dy = Ly / (Ny + 1);
    const double kx = 2 * PI / Lx;
    const double ky = PI / Ly;
    const double dt = 1.0E-3;
    const double t_final = 0.4;
    const int n_timesteps = floor(t_final/dt);
    const double w = 1.5;
    const int max_iteration = 30;
    double xval;
    double yval;

    double (*s_source)[Ny] = malloc(sizeof(*s_source) * Nx);
    double (*C)[Ny] = malloc(sizeof(*C) * Nx);
    double (*RHS)[Ny] = malloc(sizeof(*RHS) * Nx);

    const double ax = dt / (dx * dx);
    const double ay = dt / (dy * dy);

    int i, j;
    // Forcing term and initial condition
#pragma omp parallel shared(s_source, C, RHS, A0, kx, ky) private(xval, yval, i, j)
{
    #pragma omp for schedule(static)
    for (i = 0; i < Nx; i++)
    {
        for (j = 0; j < Ny; j++)
        {
            xval = i * dx;
            yval = j * dy;

            s_source[i][j] = A0 * cos(kx * xval) * cos(ky * yval);
            C[i][j] = cos(kx * xval) * cos(ky * yval) + \
                      cos(2 * kx * xval) * cos(ky * yval) +\
                      cos(kx * xval) * cos(4 * ky * yval);
            RHS[i][j] = 0;
        }
    }
}

    do_Jacobi(Nx, Ny, n_timesteps, C, RHS, s_source, dt, ax, ay, w, max_iteration);

    free(s_source);
    free(C);
    free(RHS);

    double end = omp_get_wtime();
    double time_spent = end - begin;
    printf("program time :  %lf \n", time_spent);

    return 0;
}


void getRHS(const int Nx, const int Ny, double C[][Ny], double RHS[][Ny], double s_source[][Ny],
            const double dt, const double ax, const double ay)
{
    int ip1, im1, i, j;
    double Diffusion;
#pragma omp parallel shared(RHS, Ny, Nx, C, s_source) private(ip1, im1, Diffusion, i, j) 
{
#pragma omp for schedule(static)
    for (i = 0; i < Nx; i++)
    {
        if (i == 0) {im1 = Nx - 1;} else {im1 = i - 1;}
        if (i == Nx-1){ip1 = 0;} else {ip1 = i + 1;}
  	
	for (j = 1; j < Ny-1; j++)
        {
           // Centered differences
            Diffusion = ax * (C[ip1][j]  + C[im1][j] - 2 * C[i][j]) +\
                        ay * (C[i][j+1] + C [i][j-1] - 2 * C[i][j]);

            RHS[i][j] = C[i][j] + 0.5 * Diffusion + dt * s_source[i][j];
        }
    }
}
}

void do_Jacobi(const int Nx, const int Ny, const int n_timesteps, double C[][Ny], double RHS[][Ny], double s_source[][Ny],
                   const double dt, const double  ax, const double ay, const double w, const int max_iteration)
{
    const float diag = 1 + ax + ay;
    int ip1, im1, i, j, tid;

    for(int t = 0; t < n_timesteps; t++)
    {
        getRHS(Nx, Ny, C, RHS, s_source, dt, ax, ay);
        for (int SORiter = 0; SORiter < max_iteration; SORiter++)
        {
#pragma omp parallel shared(Nx, Ny, C, RHS, ax, ay, diag, t) private(ip1, im1, i, j, tid)
{
    tid = omp_get_thread_num();
#pragma omp for schedule(static) collapse(2)
            for (i = 1; i < Nx; i++)
            {
               for (j = 1; j < Ny-1; j++)
                {
		    if (i == 0) {im1 = Nx - 1 ;} else {im1 = i - 1;}
                    if (i == Nx - 1){ip1 = 0 ;} else {ip1 = i + 1;}
                
                    // Crank-Nicholson with SOR
                    C[i][j] = (ax / 2) * (C[ip1][j] + C[im1][j]) / diag + 
                              (ay / 2) * (C[i][j+1] + C[i][j-1]) / diag +
                              (1 / diag) * RHS[i][j];
                    // Neumann BCs
                    if (j == 0)  {C[i][0] = C[i][2];}
                    if (j == Ny) {C[i][Ny] = C[i][Ny-1];}
                }
            }
        }
        }
    }
}

