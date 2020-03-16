#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <fftw3.h>

void initial_conditions(const int Nx, const int Ny, double xx[], double yy[],
                        long double *w0, long double *g0);
void ift_scaling(int Nx, int Ny, long double *x);

int blowup_test(long double *g, int Nx, int Ny);

// print to test shits working
void print_array(const int Nx, const int Ny, long double *a){
    for(int i=0;i<Nx;i++){
        for(int j=0;j<Ny;j++){
            printf("%Le ", *((a+i*Ny)+j));
        }
        printf("\n");
    }
}

int main(){
    // Grid specifications
    const int Nx = 256;
    const int Ny = 256;
    const double dt = 0.001;
    const double tfinal = 2;
    const int n_timesteps = floor(tfinal/dt);
    printf("~~ Euler Vorticity Solver ~~ \n");
    printf("Parameters......\n");
    printf("Grid points = %d x %d \n", Nx, Ny);
    printf("Final time = %fs \n", tfinal);
    printf("Number of timesteps : %d \n", n_timesteps);
    // Grid Spacings
    const double dx = 2*M_PI/Nx;
    const double dy = 2*M_PI/Ny;
    // Discretized grid
    double xx[Nx];
    for(int i=0;i<Nx;i++){
        xx[i]=i*dx;
    }
    double yy[Ny];
    for(int i=0;i<Ny;i++){
        yy[i]=i*dy;
    }
    // *************************************************************************
    int sizeFT = Nx*(Ny/2+1)*sizeof(fftwl_complex);
    
    int kx[Nx];
    int ky[Ny/2+1];
    int k2[Nx][Ny/2+1];
    int dealias[Nx][Ny/2+1];

    long double w[Nx][Ny];
    long double g[Nx][Ny];
    long double avg_g2;

    long double u[Nx][Ny]; 
    long double v[Nx][Ny];
    
    long double wx[Nx][Ny];
    long double wy[Nx][Ny];
    long double gx[Nx][Ny];
    long double gy[Nx][Ny];

    fftwl_complex *u_hat  = (fftwl_complex *)fftwl_malloc(sizeFT);
    fftwl_complex *v_hat  = (fftwl_complex *)fftwl_malloc(sizeFT);
    fftwl_complex *w_hat  = (fftwl_complex *)fftwl_malloc(sizeFT);
    fftwl_complex *g_hat  = (fftwl_complex *)fftwl_malloc(sizeFT);
    fftwl_complex *wx_hat = (fftwl_complex *)fftwl_malloc(sizeFT);
    fftwl_complex *wy_hat = (fftwl_complex *)fftwl_malloc(sizeFT);
    fftwl_complex *gx_hat = (fftwl_complex *)fftwl_malloc(sizeFT);
    fftwl_complex *gy_hat = (fftwl_complex *)fftwl_malloc(sizeFT);

    // *************************************************************************
    // Plan the FFT & IFFT to increase performance in long run
    printf("Planning fast Fourier transform... \n");
    fftwl_plan fft;
    // Point to the memory address of the array to FFT
    long double *in_fft;
    // Point to memory address of array to store FFT
    fftwl_complex *out_fft;
    in_fft = *w;
    out_fft = *w_hat;
    fft = fftwl_plan_dft_r2c_2d(Nx, Ny, in_fft, out_fft, FFTW_MEASURE);
    printf("Planning inverse fast Fourier transform... \n");
    fftwl_plan ift;
    // Point to memory address of array to IFFT
    fftwl_complex *in_ift;
    // Point to memory address of array to store IFFT
    long double *out_ift;
    in_ift = *w_hat;
    out_ift = *w;
    ift = fftwl_plan_dft_c2r_2d(Nx, Ny, in_ift, out_ift, FFTW_MEASURE);
    // *************************************************************************
    printf("Setting initial conditions... \n");
    initial_conditions(Nx, Ny, xx, yy, *w, *g);

    // Perform FFT on w, FFT is stored in w_hat 
    fftwl_execute(fft);

    // Matrices of wavenumbers
    for(int i=0;i<Nx;i++) {
        kx[i] = i;
        for(int j=0;j<Ny/2+1;j++){
            ky[j] = j;
            k2[i][j] = kx[i]*kx[i] + ky[j]*ky[j];
            if(kx[i]<2/3*(Nx/2) && ky[j]<2/3*(Ny/2)) {dealias[i][j] = 1;}
        }
    }
    // Prevent dividing by 0 when solving for velocities in Fourier space
    k2[0][0] = 1;
    // Initialize the velocities

    printf("Entering time loop... \n");
    for(int iteration_time=0;iteration_time<n_timesteps;iteration_time++){
        printf("%d \n", iteration_time);
        for(int i=0;i<Nx;i++){
            for(int j=0;j<Ny/2+1;j++){
                // Real and complex parts stored in the [0] and [1] parts
                u_hat[i][j][1] = (kx[i]* *(g_hat+i*(Ny/2+1)+j+0)+ky[j]* *(w_hat+i*(Ny/2+1)+j+0) )/k2[i][j];
                v_hat[i][j][1] = (ky[i]*g_hat[i][j][0]-kx[j]*w_hat[i][j][0])/k2[i][j];
                u_hat[i][j][0] = -(kx[i]*g_hat[i][j][1]+ky[j]*w_hat[i][j][1])/k2[i][j];
                v_hat[i][j][0] = -(ky[i]*g_hat[i][j][1]-kx[j]*w_hat[i][j][1])/k2[i][j];
            }
        }
        // No zero mode
        u_hat[0][0][0] = 0;
        u_hat[0][0][1] = 0;
        v_hat[0][0][0] = 0;
        v_hat[0][0][1] = 0;
        // Inverse transform and scaling of velocities
        in_ift = *u_hat;
        out_ift = *u;
        fftwl_execute(ift);
        in_ift = *v_hat;
        out_ift = *v;
        fftwl_execute(ift);
        ift_scaling(Nx, Ny, *u);
        ift_scaling(Nx, Ny, *v);

        // Spatial average of gamma^2
        avg_g2 = 0;
        for(int i=0;i<Nx;i++){
            for(int j=0;j<Ny;j++){
                avg_g2 += (g[i][j])*(g[i][j]);
            }
        }
        avg_g2 *= dx*dy/(4*M_PI*M_PI);
        // Calculate derivatives for convection terms
        for(int i=0;i<Nx;i++){
            for(int j=0;j<Ny/2+1;j++){
                wx_hat[i][j][0] = kx[i]*w_hat[i][j][1];
                wx_hat[i][j][1] = -kx[i]*w_hat[i][j][0];
                wy_hat[i][j][0] = ky[j]*w_hat[i][j][1];
                wy_hat[i][j][1] = -ky[j]*w_hat[i][j][0];

                gx_hat[i][j][0] = kx[i]*g_hat[i][j][1];
                gx_hat[i][j][1] = -kx[i]*g_hat[i][j][0];
                gy_hat[i][j][0] = ky[j]*g_hat[i][j][1];
                gy_hat[i][j][1] = -ky[j]*g_hat[i][j][0];
            }
        }
        in_ift = *wx_hat;
        out_ift = *wx;
        fftwl_execute(ift);

        in_ift = *wy_hat;
        out_ift = *wy;
        fftwl_execute(ift);

        in_ift = *gx_hat;
        out_ift = *gy;
        fftwl_execute(ift);

        in_ift = *gy_hat;
        out_ift = *gy;
        fftwl_execute(ift);
        ift_scaling(Nx, Ny, *wx);
        ift_scaling(Nx, Ny, *wy);
        ift_scaling(Nx, Ny, *gx);
        ift_scaling(Nx, Ny, *gy);

        // Update vorticity in real space
        for(int i=0;i<Nx;i++){
            for(int j=0;j<Ny;j++){
                w[i][j] = dt*(g[i][j]*w[i][j]-u[i][j]*wx[i][j]-v[i][j]*wy[i][j])+w[i][j];
                g[i][j] = dt*(2*avg_g2-g[i][j]*g[i][j]-u[i][j]*gx[i][j]-v[i][j]*gy[i][j])+g[i][j];
            }
        }
        int blowup = blowup_test(*g, Nx, Ny);
        if(blowup==1){
            float time = iteration_time * dt;
            printf("Solution has blownup at T* = %f", time);
            break;
        }

        // Update the FFTs and dealiasing
        in_fft = *w;
        out_fft = *w_hat;
        fftwl_execute(fft);
        in_fft = *g;
        out_fft = *g_hat;
        fftwl_execute(fft);
        for(int i=0;i<Nx;i++){
            for(int j=0;j<Ny/2+1;j++){
                w_hat[i][j][0] = w_hat[i][j][0]*dealias[i][j];
                w_hat[i][j][1] = w_hat[i][j][1]*dealias[i][j];
                g_hat[i][j][0] = g_hat[i][j][0]*dealias[i][j];
                g_hat[i][j][1] = g_hat[i][j][1]*dealias[i][j];
            }
        }
    }

    fftwl_free(w_hat);
    fftwl_free(g_hat);
    fftwl_free(u_hat);
    fftwl_free(v_hat);
    fftwl_free(wx_hat);
    fftwl_free(wy_hat);
    fftwl_free(gx_hat);
    fftwl_free(gy_hat);
    return 0;
}

int blowup_test(long double *g, int Nx, int Ny){
    long double norm_g[Nx];
    int blowup = 0;
    // Infinity norm
    for(int i=0;i<Nx;i++){
        for(int j=0;j<Ny;j++){
            norm_g[i] += *((g+i*Ny)+j);
        }
        if(norm_g[i]>= pow(2, 32)-1){
            blowup = 1;
            break;
        }
    }
    return blowup;
}

void initial_conditions(const int Nx, const int Ny, double xx[], double yy[],
                        long double *w0, long double *g0){
    // Passing in pointers *w0 and *g0 of the respective arrays.
    // Points to 1st element, then *((w0+i*Ny)+j) iterates over the values.
    for(int i=0;i<Nx;i++){
        for(int j=0;j<Ny;j++){
            *((w0+i*Ny)+j) = -sin(xx[i]) - cos(xx[i]) * cos(yy[j]);
            *((g0+i*Ny)+j) =  sin(xx[i]) * sin(yy[i]) - cos(yy[j]);
        }
    }
}

void ift_scaling(int Nx, int Ny, long double *x){
    for(int i=0;i<Nx;i++){
        for(int j=0;j<Ny;j++){
            *((x+i*Ny)+j) = *((x+i*Ny)+j)/(Nx*Ny);
        }
    }
}