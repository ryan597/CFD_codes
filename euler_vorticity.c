#include <stdio.h>
#include <math.h>
#include <fftw3.h>

void initial_conditions(const int Nx, const int Ny, double xx[], double yy[],
                        long double *w0, long double *g0);
void ift_scaling(int Nx, int Ny, long double *x);


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
    const int Nx = 8;
    const int Ny = 8;
    const double dt = 0.01;
    const double tfinal = 1.17;
    const int n_timesteps = floor(tfinal/dt);
    printf("~~ Euler Vorticity Solver ~~ \n");
    printf("Parameters......\n");
    printf("Grid points = %d x %d \n", Nx, Ny);
    printf("Final time = %fs \n", tfinal);
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
    
    long double w[Nx][Ny];
    long double g[Nx][Ny];
    fftwl_complex w_hat[Nx][Ny/2+1];
    fftwl_complex g_hat[Nx][Ny/2+1];

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

    //Test FFT***
    fftwl_execute(fft);
    fftwl_execute(ift);
    ift_scaling(Nx, Ny, *w);
    print_array(Nx, Ny, *w);
    return 0;
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