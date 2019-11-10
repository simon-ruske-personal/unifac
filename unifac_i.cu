// ===========================================
// NOTE : WILL NEED TO BE COMPILED WITH CUBLAS
// Specifically I have used the following 
// command to command the code : 
// > nvcc unifac_u.cu -o unifac.out -l cublas
// ===========================================

// ===========================================
// Notes on the implementation 
// ===========================================

// =====================
// 1) THE COMBINITAROIAL
// =====================

// The combinitaorial part can be calculated in the following way 
// 1.1) reduce UFC_Data_Q and UFC_Data_R to only the values that are needed 
//    which are then stored in Q and R. This is relatively easy and is 
//    done on the host. 
// 1.2) multiply V by Q and V by R, this gives us q_i and v_i which we 
//    denote by q and v. These are matrix multiplications that can be 
//    done on the device using cublas. 
// 1.3) calculate l. This can be done in a simple linear fashion using 
//    calc_L_GPU.
// 1.4) calculate dot products (x, r), (x, q), (x, l). This can be done on 
//    the device using cublas. 
// 1.4) calculate psi/x. This can be done on the device using divideGPU
// 1.5) calculate theta/psi. Again use divideGPU, but also need to divideGPU
//    by a scalar so requires divideGPU_scalar also. 
// 1.6) calculate ln_gamma_c. This is done using combinitorial_GPU

// To be able to be used with cublas I have 
// rewritten the kernel function as follows. 
// Here psi_t indicates the transpose of 

// 1) thePsi = Theta * Psi (matrix mult.)
// 2) thediv = Theta / thePsi (elementwise div)
// 3) thePsi_t = thediv * Psi_t (matrix mult.)
// 4) GPU result = Q[k] *(1 - log(thePsi) - thePsi_t)

// 1 & 3 are executed with CUBLAS.
// 2 & 4 are executed with self writen functions

// ==================
// INCLUDE STATEMENTS
// ==================

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include "cublas_v2.h"
#include<cuda.h>

// =============================
// ERROR check for kernel launch
// =============================
void Check_CUDA_Error(const char *message)
{
   cudaError_t error = cudaGetLastError();
   if(error!=cudaSuccess)
   {
      fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
      exit(-1);
   }
}

// ======
// STUCTS
// ======

typedef struct 
{
	FILE* UFC_Data_Q_file;
	FILE* UFC_Data_R_file;
	FILE* UFC_Data2_file;
	FILE* UFC_Data_main_file;
	FILE* group_flag_array_file;
	FILE* v_file;
	FILE* x_file;
} FilePointers;

//===========================
//ADDITIONAL KERNEL FUNCTIONS
//===========================

// divideGPU for c = a/b (element wise)

__global__ void divideGPU(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	while(idx < n)
	{
		c[idx] = a[idx] / b[idx];
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void multiplyGPU(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    while(idx < n)
    {
        c[idx] = a[idx] * b[idx]; 
        idx += blockDim.x * gridDim.x;
    }

}

// divideGPU_scalar for c = a /s where s is a scalar. 

__global__ void divideGPU_scalar(float *a, float *b, float s, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	while(idx < n)
	{
		b[idx] = a[idx] / s;
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void calc_l_GPU(float *r, float *q, float *l, float z, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	
	while(idx < n)
	{
		l[idx] = z/2 * (r[idx]-q[idx]) - (r[idx]-1);
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void combinitorial_GPU(float* psi_x, float *q, float *theta_psi,\
                              float* l, float z,\
                              float xdotl, float n, float* ln_gamma_c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	while(idx < n)
	{
		float term1 = log(psi_x[idx]);
		float term2 = z/2 * q[idx] * log(theta_psi[idx]);
		float term3 = psi_x[idx] * xdotl; 
		ln_gamma_c[idx] = term1 + term2 - term3 + l[idx]; 
                idx += blockDim.x * gridDim.x;
	}
}

__global__ void find_Theta_i(float *a, float *b, float* c, float *sum, int n, int m)
{
    // Load in from global to shared memory.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = 0; i < n; i ++)
    {
        while(idx < m * (i+1))
        {
            c[idx] = b[idx / m] * a[idx];
            sum[idx % m] += c[idx];
            idx += blockDim.x * gridDim.x;
        }
        __syncthreads();
    }
}

__global__ void divide_sum(float *a, float *sum, float *b, int n, int m)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx < m * n)
    {
        b[idx] = a[idx] / sum[idx% m];
        idx += blockDim.x * gridDim.x;
    }
}


// formula for (1.0 - log(a) - b)

__global__ void formula(float *a, float *b, float *c, float *q, int n, int m)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	while(idx < n)
	{
	    c[idx] = q[idx / m] * (1.0 - log(a[idx]) - b[idx]);
            idx += blockDim.x * gridDim.x;
	}
}

__global__ void formula_k(float *a, float *b, float *c, float *q, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    while(idx < n)
    {
        c[idx] = q[idx] * (1.0 - log(a[idx]) - b[idx]);
        idx += blockDim.x * gridDim.x;
    }
}

bool CorrectNumberOfArguments(int argc)
{
	if (argc == 2)
	{
		return true;
	}
	else
	{
		printf("Program recieved %d arguments. A single argument for a .cfg file is expected", argc - 1);
		return false;
	}
}

bool ArgumentsOkay(int argc, char* argv[])
{
	if (CorrectNumberOfArguments(argc) == false) return false;
	return true;
}

void LoadFilePointers(char* configFilename, FilePointers* filePointers)
{
	FILE* configurationFile = fopen(configFilename, "r");
    char key[100], value[100];
	
	while (fscanf(configurationFile, "%s : %s\n", key, value) != EOF)
	{
		if (key == NULL)
		{
			printf("Null key");
		}
		printf("Key: %s, Value: %s\n", key, value);
		if (strcmp(key, "Q") == 0) {
			(*filePointers).UFC_Data_Q_file = fopen(value, "r");
		} 
		else if (strcmp(key, "R") == 0) {
			(*filePointers).UFC_Data_R_file = fopen(value, "r");
		} 
		else if (strcmp(key, "Data2") == 0) {
			(*filePointers).UFC_Data2_file = fopen(value, "r");
		} 
		else if (strcmp(key, "Data_main") == 0) {
			(*filePointers).UFC_Data_main_file = fopen(value, "r");
		}
		else if (strcmp(key, "Group_Flag") == 0) {
			(*filePointers).group_flag_array_file = fopen(value, "r");
		}
		else if (strcmp(key, "v") == 0) {
			(*filePointers).v_file = fopen(value, "r");
		}
		else if (strcmp(key, "x") == 0) {
			(*filePointers).x_file = fopen(value, "r");
		}
	}
	
	fclose(configurationFile);
}

int main( int argc, char *argv[] )
{
	if (ArgumentsOkay(argc, argv) == false) return 3;

    // ===============
    // set CUDA device
    // ===============
    cudaSetDevice(0);
    cudaDeviceReset();

    // =================
    // Declare variables
    // =================

    // Dot products

    float xdotl;
    float xdotr; 
    float xdotq;
	
    // scalar variables
    int maxGroupNum = 23;
    int molecules = 183;
    int z = 10; 
    float T = 298.149994;
    	
    // files
	FilePointers filePointers;
	LoadFilePointers(argv[1], &filePointers);

    FILE *UFC_Data_Q_file = filePointers.UFC_Data_Q_file;
    FILE *UFC_Data_R_file = filePointers.UFC_Data_R_file;
    FILE *UFC_Data2_file = filePointers.UFC_Data2_file;
    FILE *UFC_Data_main_file = filePointers.UFC_Data_main_file;
	FILE* group_flag_array_file = filePointers.group_flag_array_file;
    FILE *v_file = filePointers.v_file; 
    FILE *x_file = filePointers.x_file;

    // check reading of files
	if(UFC_Data_Q_file == NULL) { perror("Error Opening UFC_Data_Q"); return 0;}
    if(UFC_Data_R_file == NULL) { perror("Error Opening UFC_Data_R"); return 0;}
    if(group_flag_array_file == NULL) { perror("Error Opening Group flags"); return 0;}
    if(v_file == NULL) { perror("Error opening v"); return 0; }
    if(x_file == NULL) { perror("Error opening x"); return 0; }
    if(UFC_Data2_file == NULL) { perror("Error opening Data2"); return 0; }
	if(UFC_Data_main_file == NULL) { perror("Error opening Main data"); return 0; }
	   
	
    // vectors 
    float * UFC_Data_Q;
    float * Q;
    float * UFC_Data_R;
    float * R;
    float * x;
    float * r; 
    float * q; 
    float * sum_v;
    int * UFC_Data_main;
    float * UFC_Data2;
    int * group_flag_array;
    
    // matrices
    float * V; 
	bool verbose = true;

    // ===========================
    // Allocate memory on the host
    // ===========================
	if(verbose) printf("Allocating memory on host\n");
	
    // vectors
    UFC_Data_Q = (float *) malloc(572 * sizeof(float));
    UFC_Data_R = (float *) malloc(572 * sizeof(float));
    UFC_Data_main = (int *) malloc(572 * sizeof(int));
    UFC_Data2 = (float *) malloc (76 * 76 * sizeof(float));
    x = (float *) malloc (molecules * sizeof(float));
    group_flag_array = (int *) malloc(maxGroupNum * sizeof(int)); 
    R = (float *) malloc(maxGroupNum * sizeof(float));
    Q = (float *) malloc(maxGroupNum * sizeof(float));
    r = (float *) malloc(molecules * sizeof(float)); 
    q = (float *) malloc(molecules * sizeof(float));
    sum_v = (float*) malloc(molecules * sizeof(float));
	
    // matrices 
    V = (float *) malloc(molecules * maxGroupNum * sizeof(float));


    // ====================
    // Variables for timing
    // ====================

    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
 
	
    // ========================
    // Declare Device Variables
    // ========================
	if(verbose) printf("Declaring Device Variables\n");
    float* d_Q; 
    float* d_R; 
    float* d_V; 
    float* d_q; 
    float* d_r;
    float* d_l; 
    float* d_x; 
    float *d_psi_x;
    float *d_q_psi_x;
    float *d_theta_psi;
    float* d_ln_gamma_c;
	float* d_ln_gamma_r;
    float *d_Theta;
    float *d_Theta_i;
    float *d_Theta_i_u;
    float *d_sum;
    float *d_Psi;
    float *d_ln_Gamma_i;               // device memory pointers
    float *d_thePsi_i; // additional pointer thePsi = Theta * Psi
    float *d_theDiv_i; // to store theta / thePsi
    float *d_thePsi_t_i; // additional pointer to store thetaDiv * transpose(Psi);
    float *d_ln_Gamma_k;
    float *d_thePsi;
    float *d_theDiv;
    float *d_thePsi_t;
		
		
    // =============================
    // Allocate memory on the device
    // =============================
	if(verbose) printf("Allocating memory on the device\n");
    cudaMalloc((void **) &d_ln_Gamma_i, molecules * maxGroupNum * sizeof(float));
    cudaMalloc((void **) &d_thePsi_i, molecules * maxGroupNum * sizeof(float));
    cudaMalloc((void **) &d_theDiv_i, molecules * maxGroupNum * sizeof(float));
    cudaMalloc((void **) &d_thePsi_t_i, molecules * maxGroupNum * sizeof(float));
    cudaMalloc((void **) &d_ln_Gamma_k, maxGroupNum * sizeof(float));
    cudaMalloc((void **) &d_thePsi, maxGroupNum * sizeof(float));
    cudaMalloc((void **) &d_theDiv, maxGroupNum * sizeof(float));
    cudaMalloc((void **) &d_thePsi_t,maxGroupNum * sizeof(float));
    cudaMalloc((void**) &d_Q, maxGroupNum * sizeof(float));
    cudaMalloc((void**) &d_R, maxGroupNum * sizeof(float));
    cudaMalloc((void**) &d_V, maxGroupNum * molecules * sizeof(float));
    cudaMalloc((void**) &d_q, molecules * sizeof(float));
    cudaMalloc((void**) &d_r, molecules * sizeof(float));
    cudaMalloc((void**) &d_l, molecules*sizeof(float));
    cudaMalloc((void**) &d_x, molecules * sizeof(*x)); //allocate memory
    cudaMalloc((void**) &d_psi_x, molecules * sizeof(float)); 
    cudaMalloc((void**) &d_q_psi_x, molecules *sizeof(float));
    cudaMalloc((void**) &d_theta_psi, molecules * sizeof(float));
    cudaMalloc((void**) &d_ln_gamma_c, molecules * sizeof(float));
	cudaMalloc((void**) &d_ln_gamma_r, molecules * sizeof(float));
    cudaMalloc((void**) &d_Theta, maxGroupNum * sizeof(float));
    cudaMalloc((void**) &d_Theta_i, maxGroupNum * molecules * sizeof(float));
    cudaMalloc((void**) &d_sum, molecules * sizeof(float));
    cudaMalloc((void**) &d_Theta_i_u, maxGroupNum * molecules * sizeof(float));
    cudaMalloc((void**) &d_Psi, maxGroupNum * maxGroupNum * sizeof(float));
    
    // ====================
    // read data from files
    // ====================
	if(verbose) printf("Reading data from files\n");
	
    // Read in the vector files first
    // UFC_Data_Q, UFC_Data_R, x, group_flag_array
	if(verbose) printf("Reading Q and R\n");
    for(int i = 0; i < 572; i++) // possibly need 572 as a variable
    {
        fscanf(UFC_Data_Q_file, "%f ", &UFC_Data_Q[i]);
		fscanf(UFC_Data_R_file, "%f ", &UFC_Data_R[i]);
    }
	if(verbose) printf("Reading x\n");
    for(int i = 0; i < molecules; i++)
    {
        fscanf(x_file, "%f ", &x[i]);
    }
	if(verbose) printf("Reading Group Flag Array\n");
    for(int i = 0; i < maxGroupNum; i++)
    {
        fscanf(group_flag_array_file, "%i ", &group_flag_array[i]);
    }
	if(verbose) printf("Reading UFC Data main\n");
    for(int i = 0; i < 572; i++)
    {
        fscanf(UFC_Data_main_file, "%i ", &UFC_Data_main[i]);
    }
	
    for(int i = 0; i < molecules; i++)
    {
        sum_v[i] = 0;
    }
        
    // And then the matrices (must be stored in column major form
    // for cublas!!)
	if(verbose) printf("Reading in Data2\n");
    for(int i = 0; i < 76; i++)
    {
        for(int j = 0; j < 76; j++)
        {
            fscanf(UFC_Data2_file, "%f ", &UFC_Data2[i + j * 76]);
        }
    }
    if(verbose) printf("Reading v\n");
    for(int i = 0; i < molecules; i++)
    {
        for(int j = 0; j < maxGroupNum; j++)
		{
            fscanf(v_file, "%f ", &V[i + j * molecules]);
		}
    }
    
        
    
    // TO OUTPUT GROUP FLAGS
	if(verbose) printf("Writing group flags\n");
	if(verbose)
	{
		FILE *groupFile = fopen("cuda_validation_files\\group.csv", "w");
		for(int j = 0; j < maxGroupNum; j++)
			fprintf(groupFile, "%i ", group_flag_array[j]);
		
		fprintf(groupFile, "\n");
		fclose(groupFile);
	}

	if(verbose) printf("Finished writing group flags\n");

    
        
    /*        
    // TO PRINT X
    printf("Printing x: \n");
    for(int i = 0; i < molecules; i++)
        printf("%f ", x[i]);
    printf("\n");
    */

    /*       
    // TO PRINT V
    
    for(int i = 0; i < molecules; i++)
    {
        for(int j = 0; j < maxGroupNum; j++)
        {
            printf("%f ", V[i + j * molecules]);
        }
        printf("\n");
    }
    printf("\n");
    */

		
    // =============================
    // Copy data from host to device
    // =============================
       
    cublasSetMatrix(molecules, maxGroupNum, sizeof(*V), V, molecules, d_V, molecules);    
    cublasSetVector(molecules, sizeof(float), q, 1, d_q, 1); 
    cublasSetVector(molecules, sizeof(float), r, 1, d_r, 1);
    cublasSetVector(molecules, sizeof(*x), x, 1, d_x, 1);
	
    // Create handle 
    cublasHandle_t handle;
    cublasCreate(&handle);
	
    // ============
    // BEGIN UNIFAC
    // ============
	
    //start the timer---------------------------------------------
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
	
    // =====================================
    // Step 1 : Calculate the combinitaorial
    // =====================================
  
    // Step 1.1 (copied and edited from original unifac) 
    // -------------------------------------------------
    // Approx 0.1% of the time of python version
	
    for (int j = 0; j < maxGroupNum; j++) 
    {
        Q[j] = UFC_Data_Q[group_flag_array[j]];
        R[j] = UFC_Data_R[group_flag_array[j]];
    } 

    cublasSetVector(maxGroupNum, sizeof(*Q), Q, 1, d_Q, 1);
    cublasSetVector(maxGroupNum, sizeof(*R), R, 1, d_R, 1);	
    cublasGetVector(maxGroupNum, sizeof(*Q), d_Q, 1, Q, 1);
    cublasGetVector(maxGroupNum, sizeof(*R), d_R, 1, R, 1);
        
    /*    
    // print Q and R
    printf("Printing Q : \n");
    for(int j = 0; j < maxGroupNum; j++)
        printf("%f ", Q[j]);
    printf("\n");
    printf("Printing R : \n");
    for(int j = 0; j < maxGroupNum; j++)
        printf("%f ", R[j]);
    printf("\n");
    */
     
    // Step 1.2  q = VQ and r = VR  
    //--------------------------------
    // approx 4.7 of the time of python version
        
    // scalars for matrix vector multiplication 
    // y = al Ax + bet y
    float al = 1.0f; 
    float bet = 0.0f;
    
    // Perform the matrix-vector multiplication 
    cublasSgemv(handle, CUBLAS_OP_N, molecules, maxGroupNum, &al, d_V, molecules, d_Q, 1, &bet, d_q, 1);
    cublasSgemv(handle, CUBLAS_OP_N, molecules, maxGroupNum, &al, d_V, molecules, d_R, 1, &bet, d_r, 1);
         

    /*
    // Return result to host
    cublasGetVector(molecules, sizeof(*q), d_q, 1, q, 1);
    cublasGetVector(molecules, sizeof(*r), d_r, 1, r, 1);

    // Print 
    printf("Printing q: \n");
    for(int i = 0; i < molecules; i++)
        printf("%f ", q[i]);
    printf("\n");

    printf("Printing r: \n");
    for(int i = 0; i < molecules; i++)
        printf("%f ", r[i]); 
    printf("\n");
    */
          

    // Step 1.3 : Calculate l
    // ----------------------
    // Responsible for 0.2% of CPU code
 
    calc_l_GPU<<<1024, 1024>>>(d_r, d_q, d_l, z, molecules);


    /*        
    // print l
    cudaThreadSynchronize();
    float* l; 
    l = (float*) malloc(molecules * sizeof(float));
    cudaMemcpy(l, d_l, sizeof(float) * molecules, cudaMemcpyDeviceToHost); 
    for(int i = 0; i < molecules; i++)
        printf("%f ", l[i]);
    printf("\n");
    */    
    

    // Step 1.4 : calculate dot products
    // ---------------------------------
    // responsible for 0.2% of CPU execution


    // output variables 
		
    // calculate dot products
    cublasSdot(handle, molecules, d_x, 1, d_l, 1, &xdotl);
    cublasSdot(handle, molecules, d_x, 1, d_r, 1, &xdotr);
    cublasSdot(handle, molecules, d_x, 1, d_q, 1, &xdotq); 

        
    // print results
    /* 
    printf("xdotr : %f\n", xdotr); 
    printf("xdotq : %f\n", xdotq);
    printf("xdotl : %f\n", xdotl);
    */       
           
    // Step 1.5 : calculate theta/x
    // ----------------------------
    // responsible for 0.1% of the time on CPU

    divideGPU_scalar<<<1024, 1024>>>(d_r, d_psi_x, xdotr, molecules);
 
    // Step 1.6 calculate theta/psi
    // ----------------------------
    // responsible for 3.1% of the time on the CPU

    divideGPU<<<1024, 1024>>>(d_q, d_psi_x, d_q_psi_x, molecules);
    divideGPU_scalar<<<1024, 1024>>>(d_q_psi_x, d_theta_psi, xdotq, molecules);
        

    /*    
    // print d_theta_psi and psi_x

    // copy to host
    float* theta_psi; 
    float* psi_x;
    theta_psi = (float*) malloc(sizeof(float) * molecules);
    psi_x = (float*) malloc(sizeof(float) * molecules); 
    cudaMemcpy(theta_psi, d_theta_psi, sizeof(float)* molecules, cudaMemcpyDeviceToHost); 
    cudaMemcpy(psi_x, d_psi_x, sizeof(float) * molecules, cudaMemcpyDeviceToHost);
        
        
    // print 
        
    printf("Printing theta_psi: \n");
    for(int i = 0; i < molecules; i++)
        printf("%f ", theta_psi[i]);
    printf("\n"); 

    printf("Printing psi_x: \n"); 
    for(int i = 0; i < molecules; i++)
        printf("%f ", psi_x[i]);
    printf("\n");	
    */
         
        
    // Part 1.7 : Calculate ln_gamma_c
    // -------------------------------
    // 3.4% of time on CPU



    combinitorial_GPU<<<1024, 1024>>>(d_psi_x, d_q, d_theta_psi, d_l, z,\
                                           xdotl, molecules, d_ln_gamma_c);
										   

    /*
    // print ln_gamma_c

    		
    float* ln_gamma_c; 
    ln_gamma_c = (float*) malloc(sizeof(float) * molecules);
    cudaMemcpy(ln_gamma_c, d_ln_gamma_c, sizeof(float)* molecules, cudaMemcpyDeviceToHost);
    printf("Printing ln_gamma_c");
    for(int i = 0; i < molecules; i++)
        printf("%f ", ln_gamma_c[i]);
    printf("\n"); 
     
    */
    

    /// Step 2.1 Calculate Theta 

	float* d_xv;
  cudaMalloc((void**) &d_xv, maxGroupNum * sizeof(float));
    cublasSgemv(handle, CUBLAS_OP_T, molecules, maxGroupNum, &al, d_V, molecules, d_x, 1, &bet, d_xv, 1);
    multiplyGPU<<<1024, 1024>>>(d_Q, d_xv, d_Theta, maxGroupNum);
	cudaFree(d_xv);

    float* Theta; 
    Theta = (float*) malloc(sizeof(float) * maxGroupNum);
    cublasGetVector(maxGroupNum, sizeof(*Theta), d_Theta, 1, Theta, 1);


    float sum = 0; 
    for(int i = 0; i < maxGroupNum; i++)
    {
         sum += Theta[i];
    }
    //printf("%f", sum); 
    for(int i = 0; i < maxGroupNum; i++)
    {
        Theta[i] = Theta[i] / sum;
    }

    cublasSetVector(maxGroupNum, sizeof(*Theta), Theta, 1, d_Theta, 1);
        

    /*        
    // print d_Theta
 
    cudaThreadSynchronize();
    cublasGetVector(maxGroupNum, sizeof(*Theta), d_Theta, 1, Theta, 1);
        

    printf("Printing Theta: ");
    for(int i = 0; i < maxGroupNum; i++)
        printf("%f ", Theta[i]);
    printf("\n"); 
    
    */    

    // Section 2.2 : Find Theta_i
    cublasSetVector(molecules, sizeof(*sum_v), sum_v, 1, d_sum, 1);
    find_Theta_i<<<1024, 1024>>>(d_V, d_Q, d_Theta_i_u, d_sum, maxGroupNum, molecules);
    divide_sum<<<1024, 1024>>>(d_Theta_i_u, d_sum, d_Theta_i, maxGroupNum, molecules);        


    /*
    // print sum 
     
    cudaDeviceSynchronize();  
    cublasGetVector(molecules, sizeof(*sum_v), d_sum, 1, sum_v, 1);
    for(int i = 0; i < molecules; i++)
        printf("%f ", sum_v[i]);
    printf("\n");

    // print d_Theta_i


    cudaThreadSynchronize();
    float* Theta_i = (float*) malloc(sizeof(float) * molecules * maxGroupNum);
    cublasGetMatrix(molecules, maxGroupNum, sizeof(*Theta_i), d_Theta_i_u,molecules, Theta_i, molecules);
    for(int i = 0; i < molecules; i++)
    {
        for(int j = 0; j < maxGroupNum; j++)
        {
            printf("%f ", Theta_i[i + j * molecules]);
        }
        printf("\n");
    }
    printf("\n");           

    */     

    // Section 2.3 : Get Psi

    int * needed = (int *) malloc(sizeof(int) * maxGroupNum);
    for(int i = 0; i < maxGroupNum; i++)
    {
        needed[i] = UFC_Data_main[group_flag_array[i]];
    }
    float * Psi = (float *) malloc(sizeof(float) * maxGroupNum * maxGroupNum); 
    for(int i = 0; i < maxGroupNum; i++)
    {
        for(int j = 0; j < maxGroupNum; j++)
        {
            float a = UFC_Data2[needed[i] + needed[j] * 76];
            Psi[i + j * maxGroupNum] = exp(-a / T); 
//            printf("%f ", Psi[i + j * maxGroupNum]);
        }
//        printf("\n");
    }
//    printf("\n");

    // Copy psi to the device 

    cublasSetMatrix(maxGroupNum, maxGroupNum, sizeof(*Psi), Psi, maxGroupNum, d_Psi, maxGroupNum);

 
    // Find d_ln_Gamma_i
    /*
    // Perform the matrix multiplication Theta * Psi and store in d_thePsi
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, molecules, maxGroupNum, maxGroupNum, &al, d_Theta_i, molecules, d_Psi, maxGroupNum, &bet, d_thePsi_i, molecules);

    // Perform the elementwise division : Theta / ThePsi and store in d_theDiv
    divideGPU<<<1024, 1024>>>(d_Theta_i, d_thePsi_i, d_theDiv_i, molecules * maxGroupNum);

    // Perform the second matrix multiplication d_theDiv * Psi (transpose)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, molecules, maxGroupNum, maxGroupNum, &al, d_theDiv_i, molecules, d_Psi, maxGroupNum, &bet, d_thePsi_t_i, molecules);
	
    // substitute d_thePsi and d_thePsi_t into Q ( 1.0 - log(d_thePsi_t) - d_thePsi_t) and store as d_formres
    formula<<<1024, 1024>>>(d_thePsi_i, d_thePsi_t_i, d_ln_Gamma_i, d_Q, maxGroupNum * molecules, molecules);
    
       	
    // Find d_ln_Gamma_k
    	
    // Perform the matrix multiplication Theta * Psi and store in d_thePsi
    cublasSgemv(handle, CUBLAS_OP_T, maxGroupNum, maxGroupNum, &al, d_Psi, maxGroupNum, d_Theta, 1, &bet, d_thePsi, 1);
    
    // Perform the elementwise division : Theta / ThePsi and store in d_theDiv
    divideGPU<<<1024, 1024>>>(d_Theta, d_thePsi, d_theDiv, maxGroupNum);

    // Perform the second matrix multiplication d_theDiv * Psi (transpose)
    cublasSgemv(handle, CUBLAS_OP_N, maxGroupNum, maxGroupNum, &al, d_Psi, maxGroupNum, d_theDiv, 1, &bet, d_thePsi_t, 1);
	
	// substitute d_thePsi and d_thePsi_t into Q ( 1.0 - log(d_thePsi_t) - d_thePsi_t) and store as d_formres
    formula_k<<<1024, 1024>>>(d_thePsi, d_thePsi_t, d_ln_Gamma_k, d_Q, maxGroupNum);

	// calculate the residual
    matrixVectorRowSubtract<<<1024, 1024>>>(d_ln_Gamma_i, d_ln_Gamma_k, molecules, maxGroupNum);

    // print d_ln_Gamma_i 
    /*
                
    float* ln_Gamma_i = (float*) malloc(sizeof(float) * molecules * maxGroupNum);
    cublasGetMatrix(molecules, maxGroupNum, sizeof(*ln_Gamma_i), d_ln_Gamma_i, molecules, ln_Gamma_i, molecules);
    for(int i = 0; i < molecules; i++)
    {
        for(int j = 0; j < maxGroupNum; j++)
        {
            printf("%f ", ln_Gamma_i[i + j * molecules]);
        }
        printf("\n"); 
    }
    printf("\n");
    */


	//calculateResidual <<<1024, 1024 >>> (d_V, , d_ln_Gamma_i, d_ln_gamma_r, molecules, maxGroupNum);
    
    // Print d_ln_Gamma
    cudaThreadSynchronize();
    float* ln_Gamma_k = (float*) malloc(sizeof(float) * maxGroupNum);
    cublasGetVector(maxGroupNum, sizeof(*ln_Gamma_k), d_ln_gamma_c, 1, ln_Gamma_k, 1); 
    for(int i = 0; i < maxGroupNum; i++)
        printf("%f ", ln_Gamma_k[i]);
    printf("\n");    

    Check_CUDA_Error("");
     //stop the timer----------------------------------------------
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, start, stop);
     
    printf(" GPU Done. Execution time: %f (ms)\n", elapsedTimeGPU);
 
	return EXIT_SUCCESS;
}





