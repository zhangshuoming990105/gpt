#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
// #include <cuda_runtime.h>
#define T 4096
#define C 2048

void fill_val(float *A, int m, int n, float value){
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A[i * n + j] = value;
        }
    }
}

__global__ void matmul(float *A, float *B, float *O, int m, int n, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < m && y < n) {
        float tmp = 0.0f;
        for(int i = 0; i < k; i++) {
            tmp += A[x * k + i] * B[i * n + y];
        }
        O[x * n + y] = tmp;
        // O[x * n + y] = 0.0f;
    }
}

__global__ void matmul_transpose(float *A, float *B, float *O, int m, int n, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // A[m, k] * B[n, k] = O[m, n]
    if(x < m && y < n) {
        float tmp = 0.0f;
        for(int i = 0; i < k; i++) {
            tmp += A[x * k + i] * B[y * n + i];
        }
        O[x * n + y] = tmp;
    }
}

__global__ void mul_scalar(float *A, float *B, int n, float value) {
    // B[i] = A[i] * value
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        B[i] = A[i] * value;
    }
}

__global__ void div_scalar(float *A, float *B, int n, float *value) {
    // B[i] = A[i] / value
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        B[i] = A[i] / value[0];
    }
}

__global__ void reduce_sum(const float* input, float *sum, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
        atomicAdd(sum, input[idx]);
}

__global__ void exp(const float* input, float *output, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
        output[idx] = expf(input[idx]);
}

__global__ void naiveSoftmax(const float* input, float* output, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numElements) {
        // Compute the sum of exponentials for all elements
        float expSum = 0.0f;
        for (int i = 0; i < numElements; i++) {
            expSum += expf(input[i]);
        }
        // Compute the softmax for the current element
        output[idx] = expf(input[idx]) / expSum;
    }
}

void RunGPU(int n) {
    // host init
    float *input;
    float *Q;
    float *K;
    float *V;
    float *Q_o;
    float *K_o;
    float *V_o;
    float *att_0;
    float *att_1;
    float *exp_att_1;
    float *sum;
    float *att_2;
    float *output;
    input = (float *)malloc(T*C*sizeof(float));
    Q = (float *)malloc(C*C*sizeof(float));
    K = (float *)malloc(C*C*sizeof(float));
    V = (float *)malloc(C*C*sizeof(float));
    Q_o = (float *)malloc(T*C*sizeof(float));
    K_o = (float *)malloc(T*C*sizeof(float));
    V_o = (float *)malloc(T*C*sizeof(float));
    att_0 = (float *)malloc(T*T*sizeof(float));
    att_1 = (float *)malloc(T*T*sizeof(float));
    exp_att_1 = (float *)malloc(T*T*sizeof(float));
    sum = (float *)malloc(1*sizeof(float));
    att_2 = (float *)malloc(T*T*sizeof(float));
    output = (float *)malloc(T*C*sizeof(float));

    // device init
    float *input_d;
    float *Q_d;
    float *K_d;
    float *V_d;
    float *Q_o_d;
    float *K_o_d;
    float *V_o_d;
    float *att_0_d;
    float *att_1_d;
    float *exp_att_1_d;
    float *sum_d;
    float *att_2_d;
    float *output_d;
    cudaMalloc((void **)&input_d, T*C*sizeof(float));
    cudaMalloc((void **)&Q_d, C*C*sizeof(float));
    cudaMalloc((void **)&K_d, C*C*sizeof(float));
    cudaMalloc((void **)&V_d, C*C*sizeof(float));
    cudaMalloc((void **)&Q_o_d, T*C*sizeof(float));
    cudaMalloc((void **)&K_o_d, T*C*sizeof(float));
    cudaMalloc((void **)&V_o_d, T*C*sizeof(float));
    cudaMalloc((void **)&att_0_d, T*T*sizeof(float));
    cudaMalloc((void **)&att_1_d, T*T*sizeof(float));
    cudaMalloc((void **)&exp_att_1_d, T*T*sizeof(float));
    cudaMalloc((void **)&sum_d, 1*sizeof(float));
    cudaMalloc((void **)&att_2_d, T*T*sizeof(float));
    cudaMalloc((void **)&output_d, T*C*sizeof(float));

    // init host data
    fill_val(input, T, C, 1.0f);
    fill_val(Q, C, C, 1e-2f);
    fill_val(K, C, C, 1e-2f);
    fill_val(V, C, C, 1e-2f);

    // copy data to device
    cudaMemcpy(input_d, input, T*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Q_d, Q, C*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K, C*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, C*C*sizeof(float), cudaMemcpyHostToDevice);

    // start compute
    // (T, C) * (C, C)
    // 2d tile
    dim3 gridDim1(ceil(T/32), ceil(C/32), 1);
    dim3 blockDim1(32, 32, 1);   // launch_bound = 1024
    // 1d tile
    dim3 gridDim2(ceil(T*T/1024), 1, 1);
    dim3 blockDim2(1024, 1, 1);
    clock_t start, end;
    start = clock();
    for(int i = 0; i < n; i++) {
        matmul<<<gridDim1, blockDim1>>>(input_d, Q_d, Q_o_d, T, C, C);
        matmul<<<gridDim1, blockDim1>>>(input_d, K_d, K_o_d, T, C, C);
        matmul<<<gridDim1, blockDim1>>>(input_d, V_d, V_o_d, T, C, C);
        // cudaMemcpy(Q_o, Q_o_d, T*C*sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(K_o, K_o_d, T*C*sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(V_o, V_o_d, T*C*sizeof(float), cudaMemcpyDeviceToHost);
        // printf("%f\n", Q_o[T*C-1]);
        // printf("%f\n", K_o[T*C-1]);
        // printf("%f\n", V_o[T*C-1]);
        matmul_transpose<<<gridDim1, blockDim1>>>(Q_o_d, K_o_d, att_0_d, T, T, C);
        // cudaMemcpy(att_0, att_0_d, T*T*sizeof(float), cudaMemcpyDeviceToHost);
        // printf("%f\n", att_0[T*T-1]);

        float rec_sqrt = 1.0f/sqrtf(C);
        mul_scalar<<<gridDim2, blockDim2>>>(att_0_d, att_1_d, T*T, rec_sqrt);
        // cudaMemcpy(att_1, att_1_d, T*T*sizeof(float), cudaMemcpyDeviceToHost);
        // printf("%f\n", att_1[T*T-1]);

        // softmax
        exp<<<gridDim2, blockDim2>>>(att_1_d, exp_att_1_d, T*T);
        // cudaMemcpy(exp_att_1, exp_att_1_d, T*T*sizeof(float), cudaMemcpyDeviceToHost);
        // printf("%f\n", exp_att_1[T*T-1]);
        reduce_sum<<<gridDim2, blockDim2>>>(exp_att_1_d, sum_d, T*T);
        // cudaMemcpy(sum, sum_d, 1*sizeof(float), cudaMemcpyDeviceToHost);
        // printf("%f\n", sum[0]);
        div_scalar<<<gridDim2, blockDim2>>>(exp_att_1_d, att_2_d, T*T, sum_d);
        // cudaMemcpy(att_2, att_2_d, T*T*sizeof(float), cudaMemcpyDeviceToHost);
        // printf("%f\n", att_2[T*T-1]);

        matmul<<<gridDim1, blockDim1>>>(att_2_d, V_o_d, output_d, T, C, T);
        cudaDeviceSynchronize();
        
    }
    end = clock();
    cudaMemcpy(output, output_d, T*C*sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU average time: %f ms\n", (double)(end - start) / (CLOCKS_PER_SEC*n/1000));
    
    // printf("%f\n", output[T*C-1]);


    free(input);
    free(Q);
    free(K);
    free(V);
    free(Q_o);
    free(K_o);
    free(V_o);
    free(att_0);
    free(att_1);
    free(exp_att_1);
    free(sum);
    free(att_2);
    free(output);
    cudaFree(input_d);
    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(Q_o_d);
    cudaFree(K_o_d);
    cudaFree(V_o_d);
    cudaFree(att_0_d);
    cudaFree(att_1_d);
    cudaFree(exp_att_1_d);
    cudaFree(sum_d);
    cudaFree(att_2_d);
    cudaFree(output_d);
}


int main() {
    // prior simplify:
    /*
    1. no dropout
    2. only one head
    3. only one block
    4. skip layernorm
    */
    RunGPU(10);
    return 0;
}