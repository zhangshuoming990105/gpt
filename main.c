#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define T 512
#define C 2048

void fill_val(float *A, int m, int n, float value){
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A[i * n + j] = value;
        }
    }
}

void matmul_cpu(float *A, float *B, float *O, int m, int n, int k) {
    // O[i, j] += A[i, t] * B[t, j] O[m][n], A[m][k], B[k][n]
    // for(int i = 0; i < m; i++) {
    //     for(int j = 0; j < n; j++) {
    //         O[i * n + j] = 0.0f;
    //     }
    // }
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float temp = 0;
            for(int t = 0; t < k; t++) {
                temp += A[i*k + t]*B[t*n + j];
            }
            O[i * n + j] = temp;
        }
    }
}

void matmul_transpose_cpu(float *A, float *B, float *O, int m, int n, int k) {
    // O[i, j] += A[i, t] * B[j, t] O[m][n], A[m][k], B[n][k]
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            O[i * n + j] = 0.0f;
        }
    }
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float temp = 0;
            for(int t = 0; t < k; t++) {
                temp += A[i*k + t]*B[j*n + t];
            }
            O[i * n + j] = temp;
        }
    }
}

void mul_scalar(float *A, float *B, float value, int n) {
    // B[i] = A[i] * value;
    for(int i = 0; i < n; i++) {
        B[i] = A[i] * value;
    }
}

void softmax(float *A, float *B, int n) {
    float temp[n];
    float sum = 0;
    for(int i = 0; i < n; i++) {
        temp[i] = exp(A[i]);
        sum += temp[i];
    }
    for(int i = 0; i < n; i++) {
        B[i] = temp[i] / sum;
    }
}

void RunCPU(int n) {
    float *input = (float *)malloc(T*C*sizeof(float));
    float *Q = (float *)malloc(C*C*sizeof(float));
    float *K = (float *)malloc(C*C*sizeof(float));
    float *V = (float *)malloc(C*C*sizeof(float));
    float *Q_o = (float *)malloc(T*C*sizeof(float));    // input@Q ->Q_o
    float *K_o = (float *)malloc(T*C*sizeof(float));    // input@K ->K_o
    float *V_o = (float *)malloc(T*C*sizeof(float));    // input@V ->V_o
    float *att_0 = (float *)malloc(T*T*sizeof(float));  // matmul_transpose(Q, K)->att_0
    float norm_val = 1.0/sqrt(C);
    float *att_1 = (float *)malloc(T*T*sizeof(float));  // att_0*norm_val->att1
    float *att_2 = (float *)malloc(T*T*sizeof(float));  // att_1->softmax->att_2

    float *output = (float *)malloc(T*C*sizeof(float));
    // float input[T][C];
    // float Q[C][C];
    // float K[C][C];
    // float V[C][C];
    // float Q_o[T][C];    // input@Q ->Q_o
    // float K_o[T][C];    // input@K ->K_o
    // float V_o[T][C];    // input@V ->V_o
    // float att_0[T][T];  // matmul_transpose(Q, K)->att_0
    // float norm_val = 1.0/sqrt(C);
    // float att_1[T][T];  // att_0*norm_val->att1
    // float att_2[T][T];  // att_1->softmax->att_2
    
    // float output[T][C];

    fill_val(input, T, C, 1.0f);
    fill_val(Q, C, C, 1e-2f);
    fill_val(K, C, C, 1e-2f);
    fill_val(V, C, C, 1e-2f);
    // time testing
    clock_t start, end;
    start = clock();
    for(int i = 0; i < n; i++) {
        matmul_cpu(input, Q, Q_o, T, C, C);
        matmul_cpu(input, K, K_o, T, C, C);
        matmul_cpu(input, V, V_o, T, C, C);
        // printf("%f\n", Q_o[0][0]);
        // printf("%f\n", K_o[0][0]);
        // printf("%f\n", V_o[0][0]);
        matmul_transpose_cpu(Q_o, K_o, att_0, T, T, C); //(T,C)@(C, T)->(T,T)
        // printf("%f\n", att_0[0][0]);
        mul_scalar(att_0, att_1, norm_val, T*T);  //(T, T)
        // printf("%f\n", att_1[0][0]);
        softmax(att_1, att_2, T*T);   //(T, T)
        // printf("%f\n", att_2[0][0]);
        matmul_cpu(att_2, V_o, output, T, C, T);
        // printf("%f\n", output[0][0]);
    }
    end = clock();
    printf("CPU average time: %f ms\n", (double)(end - start) / (CLOCKS_PER_SEC*n/1000));
    free(input);
    free(Q);
    free(K);
    free(V);
    free(Q_o);
    free(K_o);
    free(V_o);
    free(att_0);
    free(att_1);
    free(att_2);
    free(output);
}

int main() {
    // prior simplify:
    /*
    1. no dropout
    2. only one head
    3. only one block
    4. skip layernorm
    */
    RunCPU(1);
    return 0;
}