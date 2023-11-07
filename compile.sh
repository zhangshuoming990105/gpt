# my base env is using A100 80G
# compile naive cpu 
gcc -O2 main.c -lm -o cpu_main -w
# compile naive gpu
nvcc -O2 main.cu -o gpu_main

