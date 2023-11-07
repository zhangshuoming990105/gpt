# run pytorch to get native performance
python run_gpt.py

# run gpu performance
./gpu_main
# run cpu performance
./cpu_main

# to further get the tvm optimized performance, need to install tvm
# and uncomment the lines in run_gpt.py

# todo:
# 1. measure the per layer performance in each setup
# 2. optimize the naive kernel used in naive gpu, 2 types, elemwise + matmul
# 3. test scale up, (T,C) test in (32, 64)v, (32, 128), (64, 128), (64, 256), (128, 256), (128, 512), (256,512)
# 4. (optional) optimize the naive implementation in cpu using omp parallel?
# 5. try fused kernels