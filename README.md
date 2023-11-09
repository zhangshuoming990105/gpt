# manual optimize GPT like LLM using optimizations

## we target on the attention block optimization for simplicity

```bash
nsys profile gpu_main
nsys export -t json report1.nsys-rep
nsys stats report1.nsys-rep
```

the result shows that ~99% of the time is spent on the `matmul` kernel, which is the bottleneck of the attention block.

## currently we probably will test on:

- [x] pytorch cpu
- [x] pytorch gpu
- [x] tvm cpu
- [x] tvm gpu
- [x] naive cpu implementation
- [x] naive gpu implementation
- [ ] optimized gpu implementation
- [ ] autotvm gpu
- [ ] ansor gpu
