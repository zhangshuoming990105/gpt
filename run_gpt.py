import torch
import torch.nn as nn
import inspect
import numpy as np
import torch.nn.functional as F
import math
from dataclasses import dataclass
import time

torch.manual_seed(42)


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # assert config.n_embd % config.n_head == 0
        # K, Q, V in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = False
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = (q @ k.transpose(-2, -1)) * torch.reciprocal(torch.tensor([math.sqrt(k.size(-1))]))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

@dataclass
class GPTConfig:
    # vocab_size: int = 10  # vocab size
    # n_layer: int = 2
    n_head: int = 1
    block_size: int = 512  # length of sequence
    n_embd: int = 2048
    dropout: float = 0.0
    bias: bool = False


if __name__ == "__main__":

    config = GPTConfig()
    print(config)
    B = 1
    T = 512
    C = 2048
    n = 1


    # native pytorch baseline time test
    device = "cuda"
    print("Using {} device".format(device))
    x = torch.randn((B, T, C), device=device)
    model = CasualSelfAttention(config)
    model.eval()
    model.to(device)
    t1 = time.perf_counter()
    for i in range(n):
        y = model.forward(x)
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    print("Pytorch gpu baseline time: {} ms".format((t2 - t1)*1000/n))
    # time.sleep(1)
    # device = "cpu"
    # print("Using {} device".format(device))
    # x = torch.randn((B, T, C), device=device)
    # model = CasualSelfAttention(config)
    # model.eval()
    # model.to(device)
    # # native pytorch baseline time test
    # t1 = time.perf_counter(), time.process_time()
    # for i in range(n):
    #     y = model.forward(x)
    #     torch.cpu.synchronize()
    # t2 = time.perf_counter(), time.process_time()
    # print(f"Pytorch cpu baseline time: {(t2[0] - t1[0])*1000/n:.2f} ms")
    # print(f"total CPU time: {(t2[1] - t1[1])*1000/n:.2f} ms")
    



    ##### tvm part not used yet
    # import tvm
    # from tvm.topi.utils import traverse_inline, get_const_tuple
    # from tvm import te, tir, auto_scheduler, topi, autotvm
    # from tvm import relay
    # scripted_model = torch.jit.trace(model, x).eval()
    # input_name = "input0"
    # shape_list = [(input_name, (B, T, C))]
    # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    # # print(mod.astext())
    # target = tvm.target.Target("cuda", host="llvm")
    # dev = tvm.cuda(0)
    # # naive tvm time test, O0
    # with tvm.transform.PassContext(opt_level=0):
    #     lib = relay.build(mod, target=target, params=params)
    # dtype = "float32"
    # # print(lib.ir_mod)
    # # print(lib.function_metadata)
    # # print("imported modules")
    # # print(lib.lib.imported_modules[0].get_source())



    # from tvm.contrib import graph_executor
    # m = graph_executor.GraphModule(lib["default"](dev))
    # x = torch.randn((B, T, C))
    
    # t3 = time.time()
    # for i in range(1000):
    #     m.set_input(input_name, tvm.nd.array(x))
    #     m.run()
    # t4 = time.time()

    # print("Naive tvm time: {} ms".format((t4 - t3)))


    # # optimal tvm time test, O3
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target=target, params=params)
    # m = graph_executor.GraphModule(lib["default"](dev))
    # t5 = time.time()
    # for i in range(1000):
    #     m.set_input(input_name, tvm.nd.array(x))
    #     m.run()
    # t6 = time.time()

    # print("Optimized tvm time: {} ms".format((t6 - t5)))

    
