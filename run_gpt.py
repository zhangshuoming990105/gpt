import torch
import torch.nn as nn
import inspect
import numpy as np
import torch.nn.functional as F
import math
from dataclasses import dataclass
import time

torch.manual_seed(42)

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


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


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    # vocab_size: int = 10  # vocab size
    # n_layer: int = 2
    n_head: int = 1
    block_size: int = 4096  # length of sequence
    n_embd: int = 2048
    dropout: float = 0.0
    bias: bool = False


if __name__ == "__main__":
    device = "cuda"
   
    print("Using {} device".format(device))
    config = GPTConfig()
    print(config)
    B = 1
    T = 4096
    C = 2048

    # x = torch.randn((1, 16, 128))
    x = torch.randn((B, T, C), device=device)
    model = CasualSelfAttention(config)
    model.eval()
    model.to(device)
    # native pytorch baseline time test
    t1 = time.time()
    for i in range(1000):
        y = model.forward(x)
    t2 = time.time()
    print("Pytorch gpu baseline time: {} ms".format((t2 - t1)))
    device = "cpu"
    x = torch.randn((B, T, C), device=device)
    model = CasualSelfAttention(config)
    model.eval()
    model.to(device)
    # native pytorch baseline time test
    t1 = time.time()
    for i in range(100):
        y = model.forward(x)
    t2 = time.time()
    print("Pytorch cpu baseline time: {} ms".format((t2 - t1)*10))
    
    import tvm
    from tvm.topi.utils import traverse_inline, get_const_tuple
    from tvm import te, tir, auto_scheduler, topi, autotvm
    from tvm import relay
    # scripted_model = torch.jit.trace(model, x).eval()
    # input_name = "input0"
    # shape_list = [(input_name, (1, 16, 128))]
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
    # print("imported modules")
    # print(lib.lib.imported_modules[0].get_source())



    # from tvm.contrib import graph_executor
    # m = graph_executor.GraphModule(lib["default"](dev))
    # x = torch.randn((1, 16, 128))
    
    # t3 = time.time()
    # for i in range(1000):
    #     m.set_input(input_name, tvm.nd.array(x))
    #     m.run()
    # t4 = time.time()

    # print("Naive tvm time: {} ms".format((t4 - t3)))


    # optimal tvm time test, O3
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target=target, params=params)
    # m = graph_executor.GraphModule(lib["default"](dev))
    # t5 = time.time()
    # for i in range(1000):
    #     m.set_input(input_name, tvm.nd.array(x))
    #     m.run()
    # t6 = time.time()

    # print("Optimized tvm time: {} ms".format((t6 - t5)))

    
