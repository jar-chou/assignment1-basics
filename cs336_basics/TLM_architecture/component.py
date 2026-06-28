import torch
from torch import nn
import math
import einops
from torch.nn import functional as F
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((out_features,in_features), device=device, dtype=dtype))
        std = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)
    
    def forward(self, X):
        # return X@self.W.T
        return einops.einsum(X, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embedding:int,embedding_dim:int,device=None,dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((num_embedding, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight,mean=0,std=1,a=-3,b=3)
        
    def forward(self, X):
        # return F.one_hot(X,self.weight.shape[0]).type(dtype=self.weight.dtype)@self.weight
        return self.weight[X]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model, self.eps = d_model, eps
        self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
    
    def forward(self, x:torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x*x).sum(dim=-1)/self.d_model+self.eps)
        # rms = einops.repeat(rms,"... -> ... c", c=self.d_model)
        return (x/rms[..., None]*self.weight[None, ...]).to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        self.d_model, self.d_ff = d_model, d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    
    def SiLU(self,X):
        X = self.w1(X)
        return X*(torch.sigmoid(X))
    
    def forward(self,X):
        return self.w2(self.w3(X)*self.SiLU(X))

class RoPE(nn.Module):
    '''
    # 这是朴素的实现
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta, self.d_k, self.max_seq_len = theta, d_k, max_seq_len
        self.pre_computed_matrix = torch.zeros((max_seq_len+1,d_k,d_k), device=device)
        for i in range(0,1+max_seq_len):
            for k in range(round(d_k/2)):
                theta_i_k = i/(math.pow(self.theta, 2*float(k)/self.d_k))
                self.pre_computed_matrix[i][k*2][k*2] = self.pre_computed_matrix[i][k*2+1][k*2+1] = math.cos(theta_i_k)
                self.pre_computed_matrix[i][k*2+1][k*2] = math.sin(theta_i_k)
                self.pre_computed_matrix[i][k*2][k*2+1] = -self.pre_computed_matrix[i][k*2+1][k*2]
    
    def __get_RoPE_matrix(self, token_position: torch.Tensor):
        return self.pre_computed_matrix[token_position]
        
    def forward(self,x:torch.Tensor, token_position: torch.Tensor)->torch.Tensor:
        RoPE_matrix=self.__get_RoPE_matrix(token_position)
        out = torch.empty_like(x, device=RoPE_matrix.device)
        for j in range(x.shape[1]):
            out[...,j,:] = x[...,j,:]@RoPE_matrix[j].T
        return out
    
    '''
    # Llama 的实现：这里的cache大小不用存很多零的稀疏矩阵了，而且计算的时候不需要做矩阵乘法大量乘零元素
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_k = d_k
        
        # 1. 预计算频率，大小为 d_k // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device).float()
        
        # freqs 形状: [max_seq_len, d_k // 2]
        freqs = torch.outer(t, inv_freq)
        
        # 2. 核心：利用欧拉公式，直接计算出复数旋转因子 e^(i * theta)
        # torch.polar(magnitudes, angles) 根据模长(1.0)和角度生成复数
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(dtype=dtype)
        
        # 缓存复数因子（不参与梯度，形状为 [max_seq_len, d_k // 2]）
        self.register_buffer("freqs_cis_cached", freqs_cis, persistent=False)
    
    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Num_Heads, Seq_Len, d_k] 
        token_position: [Seq_Len]
        """
        # 1. 将输入的最后一个维度 [..., d_k] 转化为复数形式 [..., d_k // 2]
        # view_as_complex 要求最后一个维度必须是连续的，且大小为 2 的倍数
        # 它会把原本相邻的 (x0, x1) 自动变成 x0 + i*x1
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # print(token_position.shape)
        
        # 2. 从缓存中根据当前 batch 里的位置获取复数因子
        # freqs_cis 形状: [seq_len, d_k // 2]
        freqs_cis = self.freqs_cis_cached[token_position]
        
        # 3. 完美的复数乘法！(x0 + i*x1) * (cos + i*sin)
        # 这一步瞬间完成了原论文中所有相邻通道的两两旋转
        x_rotated_complex = x_complex * freqs_cis
        
        # 4. 最后，将复数重新展平展开为实数张量，恢复原来的 [Batch, Num_Heads, Seq_Len, Head_Dim] 
        x_out = torch.view_as_real(x_rotated_complex).flatten(-2)
        
        return x_out.type_as(x)

rope = None
rope_sign = 0

def softmax(V: torch.Tensor, dim=-1):
    V_max = V.max(dim=dim, keepdim=True)
    V = V-V_max.values
    exp_V = torch.exp(V)
    return exp_V/exp_V.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    Q_multiply_KT = einops.einsum(Q, K, "... queries d_k, ... keys d_k-> ... queries keys")/math.sqrt(Q.shape[-1])
    if isinstance(mask, torch.Tensor):
        # Q_multiply_KT[mask==False]= Q_multiply_KT[mask==False] - torch.inf
        Q_multiply_KT = Q_multiply_KT.masked_fill(~mask,-torch.inf)
    attention_weight = softmax(Q_multiply_KT, -1)
    return einops.einsum(attention_weight, V, "... queries keys, ... keys d_v->  ... queries d_v")

class MultiHeadSelfAttention(nn.Module):
    rope = None
    rope_sign = 0

    def __init__(self, num_heads, d_model, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        
        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_v)
        self.output_proj = Linear(num_heads * self.d_v, d_model)
        
        if max_seq_len and theta and (MultiHeadSelfAttention.rope_sign == 0):
            # 明确赋值给类变量
            MultiHeadSelfAttention.rope = RoPE(theta, self.d_k, max_seq_len, device=device, dtype=dtype)
            MultiHeadSelfAttention.rope_sign = 1
            
        
    
    
    def forward(self, x, token_positions=None):
        bsz, seqlen, _ = x.shape
        Qs = self.q_proj(x)
        Ks = self.k_proj(x)
        Vs = self.v_proj(x)
        
        Qs = einops.rearrange(Qs, "b l (h d) -> b h l d", h=self.num_heads)
        Ks = einops.rearrange(Ks, "b l (h d) -> b h l d", h=self.num_heads)
        Vs = einops.rearrange(Vs, "b l (h d) -> b h l d", h=self.num_heads)
        if (MultiHeadSelfAttention.rope_sign):
            if not isinstance(token_positions,torch.Tensor):
                token_positions = torch.tensor(range(x.shape[-2]))
            Qs = MultiHeadSelfAttention.rope(Qs, token_positions)
            Ks = MultiHeadSelfAttention.rope(Ks, token_positions)
        mask = None
        mask = torch.tril(torch.ones((seqlen, seqlen), device=x.device)).bool()
        
        attention = scaled_dot_product_attention(Qs,Ks,Vs,mask)
        
        attention = einops.rearrange(attention,"b h l d -> b l (h d)")
        
        return self.output_proj(attention)

class Transformer_Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(num_heads, d_model, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
    def forward(self, x, token_positions=None):
        x = x+self.attn(self.ln1(x), token_positions)
        return x+self.ffn(self.ln2(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=None, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            Transformer_Block(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

