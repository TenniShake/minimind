import torch
import numpy as np

def demo_shape_unpacking():
    """演示 bsz, seq_len, _ = x.shape 的解包操作"""
    print("=" * 50)
    print("1. 形状解包演示: bsz, seq_len, _ = x.shape")
    print("=" * 50)
    
    # 创建一个模拟的输入张量 (batch_size=2, seq_len=4, hidden_size=3)
    x = torch.randn(2, 4, 3)
    print(f"原始张量 x 的形状: {x.shape}")
    print(f"张量内容:\n{x}")
    
    # 解包形状
    bsz, seq_len, hidden_size = x.shape
    print(f"\n解包结果:")
    print(f"bsz (batch_size) = {bsz}")
    print(f"seq_len (sequence_length) = {seq_len}")
    print(f"hidden_size = {hidden_size}")
    
    # 如果不需要某个维度，可以用下划线忽略
    bsz, seq_len, _ = x.shape
    print(f"\n忽略最后一个维度:")
    print(f"bsz = {bsz}, seq_len = {seq_len}")
    
    # 演示不同形状的张量
    print(f"\n不同形状的张量解包:")
    shapes = [(1, 5, 8), (3, 2, 6), (4, 1, 10)]
    for shape in shapes:
        tensor = torch.randn(shape)
        bsz, seq_len, hidden_size = tensor.shape
        print(f"形状 {shape} -> bsz={bsz}, seq_len={seq_len}, hidden_size={hidden_size}")


def demo_torch_cat():
    """演示 torch.cat 拼接操作"""
    print("\n" + "=" * 50)
    print("2. torch.cat 拼接演示")
    print("=" * 50)
    
    # 创建两个张量
    tensor1 = torch.tensor([[1, 2, 3],
                           [4, 5, 6]])
    tensor2 = torch.tensor([[7, 8, 9],
                           [10, 11, 12]])
    
    print(f"张量1:\n{tensor1}")
    print(f"张量2:\n{tensor2}")
    
    # 沿 dim=0 (行方向) 拼接
    cat_dim0 = torch.cat([tensor1, tensor2], dim=0)
    print(f"\n沿 dim=0 拼接 (垂直拼接):")
    print(f"结果形状: {cat_dim0.shape}")
    print(f"结果内容:\n{cat_dim0}")
    
    # 沿 dim=1 (列方向) 拼接
    cat_dim1 = torch.cat([tensor1, tensor2], dim=1)
    print(f"\n沿 dim=1 拼接 (水平拼接):")
    print(f"结果形状: {cat_dim1.shape}")
    print(f"结果内容:\n{cat_dim1}")
    
    # 演示 KV 缓存拼接 (类似注意力机制中的用法)
    print(f"\nKV 缓存拼接演示 (类似注意力机制):")
    # 模拟历史 KV 缓存
    past_k = torch.randn(2, 3, 2)  # [batch, past_seq_len, head_dim]
    past_v = torch.randn(2, 3, 2)
    
    # 当前时间步的 K, V
    current_k = torch.randn(2, 1, 2)  # [batch, current_seq_len, head_dim]
    current_v = torch.randn(2, 1, 2)
    
    print(f"历史 K 形状: {past_k.shape}")
    print(f"当前 K 形状: {current_k.shape}")
    
    # 拼接 (沿序列长度维度)
    new_k = torch.cat([past_k, current_k], dim=1)
    new_v = torch.cat([past_v, current_v], dim=1)
    
    print(f"拼接后 K 形状: {new_k.shape}")
    print(f"拼接后 V 形状: {new_v.shape}")
    
    # 演示多个张量拼接
    print(f"\n多个张量拼接:")
    tensors = [torch.randn(2, 1, 3) for _ in range(3)]
    for i, t in enumerate(tensors):
        print(f"张量{i+1} 形状: {t.shape}")
    
    multi_cat = torch.cat(tensors, dim=1)
    print(f"拼接后形状: {multi_cat.shape}")


def demo_transpose():
    """演示 transpose 转置操作"""
    print("\n" + "=" * 50)
    print("3. transpose 转置演示")
    print("=" * 50)
    
    # 创建一个 3D 张量 [batch, seq_len, hidden_size]
    x = torch.randn(2, 3, 4)
    print(f"原始张量形状: {x.shape}")
    print(f"原始张量内容:\n{x}")
    
    # transpose(1, 2) - 交换第1和第2个维度
    x_transposed = x.transpose(1, 2)
    print(f"\ntranspose(1, 2) 后形状: {x_transposed.shape}")
    print(f"转置后内容:\n{x_transposed}")
    
    # 详细解释维度变化
    print(f"\n维度变化解释:")
    print(f"原始: [batch={x.shape[0]}, seq_len={x.shape[1]}, hidden_size={x.shape[2]}]")
    print(f"转置: [batch={x_transposed.shape[0]}, hidden_size={x_transposed.shape[1]}, seq_len={x_transposed.shape[2]}]")
    
    # 演示注意力机制中的典型用法
    print(f"\n注意力机制中的典型用法:")
    # 模拟 Q, K, V 张量 [batch, seq_len, num_heads, head_dim]
    q = torch.randn(2, 4, 3, 2)  # batch=2, seq_len=4, num_heads=3, head_dim=2
    k = torch.randn(2, 4, 3, 2)
    v = torch.randn(2, 4, 3, 2)
    
    print(f"Q 形状: {q.shape}")
    print(f"K 形状: {k.shape}")
    print(f"V 形状: {v.shape}")
    
    # 转置为 [batch, num_heads, seq_len, head_dim] (注意力计算需要的格式)
    q_t = q.transpose(1, 2)  # [2, 3, 4, 2]
    k_t = k.transpose(1, 2)  # [2, 3, 4, 2]
    v_t = v.transpose(1, 2)  # [2, 3, 4, 2]
    
    print(f"\n转置后:")
    print(f"Q_t 形状: {q_t.shape}")
    print(f"K_t 形状: {k_t.shape}")
    print(f"V_t 形状: {v_t.shape}")
    
    # 演示注意力分数计算
    print(f"\n注意力分数计算:")
    # Q @ K^T (需要 K 再转置一次)
    k_t_transposed = k_t.transpose(-2, -1)  # [2, 3, 2, 4]
    print(f"K 转置后形状: {k_t_transposed.shape}")
    
    # 计算注意力分数
    scores = q_t @ k_t_transposed  # [2, 3, 4, 4]
    print(f"注意力分数形状: {scores.shape}")
    
    # 应用注意力权重到 V
    output = scores @ v_t  # [2, 3, 4, 2]
    print(f"注意力输出形状: {output.shape}")
    
    # 转置回原始格式
    output_final = output.transpose(1, 2)  # [2, 4, 3, 2]
    print(f"最终输出形状: {output_final.shape}")


def demo_combined_operations():
    """演示这些操作的组合使用"""
    print("\n" + "=" * 50)
    print("4. 组合操作演示 (模拟注意力机制)")
    print("=" * 50)
    
    # 模拟输入
    batch_size, seq_len, hidden_size = 2, 4, 6
    num_heads = 2
    head_dim = hidden_size // num_heads
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    print(f"输入张量形状: {x.shape}")
    
    # 模拟线性投影
    q_proj = torch.randn(hidden_size, hidden_size)
    k_proj = torch.randn(hidden_size, hidden_size)
    v_proj = torch.randn(hidden_size, hidden_size)
    
    # 计算 Q, K, V
    q = x @ q_proj  # [batch, seq_len, hidden_size]
    k = x @ k_proj
    v = x @ v_proj
    
    print(f"Q 形状: {q.shape}")
    
    # 重塑为多头格式
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, num_heads, head_dim)
    v = v.view(batch_size, seq_len, num_heads, head_dim)
    
    print(f"重塑后 Q 形状: {q.shape}")
    
    # 转置为注意力计算格式
    q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    print(f"转置后 Q 形状: {q.shape}")
    
    # 模拟 KV 缓存拼接
    past_k = torch.randn(batch_size, num_heads, 2, head_dim)  # 历史 2 个时间步
    past_v = torch.randn(batch_size, num_heads, 2, head_dim)
    
    print(f"历史 K 形状: {past_k.shape}")
    print(f"当前 K 形状: {k.shape}")
    
    # 拼接 KV 缓存
    k = torch.cat([past_k, k], dim=2)  # 沿序列长度维度拼接
    v = torch.cat([past_v, v], dim=2)
    
    print(f"拼接后 K 形状: {k.shape}")
    
    # 计算注意力分数
    scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)
    print(f"注意力分数形状: {scores.shape}")
    
    # 应用 softmax 和注意力权重
    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ v
    print(f"注意力输出形状: {output.shape}")
    
    # 转置回原始格式
    output = output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
    print(f"转置回原始格式: {output.shape}")
    
    # 重塑回原始隐藏维度
    output = output.reshape(batch_size, seq_len, hidden_size)
    print(f"最终输出形状: {output.shape}")


def main():
    """主函数"""
    print("PyTorch 张量操作详细演示")
    print("这些操作在 Transformer 注意力机制中经常使用")
    
    # demo_shape_unpacking()
    # demo_torch_cat()
    # demo_transpose()
    demo_combined_operations()
    
    print("\n" + "=" * 50)
    print("总结:")
    print("1. bsz, seq_len, _ = x.shape - 解包张量形状，获取维度信息")
    print("2. torch.cat - 沿指定维度拼接张量，常用于 KV 缓存")
    print("3. transpose - 交换张量维度，注意力计算需要特定维度顺序")
    print("=" * 50)


if __name__ == "__main__":
    main()
