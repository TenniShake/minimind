import torch

def demo_multi_dim_matrix_multiply():
    """演示多维张量矩阵乘法的规则"""
    print("=" * 60)
    print("多维张量矩阵乘法规则演示")
    print("=" * 60)
    
    print("核心规则:")
    print("1. 矩阵乘法只作用于最后两个维度")
    print("2. 前面的维度保持不变")
    print("3. 最后两个维度必须满足矩阵乘法规则")
    
    print("\n" + "=" * 40)
    print("1. 基本规则演示")
    print("=" * 40)
    
    # 创建两个张量
    A = torch.randn(2, 3, 4, 5)  # [batch, heads, seq_len, head_dim]
    B = torch.randn(2, 3, 5, 6)  # [batch, heads, head_dim, output_dim]
    
    print(f"A 形状: {A.shape}")
    print(f"B 形状: {B.shape}")
    
    # 矩阵乘法
    C = A @ B
    print(f"A @ B 结果形状: {C.shape}")
    
    print("\n计算过程:")
    print("A: [2, 3, 4, 5]")
    print("B: [2, 3, 5, 6]")
    print("结果: [2, 3, 4, 6]")
    print("\n解释:")
    print("- 前面的维度 [2, 3] 保持不变")
    print("- 最后两个维度 [4, 5] @ [5, 6] = [4, 6]")
    print("- 最终结果: [2, 3, 4, 6]")
    
    print("\n" + "=" * 40)
    print("2. 不同维度的矩阵乘法")
    print("=" * 40)
    
    # 3D 张量
    A_3d = torch.randn(2, 4, 5)
    B_3d = torch.randn(2, 5, 6)
    C_3d = A_3d @ B_3d
    print(f"3D: {A_3d.shape} @ {B_3d.shape} = {C_3d.shape}")
    
    # 4D 张量
    A_4d = torch.randn(2, 3, 4, 5)
    B_4d = torch.randn(2, 3, 5, 6)
    C_4d = A_4d @ B_4d
    print(f"4D: {A_4d.shape} @ {B_4d.shape} = {C_4d.shape}")
    
    # 5D 张量
    A_5d = torch.randn(2, 3, 4, 5, 6)
    B_5d = torch.randn(2, 3, 4, 6, 7)
    C_5d = A_5d @ B_5d
    print(f"5D: {A_5d.shape} @ {B_5d.shape} = {C_5d.shape}")
    
    print("\n规律:")
    print("- 前面的维度都保持不变")
    print("- 只有最后两个维度进行矩阵乘法")
    print("- 结果形状 = 前面维度 + 矩阵乘法结果")
    
    print("\n" + "=" * 40)
    print("3. 注意力机制中的实际应用")
    print("=" * 40)
    
    # 模拟注意力计算
    batch_size, num_heads, seq_len, head_dim = 2, 3, 4, 5
    
    # Q, K, V 张量
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"Q 形状: {q.shape}")
    print(f"K 形状: {k.shape}")
    print(f"V 形状: {v.shape}")
    
    # 转置 K
    k_transposed = k.transpose(-2, -1)
    print(f"K 转置后形状: {k_transposed.shape}")
    
    # 计算注意力分数
    scores = q @ k_transposed
    print(f"注意力分数形状: {scores.shape}")
    
    print("\n计算过程:")
    print("Q: [batch, num_heads, seq_len, head_dim]")
    print("K^T: [batch, num_heads, head_dim, seq_len]")
    print("Q @ K^T: [batch, num_heads, seq_len, seq_len]")
    print("\n解释:")
    print("- 前面的维度 [batch, num_heads] 保持不变")
    print("- 最后两个维度 [seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]")
    print("- 最终结果: [batch, num_heads, seq_len, seq_len]")
    
    # 应用注意力权重
    output = scores @ v
    print(f"\n注意力输出形状: {output.shape}")
    
    print("\n计算过程:")
    print("scores: [batch, num_heads, seq_len, seq_len]")
    print("V: [batch, num_heads, seq_len, head_dim]")
    print("scores @ V: [batch, num_heads, seq_len, head_dim]")
    print("\n解释:")
    print("- 前面的维度 [batch, num_heads] 保持不变")
    print("- 最后两个维度 [seq_len, seq_len] @ [seq_len, head_dim] = [seq_len, head_dim]")
    print("- 最终结果: [batch, num_heads, seq_len, head_dim]")
    
    print("\n" + "=" * 40)
    print("4. 广播机制的影响")
    print("=" * 40)
    
    # 演示广播
    A = torch.randn(2, 3, 4, 5)
    B = torch.randn(5, 6)  # 只有最后两个维度
    
    print(f"A 形状: {A.shape}")
    print(f"B 形状: {B.shape}")
    
    C = A @ B
    print(f"A @ B 结果形状: {C.shape}")
    
    print("\n广播规则:")
    print("- B 会被广播为 [2, 3, 5, 6]")
    print("- 然后进行矩阵乘法: [2, 3, 4, 5] @ [2, 3, 5, 6] = [2, 3, 4, 6]")
    
    print("\n" + "=" * 40)
    print("5. 错误示例")
    print("=" * 40)
    
    # 尝试不兼容的矩阵乘法
    A = torch.randn(2, 3, 4, 5)
    B = torch.randn(2, 3, 6, 7)  # 最后两个维度不兼容
    
    print(f"A 形状: {A.shape}")
    print(f"B 形状: {B.shape}")
    
    try:
        C = A @ B
        print(f"A @ B 结果形状: {C.shape}")
    except RuntimeError as e:
        print(f"错误: {e}")
        print("原因: 最后两个维度不兼容")
        print("A 的最后两个维度: [4, 5]")
        print("B 的最后两个维度: [6, 7]")
        print("矩阵乘法要求: A 的最后一维 = B 的倒数第二维")
        print("这里: 5 ≠ 6，所以无法相乘")
    
    print("\n" + "=" * 40)
    print("6. 总结")
    print("=" * 40)
    
    print("多维张量矩阵乘法的规则:")
    print("1. 矩阵乘法只作用于最后两个维度")
    print("2. 前面的维度保持不变")
    print("3. 最后两个维度必须满足矩阵乘法规则")
    print("4. 支持广播机制")
    print("5. 结果形状 = 前面维度 + 矩阵乘法结果")
    
    print("\n记忆技巧:")
    print("- 想象成批量处理多个矩阵")
    print("- 每个矩阵都是最后两个维度")
    print("- 前面的维度只是批次的标识")


def demo_attention_mechanism():
    """演示注意力机制中的矩阵乘法"""
    print("\n" + "=" * 60)
    print("注意力机制中的矩阵乘法")
    print("=" * 60)
    
    # 模拟完整的注意力计算
    batch_size, num_heads, seq_len, head_dim = 2, 3, 4, 5
    
    print(f"参数设置:")
    print(f"batch_size = {batch_size}")
    print(f"num_heads = {num_heads}")
    print(f"seq_len = {seq_len}")
    print(f"head_dim = {head_dim}")
    
    # 创建 Q, K, V
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"\n输入张量:")
    print(f"Q: {q.shape}")
    print(f"K: {k.shape}")
    print(f"V: {v.shape}")
    
    # 步骤1: 转置 K
    k_transposed = k.transpose(-2, -1)
    print(f"\n步骤1: 转置 K")
    print(f"K^T: {k_transposed.shape}")
    
    # 步骤2: 计算注意力分数
    scores = q @ k_transposed
    print(f"\n步骤2: 计算注意力分数")
    print(f"Q @ K^T: {scores.shape}")
    print("解释: [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]")
    print("结果: [batch, num_heads, seq_len, seq_len]")
    
    # 步骤3: 应用 softmax
    attn_weights = torch.softmax(scores, dim=-1)
    print(f"\n步骤3: 应用 softmax")
    print(f"attn_weights: {attn_weights.shape}")
    
    # 步骤4: 应用注意力权重
    output = attn_weights @ v
    print(f"\n步骤4: 应用注意力权重")
    print(f"attn_weights @ V: {output.shape}")
    print("解释: [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]")
    print("结果: [batch, num_heads, seq_len, head_dim]")
    
    print(f"\n最终输出形状: {output.shape}")
    print("与输入 Q 的形状相同！")


def main():
    """主函数"""
    demo_multi_dim_matrix_multiply()
    demo_attention_mechanism()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("你的理解完全正确！")
    print("1. 多维张量矩阵乘法只作用于最后两个维度")
    print("2. 前面的维度保持不变")
    print("3. 结果形状 = 前面维度 + 矩阵乘法结果")
    print("4. 这是 PyTorch 中批量处理矩阵的标准方式")
    print("=" * 60)


if __name__ == "__main__":
    main()
