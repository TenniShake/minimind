import torch

def demo_negative_indexing():
    """演示负数索引和转置操作"""
    print("=" * 60)
    print("负数索引和转置操作详细演示")
    print("=" * 60)
    
    # 创建一个 4D 张量用于演示
    tensor = torch.randn(2, 3, 4, 5)
    print(f"原始张量形状: {tensor.shape}")
    print(f"原始张量: [batch, num_heads, seq_len, head_dim]")
    
    print("\n" + "=" * 40)
    print("1. 负数索引的含义")
    print("=" * 40)
    
    print("正数索引: 0, 1, 2, 3")
    print("负数索引: -4, -3, -2, -1")
    print("\n对应关系:")
    print("索引 0 对应 索引 -4 (第1个维度)")
    print("索引 1 对应 索引 -3 (第2个维度)")
    print("索引 2 对应 索引 -2 (第3个维度)")
    print("索引 3 对应 索引 -1 (第4个维度)")
    
    print(f"\n张量形状: {tensor.shape}")
    print("维度说明:")
    for i, size in enumerate(tensor.shape):
        print(f"  维度 {i} (索引 -{len(tensor.shape)-i}): 大小 {size}")
    
    print("\n" + "=" * 40)
    print("2. transpose(-2, -1) 的转置过程")
    print("=" * 40)
    
    # 原始张量
    print(f"原始形状: {tensor.shape}")
    print("原始维度: [batch, num_heads, seq_len, head_dim]")
    
    # 转置
    transposed = tensor.transpose(-2, -1)
    print(f"\n转置后形状: {transposed.shape}")
    print("转置后维度: [batch, num_heads, head_dim, seq_len]")
    
    print("\n转置过程:")
    print("transpose(-2, -1) 表示交换倒数第2个和倒数第1个维度")
    print("倒数第2个维度: seq_len (索引 2)")
    print("倒数第1个维度: head_dim (索引 3)")
    print("结果: seq_len 和 head_dim 交换位置")
    
    print("\n" + "=" * 40)
    print("3. 不同转置方式的对比")
    print("=" * 40)
    
    # 方式1: 使用正数索引
    transposed_pos = tensor.transpose(2, 3)
    print(f"transpose(2, 3): {transposed_pos.shape}")
    
    # 方式2: 使用负数索引
    transposed_neg = tensor.transpose(-2, -1)
    print(f"transpose(-2, -1): {transposed_neg.shape}")
    
    # 验证结果是否相同
    print(f"两种方式结果相同: {torch.equal(transposed_pos, transposed_neg)}")
    
    print("\n" + "=" * 40)
    print("4. 为什么使用负数索引？")
    print("=" * 40)
    
    print("优势1: 更直观")
    print("- transpose(-2, -1) 表示交换最后两个维度")
    print("- 不需要知道张量的具体维度数")
    
    print("\n优势2: 代码更通用")
    print("- 无论张量有多少维度，-2, -1 总是表示最后两个维度")
    print("- 如果张量是 3D: [batch, seq_len, hidden_size]")
    print("- 如果张量是 4D: [batch, heads, seq_len, head_dim]")
    print("- 如果张量是 5D: [batch, heads, layers, seq_len, head_dim]")
    print("- transpose(-2, -1) 在所有情况下都表示交换最后两个维度")
    
    print("\n" + "=" * 40)
    print("5. 注意力机制中的实际应用")
    print("=" * 40)
    
    # 模拟注意力机制中的 K 张量
    batch_size, num_heads, seq_len, head_dim = 2, 3, 4, 2
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"K 张量形状: {k.shape}")
    print("K 张量维度: [batch, num_heads, seq_len, head_dim]")
    
    # 转置 K 用于计算注意力分数
    k_transposed = k.transpose(-2, -1)
    print(f"\n转置后 K 形状: {k_transposed.shape}")
    print("转置后 K 维度: [batch, num_heads, head_dim, seq_len]")
    
    print("\n为什么需要转置？")
    print("计算注意力分数: Q @ K^T")
    print("Q 形状: [batch, num_heads, seq_len, head_dim]")
    print("K 形状: [batch, num_heads, seq_len, head_dim]")
    print("K^T 形状: [batch, num_heads, head_dim, seq_len]")
    print("结果形状: [batch, num_heads, seq_len, seq_len]")
    
    print("\n" + "=" * 40)
    print("6. 其他转置示例")
    print("=" * 40)
    
    # 演示不同维度的转置
    tensor_3d = torch.randn(2, 3, 4)
    print(f"3D 张量形状: {tensor_3d.shape}")
    
    # 转置最后两个维度
    transposed_3d = tensor_3d.transpose(-2, -1)
    print(f"transpose(-2, -1) 后: {transposed_3d.shape}")
    
    # 转置前两个维度
    transposed_3d_01 = tensor_3d.transpose(0, 1)
    print(f"transpose(0, 1) 后: {transposed_3d_01.shape}")
    
    print("\n" + "=" * 40)
    print("7. 可视化转置过程")
    print("=" * 40)
    
    # 创建一个简单的张量用于可视化
    simple_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"简单张量形状: {simple_tensor.shape}")
    print(f"简单张量内容:\n{simple_tensor}")
    
    # 转置
    simple_transposed = simple_tensor.transpose(-2, -1)
    print(f"\n转置后形状: {simple_transposed.shape}")
    print(f"转置后内容:\n{simple_transposed}")
    
    print("\n转置过程:")
    print("原始: [2, 2, 2]")
    print("转置: [2, 2, 2] (交换最后两个维度)")
    print("具体变化:")
    print("simple_tensor[0, 0, :] = [1, 2] -> simple_transposed[0, :, 0] = [1, 3]")
    print("simple_tensor[0, 1, :] = [3, 4] -> simple_transposed[0, :, 1] = [2, 4]")


def demo_attention_transpose():
    """演示注意力机制中的转置操作"""
    print("\n" + "=" * 60)
    print("注意力机制中的转置操作")
    print("=" * 60)
    
    # 模拟注意力计算
    batch_size, num_heads, seq_len, head_dim = 2, 2, 3, 4
    
    # 创建 Q, K, V
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"Q 形状: {q.shape}")
    print(f"K 形状: {k.shape}")
    print(f"V 形状: {v.shape}")
    
    # 转置 K
    k_transposed = k.transpose(-2, -1)
    print(f"\nK 转置后形状: {k_transposed.shape}")
    
    # 计算注意力分数
    scores = q @ k_transposed
    print(f"注意力分数形状: {scores.shape}")
    
    print("\n计算过程:")
    print("Q: [batch, num_heads, seq_len, head_dim]")
    print("K^T: [batch, num_heads, head_dim, seq_len]")
    print("Q @ K^T: [batch, num_heads, seq_len, seq_len]")
    
    # 应用注意力权重
    output = scores @ v
    print(f"注意力输出形状: {output.shape}")
    
    print("\n最终输出:")
    print("scores @ V: [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]")
    print("结果: [batch, num_heads, seq_len, head_dim]")


def main():
    """主函数"""
    demo_negative_indexing()
    demo_attention_transpose()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("1. 负数索引从后往前计数: -1 是最后一个维度")
    print("2. transpose(-2, -1) 交换最后两个维度")
    print("3. 使用负数索引使代码更通用，不依赖具体维度数")
    print("4. 在注意力机制中，K 需要转置才能与 Q 进行矩阵乘法")
    print("=" * 60)


if __name__ == "__main__":
    main()

