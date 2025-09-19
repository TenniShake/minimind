import torch

def demo_tensor_cat_dimensions():
    """详细演示不同维度张量的拼接操作"""
    print("=" * 60)
    print("张量拼接维度详细演示")
    print("=" * 60)
    
    # 创建两个张量
    past_k = torch.randn(2, 3, 2)  # [batch, past_seq_len, head_dim]
    current_k = torch.randn(2, 1, 2)  # [batch, current_seq_len, head_dim]
    
    print(f"past_k 形状: {past_k.shape}")
    print(f"past_k 内容:\n{past_k}")
    print(f"\ncurrent_k 形状: {current_k.shape}")
    print(f"current_k 内容:\n{current_k}")
    
    print("\n" + "=" * 40)
    print("1. 沿 dim=1 拼接 (正确的做法)")
    print("=" * 40)
    
    # 沿 dim=1 拼接
    cat_dim1 = torch.cat([past_k, current_k], dim=1)
    print(f"拼接后形状: {cat_dim1.shape}")
    print(f"拼接后内容:\n{cat_dim1}")
    
    print("\n拼接过程解释:")
    print("past_k[0, :, :] 和 current_k[0, :, :] 在 dim=1 上拼接")
    print("past_k[1, :, :] 和 current_k[1, :, :] 在 dim=1 上拼接")
    print("结果: [2, 3+1, 2] = [2, 4, 2]")
    
    print("\n" + "=" * 40)
    print("2. 尝试沿 dim=0 拼接 (会出错)")
    print("=" * 40)
    
    try:
        cat_dim0 = torch.cat([past_k, current_k], dim=0)
        print(f"拼接后形状: {cat_dim0.shape}")
        print(f"拼接后内容:\n{cat_dim0}")
    except RuntimeError as e:
        print(f"错误: {e}")
        print("原因: 除了 dim=0 外，其他维度必须相同")
    
    print("\n" + "=" * 40)
    print("3. 为什么不能 dim=0？")
    print("=" * 40)
    
    print("past_k 形状: [2, 3, 2]")
    print("current_k 形状: [2, 1, 2]")
    print("\n沿 dim=0 拼接需要:")
    print("- dim=1 相同: 3 ≠ 1 ❌")
    print("- dim=2 相同: 2 = 2 ✅")
    print("因为 dim=1 不同，所以无法拼接")
    
    print("\n" + "=" * 40)
    print("4. 正确的拼接方式")
    print("=" * 40)
    
    print("沿 dim=1 拼接需要:")
    print("- dim=0 相同: 2 = 2 ✅")
    print("- dim=2 相同: 2 = 2 ✅")
    print("只有 dim=1 可以不同，所以可以拼接")
    
    print("\n" + "=" * 40)
    print("5. 其他维度的拼接示例")
    print("=" * 40)
    
    # 演示不同维度的拼接
    print("示例1: 沿 dim=2 拼接")
    tensor_a = torch.randn(2, 3, 2)
    tensor_b = torch.randn(2, 3, 1)
    cat_dim2 = torch.cat([tensor_a, tensor_b], dim=2)
    print(f"tensor_a 形状: {tensor_a.shape}")
    print(f"tensor_b 形状: {tensor_b.shape}")
    print(f"拼接后形状: {cat_dim2.shape}")
    
    print("\n示例2: 沿 dim=0 拼接 (需要其他维度相同)")
    tensor_c = torch.randn(2, 3, 2)
    tensor_d = torch.randn(1, 3, 2)  # 只有 dim=0 不同
    cat_dim0_ok = torch.cat([tensor_c, tensor_d], dim=0)
    print(f"tensor_c 形状: {tensor_c.shape}")
    print(f"tensor_d 形状: {tensor_d.shape}")
    print(f"拼接后形状: {cat_dim0_ok.shape}")


def demo_attention_kv_cache():
    """演示注意力机制中 KV 缓存的拼接"""
    print("\n" + "=" * 60)
    print("注意力机制中 KV 缓存拼接演示")
    print("=" * 60)
    
    # 模拟注意力机制中的情况
    batch_size = 2
    num_heads = 3
    past_seq_len = 4
    current_seq_len = 1
    head_dim = 2
    
    # 历史 KV 缓存
    past_k = torch.randn(batch_size, num_heads, past_seq_len, head_dim)
    past_v = torch.randn(batch_size, num_heads, past_seq_len, head_dim)
    
    # 当前时间步的 KV
    current_k = torch.randn(batch_size, num_heads, current_seq_len, head_dim)
    current_v = torch.randn(batch_size, num_heads, current_seq_len, head_dim)
    
    print(f"past_k 形状: {past_k.shape}")
    print(f"current_k 形状: {current_k.shape}")
    
    # 拼接 KV 缓存
    new_k = torch.cat([past_k, current_k], dim=2)  # 沿序列长度维度拼接
    new_v = torch.cat([past_v, current_v], dim=2)
    
    print(f"\n拼接后 new_k 形状: {new_k.shape}")
    print(f"拼接后 new_v 形状: {new_v.shape}")
    
    print("\n拼接过程:")
    print("1. past_k: [batch, num_heads, past_seq_len, head_dim]")
    print("2. current_k: [batch, num_heads, current_seq_len, head_dim]")
    print("3. 沿 dim=2 (序列长度维度) 拼接")
    print("4. 结果: [batch, num_heads, past_seq_len + current_seq_len, head_dim]")
    
    # 展示具体的拼接过程
    print(f"\n具体拼接过程:")
    print(f"past_k[0, 0, :, :] 形状: {past_k[0, 0, :, :].shape}")
    print(f"current_k[0, 0, :, :] 形状: {current_k[0, 0, :, :].shape}")
    print(f"拼接后 [0, 0, :, :] 形状: {new_k[0, 0, :, :].shape}")


def demo_visual_representation():
    """用可视化方式展示拼接过程"""
    print("\n" + "=" * 60)
    print("拼接过程可视化")
    print("=" * 60)
    
    # 创建简单的张量用于演示
    past_k = torch.tensor([[[1, 2], [3, 4], [5, 6]]])  # [1, 3, 2]
    current_k = torch.tensor([[[7, 8]]])  # [1, 1, 2]
    
    print("past_k:")
    print("形状: [1, 3, 2]")
    print("内容:")
    for i in range(past_k.shape[1]):
        print(f"  [:, {i}, :] = {past_k[0, i, :].tolist()}")
    
    print("\ncurrent_k:")
    print("形状: [1, 1, 2]")
    print("内容:")
    for i in range(current_k.shape[1]):
        print(f"  [:, {i}, :] = {current_k[0, i, :].tolist()}")
    
    # 拼接
    result = torch.cat([past_k, current_k], dim=1)
    
    print("\n拼接结果 (沿 dim=1):")
    print("形状: [1, 4, 2]")
    print("内容:")
    for i in range(result.shape[1]):
        print(f"  [:, {i}, :] = {result[0, i, :].tolist()}")
    
    print("\n拼接过程:")
    print("past_k[0, 0, :] = [1, 2]")
    print("past_k[0, 1, :] = [3, 4]")
    print("past_k[0, 2, :] = [5, 6]")
    print("current_k[0, 0, :] = [7, 8]")
    print("拼接后:")
    print("result[0, 0, :] = [1, 2]  # 来自 past_k")
    print("result[0, 1, :] = [3, 4]  # 来自 past_k")
    print("result[0, 2, :] = [5, 6]  # 来自 past_k")
    print("result[0, 3, :] = [7, 8]  # 来自 current_k")


def main():
    """主函数"""
    demo_tensor_cat_dimensions()
    demo_attention_kv_cache()
    demo_visual_representation()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("1. torch.cat 要求除了拼接维度外，其他维度必须相同")
    print("2. (2, 3, 2) 和 (2, 1, 2) 只能沿 dim=1 拼接")
    print("3. 沿 dim=0 拼接需要 dim=1 和 dim=2 都相同")
    print("4. 在注意力机制中，通常沿序列长度维度拼接 KV 缓存")
    print("=" * 60)


if __name__ == "__main__":
    main()
