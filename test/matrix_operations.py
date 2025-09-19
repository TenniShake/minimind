import torch

def main():
    """演示 PyTorch 中逐元素相除和矩阵相乘的区别"""
    
    # 创建两个 2x2 矩阵
    A = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    
    B = torch.tensor([[2.0, 1.0],
                      [1.0, 2.0]])
    
    print("矩阵 A:")
    print(A)
    print("\n矩阵 B:")
    print(B)
    
    # 逐元素相除 (element-wise division)
    element_wise_division = A / B
    print("\n逐元素相除 A / B:")
    print(element_wise_division)
    print("计算过程:")
    print(f"A[0,0] / B[0,0] = {A[0,0]} / {B[0,0]} = {A[0,0] / B[0,0]}")
    print(f"A[0,1] / B[0,1] = {A[0,1]} / {B[0,1]} = {A[0,1] / B[0,1]}")
    print(f"A[1,0] / B[1,0] = {A[1,0]} / {B[1,0]} = {A[1,0] / B[1,0]}")
    print(f"A[1,1] / B[1,1] = {A[1,1]} / {B[1,1]} = {A[1,1] / B[1,1]}")
    
    # 矩阵相乘 (matrix multiplication)
    matrix_multiplication = A @ B
    print("\n矩阵相乘 A @ B:")
    print(matrix_multiplication)
    print("计算过程:")
    print(f"结果[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = {A[0,0]}*{B[0,0]} + {A[0,1]}*{B[1,0]} = {A[0,0]*B[0,0] + A[0,1]*B[1,0]}")
    print(f"结果[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = {A[0,0]}*{B[0,1]} + {A[0,1]}*{B[1,1]} = {A[0,0]*B[0,1] + A[0,1]*B[1,1]}")
    print(f"结果[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = {A[1,0]}*{B[0,0]} + {A[1,1]}*{B[1,0]} = {A[1,0]*B[0,0] + A[1,1]*B[1,0]}")
    print(f"结果[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = {A[1,0]}*{B[0,1]} + {A[1,1]}*{B[1,1]} = {A[1,0]*B[0,1] + A[1,1]*B[1,1]}")
    
    # 其他矩阵相乘的写法
    print("\n其他矩阵相乘写法:")
    print("torch.matmul(A, B):")
    print(torch.matmul(A, B))
    print("torch.mm(A, B):")
    print(torch.mm(A, B))
    
    # 对比逐元素相乘
    print("\n逐元素相乘 A * B (对比):")
    element_wise_multiply = A * B
    print(element_wise_multiply)
    print("计算过程:")
    print(f"A[0,0] * B[0,0] = {A[0,0]} * {B[0,0]} = {A[0,0] * B[0,0]}")
    print(f"A[0,1] * B[0,1] = {A[0,1]} * {B[0,1]} = {A[0,1] * B[0,1]}")
    print(f"A[1,0] * B[1,0] = {A[1,0]} * {B[1,0]} = {A[1,0] * B[1,0]}")
    print(f"A[1,1] * B[1,1] = {A[1,1]} * {B[1,1]} = {A[1,1] * B[1,1]}")

def matrix_operations():
    x = torch.tensor([[1, 2, 3, 4], 
                      [3, 4, 5, 6],
                      [5, 6, 7, 8],
                      [7, 8, 9, 10]
                      ])
    a = x[..., x.shape[-1] // 2:]

    b = x[..., : x.shape[-1] // 2]

    c = torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    print(a)
    print(b)
    print(c)

if __name__ == "__main__":
    matrix_operations()
