import pandas as pd
import polars as pl
import time

import torch
def test_function():
    start = time.time()
    df = pd.read_json('D:\\Downloads\\sft_mini_512.jsonl', lines=True)
    total = len(df)
    print(total)
    print(df.head(5))
    end = time.time()
    print(end - start)

def test1():
    start = time.time()
    df = pl.read_json('D:\\Downloads\\sft_mini_512.jsonl')
    total = len(df)
    print(total)
    print(df.head(5))
    end = time.time()
    print(end - start)


def test2(): 
    # 假设输入张量
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # RMSNorm 计算过程
    x_squared = x.pow(2)  # [[1, 4, 9], [16, 25, 36]]
    mean_squared = x_squared.mean(-1, keepdim=True)  # [[4.67], [25.67]]
    rms = torch.rsqrt(mean_squared + 1e-5)  # [[0.46], [0.20]]
    normalized = x * rms  # 归一化结果
    print(normalized)

if __name__ == '__main__':
    test2()