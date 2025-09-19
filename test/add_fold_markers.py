#!/usr/bin/env python3
"""
为 Python 文件添加代码折叠标记的脚本
支持多种编辑器的折叠语法
"""

import re
import sys
import argparse

def add_fold_markers(content, marker_type="vscode"):
    """
    为 Python 代码添加折叠标记
    
    Args:
        content: 文件内容
        marker_type: 标记类型 ("vscode", "pycharm", "sublime", "manual")
    """
    
    lines = content.split('\n')
    result = []
    
    # 定义不同编辑器的折叠标记
    markers = {
        "vscode": ("# region", "# endregion"),
        "pycharm": ("# region", "# endregion"), 
        "sublime": ("# {{", "# }}"),
        "manual": ("# {{{", "# }}}")
    }
    
    start_marker, end_marker = markers.get(marker_type, markers["vscode"])
    
    # 查找函数和类定义
    function_pattern = r'^(\s*)def\s+(\w+)\s*\('
    class_pattern = r'^(\s*)class\s+(\w+)'
    
    for i, line in enumerate(lines):
        # 检查是否是函数定义
        func_match = re.match(function_pattern, line)
        class_match = re.match(class_pattern, line)
        
        if func_match:
            indent = func_match.group(1)
            func_name = func_match.group(2)
            result.append(f"{indent}{start_marker} {func_name}")
            result.append(line)
            
            # 找到函数结束位置
            func_end = find_function_end(lines, i)
            if func_end > i:
                # 添加结束标记
                result.append(f"{indent}{end_marker}")
            else:
                result.append(f"{indent}{end_marker}")
                
        elif class_match:
            indent = class_match.group(1)
            class_name = class_match.group(2)
            result.append(f"{indent}{start_marker} {class_name}")
            result.append(line)
            
            # 找到类结束位置
            class_end = find_class_end(lines, i)
            if class_end > i:
                # 添加结束标记
                result.append(f"{indent}{end_marker}")
            else:
                result.append(f"{indent}{end_marker}")
        else:
            result.append(line)
    
    return '\n'.join(result)

def find_function_end(lines, start_idx):
    """找到函数定义的结束位置"""
    if start_idx >= len(lines):
        return start_idx
    
    # 获取函数定义的缩进
    func_line = lines[start_idx]
    func_indent = len(func_line) - len(func_line.lstrip())
    
    # 查找下一个同级别或更少缩进的非空行
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].rstrip()
        if line:  # 非空行
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= func_indent:
                return i - 1
    
    return len(lines) - 1

def find_class_end(lines, start_idx):
    """找到类定义的结束位置"""
    if start_idx >= len(lines):
        return start_idx
    
    # 获取类定义的缩进
    class_line = lines[start_idx]
    class_indent = len(class_line) - len(class_line.lstrip())
    
    # 查找下一个同级别或更少缩进的非空行
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].rstrip()
        if line:  # 非空行
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= class_indent:
                return i - 1
    
    return len(lines) - 1

def main():
    parser = argparse.ArgumentParser(description="为 Python 文件添加代码折叠标记")
    parser.add_argument("input_file", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（默认覆盖原文件）")
    parser.add_argument("-t", "--type", choices=["vscode", "pycharm", "sublime", "manual"], 
                       default="vscode", help="折叠标记类型")
    parser.add_argument("--backup", action="store_true", help="创建备份文件")
    
    args = parser.parse_args()
    
    try:
        # 读取文件
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建备份
        if args.backup:
            backup_file = args.input_file + '.backup'
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"备份文件已创建: {backup_file}")
        
        # 添加折叠标记
        new_content = add_fold_markers(content, args.type)
        
        # 写入文件
        output_file = args.output or args.input_file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"折叠标记已添加到: {output_file}")
        print(f"使用标记类型: {args.type}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
