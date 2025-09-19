import argparse
import os
import re
from typing import List

from openpyxl import Workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a pipe-delimited TXT file to Excel (.xlsx)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .txt file (pipe-delimited)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output .xlsx file",
    )
    return parser.parse_args()


def read_and_tokenize_lines(input_path: str) -> List[List[str]]:
    # Read file lines and clean trailing newlines
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    # Drop a typical separator line (e.g., dashes and pipes) if present as the 2nd line
    lines: List[str] = raw_lines[:]
    if len(lines) >= 2:
        second_line = lines[1]
        stripped = second_line.replace("|", "").replace(" ", "")
        if stripped and set(stripped) == {"-"}:
            lines.pop(1)

    # Remove trailing pipe at end of line (to avoid creating an empty column)
    lines = [re.sub(r"\|\s*$", "", line) for line in lines]

    # Split by pipe, keep internal spaces as content, and strip outer spaces
    tokenized: List[List[str]] = []
    for line in lines:
        parts = [part.strip() for part in line.split("|")]
        tokenized.append(parts)
    return tokenized


def convert_txt_to_excel(input_path: str, output_path: str) -> None:
    rows = read_and_tokenize_lines(input_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    # Write rows; pad shorter rows so openpyxl aligns columns
    max_len = max((len(r) for r in rows), default=0)
    for r in rows:
        if len(r) < max_len:
            r = r + [""] * (max_len - len(r))
        ws.append(r)

    wb.save(output_path)


def main() -> None:
    args = parse_args()
    args.input = "../test/aa.txt"
    args.output = "../test/aa.xlsx"
    convert_txt_to_excel("test/aa.txt", "test/aa.xlsx")
    print(f"Wrote Excel: {os.path.abspath("test/aa.xlsx")}")


if __name__ == "__main__":
    main()


