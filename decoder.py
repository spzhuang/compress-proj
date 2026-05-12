#!/usr/bin/env python3
"""
Decoder V7 - 统一解压缩工具（.py / .js / .svg / .png / 通用代码）

根据文件类型自动选择解码方法：
  'p' -> Token 级解压 + 源码还原 (Python)
  'c' -> Token 级解压 + 源码还原 (通用代码 JS/C++/Java/Go/Rust...)
  's' -> 直接 UTF-8 解码（SVG 为文本变换压缩）
  'g' -> K-Means + Color RLE V2 解压为 PNG

Usage:
    python decoder.py <input.compress> [output_dir]
"""

import sys
import os
import re
import struct
import tokenize
import keyword
import lzma
import argparse
from pathlib import Path

from util import (
    generate_short_names, get_index_bytes, read_index,
    decode_png_from_stream,
    restore_latex_constants, decode_latex_const_table,
    decode_md_stream, decode_md_pattern_table, restore_md_patterns,
)


# ============================================================
#  Python 解码器
# ============================================================

def decode_dict(data):
    """从紧凑二进制解码映射字典"""
    pos = 0
    mapping = {"v": "6"}

    for key_char, key_name in [('k', 'kw'), ('b', 'bi'), ('o', 'op'), ('n', 'nu')]:
        assert data[pos] == ord(key_char)
        count = struct.unpack('>H', data[pos+1:pos+3])[0]
        pos += 3
        items = []
        for _ in range(count):
            slen = struct.unpack('>H', data[pos:pos+2])[0]
            pos += 2
            items.append(data[pos:pos+slen].decode('utf-8'))
            pos += slen
        mapping[key_name] = items

    assert data[pos] == ord('i')
    count = struct.unpack('>H', data[pos+1:pos+3])[0]
    pos += 3
    id_list = []
    for _ in range(count):
        orig_len = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2
        id_list.append(data[pos:pos+orig_len].decode('utf-8'))
        pos += orig_len

    short_names = generate_short_names(len(id_list))
    id_map = {}
    for i, orig in enumerate(id_list):
        id_map[short_names[i]] = orig
    mapping['id'] = id_map

    assert data[pos] == ord('s')
    count = struct.unpack('>H', data[pos+1:pos+3])[0]
    pos += 3
    st_list = []
    for _ in range(count):
        orig_len = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2
        st_list.append(data[pos:pos+orig_len].decode('utf-8'))
        pos += orig_len

    st_map = {}
    for i, orig in enumerate(st_list):
        st_map[f"s{i}"] = orig
    mapping['st'] = st_map

    return mapping


def decode_stream_to_lines(encoded, mapping):
    """解码字节流为 [(indent, [tokens]), ...]"""
    kw_list = mapping['kw']
    bi_list = mapping['bi']
    op_list = mapping['op']
    nu_list = mapping['nu']
    id_list = [orig for short, orig in mapping['id'].items()]
    st_list = [orig for short, orig in mapping['st'].items()]

    kw_b = get_index_bytes(len(kw_list))
    bi_b = get_index_bytes(len(bi_list))
    op_b = get_index_bytes(len(op_list))
    nu_b = get_index_bytes(len(nu_list))
    id_b = get_index_bytes(len(id_list))
    st_b = get_index_bytes(len(st_list))

    lines = []
    tokens = []
    current_indent = 0
    pos = 0

    while pos < len(encoded):
        b = encoded[pos]

        if b == 0xFF or b >= 0x80:
            if tokens:
                lines.append((current_indent, tokens))
                tokens = []
            if b >= 0x80 and b != 0xFF:
                current_indent = b - 0x80
            pos += 1
        else:
            tc = chr(b)
            pos += 1

            if tc == 'k':
                idx, pos = read_index(encoded, pos, kw_b)
                tokens.append((tokenize.NAME, kw_list[idx]))
            elif tc == 'b':
                idx, pos = read_index(encoded, pos, bi_b)
                tokens.append((tokenize.NAME, bi_list[idx]))
            elif tc == 'o':
                idx, pos = read_index(encoded, pos, op_b)
                tokens.append((tokenize.OP, op_list[idx]))
            elif tc == 'i':
                idx, pos = read_index(encoded, pos, id_b)
                tokens.append((tokenize.NAME, id_list[idx]))
            elif tc == 's':
                idx, pos = read_index(encoded, pos, st_b)
                tokens.append((tokenize.STRING, st_list[idx]))
            elif tc == 'n':
                idx, pos = read_index(encoded, pos, nu_b)
                tokens.append((tokenize.NUMBER, nu_list[idx]))
            elif tc == 'r':
                slen = struct.unpack('>H', encoded[pos:pos+2])[0]
                pos += 2
                s = encoded[pos:pos+slen].decode('utf-8')
                pos += slen
                tokens.append((tokenize.NAME, s))

    if tokens:
        lines.append((current_indent, tokens))

    return lines


def restore_from_lines(lines):
    """从行列表还原 Python 源代码"""
    result = []

    for indent, tokens in lines:
        if indent > 0:
            result.append(' ' * indent)

        prev_type = None
        prev_str = None

        for ttype, tstr in tokens:
            need_space = False

            if prev_type is not None:
                if prev_type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING) and \
                   ttype in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
                    need_space = True

                if prev_type == tokenize.NAME and prev_str in keyword.kwlist:
                    if ttype not in (tokenize.OP, tokenize.NEWLINE, tokenize.NL):
                        need_space = True
                    if ttype == tokenize.OP and tstr in ('(', '[', '{'):
                        need_space = False

                if ttype == tokenize.OP and tstr in ('=', '+=', '-=', '*=', '/=', '//=', '%=', '**=', ':='):
                    need_space = True
                if prev_type == tokenize.OP and prev_str in ('=', '+=', '-=', '*=', '/=', '//=', '%=', '**=', ':='):
                    need_space = True

                if ttype == tokenize.OP and tstr in ('==', '!=', '<=', '>=', '<', '>'):
                    need_space = True
                if prev_type == tokenize.OP and prev_str in ('==', '!=', '<=', '>=', '<', '>'):
                    need_space = True

                if prev_type == tokenize.OP and prev_str == ',':
                    if ttype not in (tokenize.OP, tokenize.NEWLINE, tokenize.NL) or \
                       (ttype == tokenize.OP and tstr not in (')', ']', '}')):
                        need_space = True

                if prev_type == tokenize.OP and prev_str == ':':
                    if ttype not in (tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT):
                        if not (ttype == tokenize.OP and tstr in (')', ']', '}', ',')):
                            need_space = True

                if prev_type == tokenize.OP and prev_str in (')', ']', '}'):
                    if ttype in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
                        need_space = True

            if need_space:
                result.append(' ')

            result.append(tstr)
            prev_type = ttype
            prev_str = tstr

        result.append('\n')

    return ''.join(result)


# ============================================================
#  通用代码解码器（类 C 语法还原）
# ============================================================

# 类 C 语言关键字集合
_KEYWORD_NEED_SPACE_AFTER = {
    'if', 'else', 'for', 'while', 'switch', 'case', 'return', 'throw',
    'catch', 'try', 'finally', 'do', 'yield', 'await', 'async', 'new',
    'delete', 'typeof', 'instanceof', 'void', 'in', 'of', 'as', 'from',
    'import', 'export', 'extends', 'implements', 'class', 'interface',
    'enum', 'struct', 'union', 'typedef', 'namespace', 'using', 'template',
    'public', 'private', 'protected', 'static', 'const', 'let', 'var',
    'function', 'def', 'fn', 'func', 'fun', 'type', 'alias', 'declare',
    'abstract', 'virtual', 'override', 'final', 'mutable', 'volatile',
    'throws', 'throw', 'goto', 'break', 'continue', 'lazy', 'defer',
    'guard', 'where', 'until', 'unless', 'elsif', 'elif',
    'match', 'with', 'lambda', 'assert', 'raise', 'except', 'rescue',
    'ensure', 'module', 'package',
}

# 关键字后跟 `{` 时需要空格（如 import/export/from 后的解构语法）
_KEYWORD_SPACE_BEFORE_BRACE = {'import', 'export', 'from', 'as'}

# 二元运算符前后需要空格
_BINARY_OPS = {
    '=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=',
    '<<=', '>>=', '>>>',
    '+', '-', '*', '/', '%',
    '&&', '||', '&', '|', '^', '<<', '>>', '>>>',
    '==', '!=', '===', '!==', '<', '>', '<=', '>=',
    '=>', '->',
}

def restore_generic_lines(lines):
    """
    从行列表还原类 C 语言源代码。

    根据类 C 语法规则在 token 之间插入适当的空格。
    保留原始的缩进信息。
    """
    result = []

    for indent, tokens in lines:
        if indent > 0:
            result.append(' ' * indent)

        prev_type = None
        prev_str = None

        for ttype, tstr in tokens:
            need_space = False

            if prev_type is not None:
                # === 基本规则 ===

                # R1: 相邻的 NAME/NUMBER/STRING 之间需要空格
                if prev_type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING) and \
                   ttype in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
                    need_space = True

                # R2: 关键字后跟标识符/字符串/数字通常需要空格
                if prev_type == tokenize.NAME and prev_str in _KEYWORD_NEED_SPACE_AFTER:
                    if ttype == tokenize.OP and tstr == '(':
                        # 关键字后跟 ( : if( / for( 格式都可以，不加空格
                        need_space = False
                    elif ttype == tokenize.OP and tstr == '{':
                        # 关键字后跟 { : import { 需要空格
                        if prev_str in _KEYWORD_SPACE_BEFORE_BRACE:
                            need_space = True
                        else:
                            need_space = False
                    elif ttype == tokenize.OP:
                        # 关键字后跟其他运算符: 通常需要空格（但 . :: < > 不需要）
                        if tstr not in ('.', '::', '<', '>', '[', ']', '(', ')', ';', ',', '{'):
                            need_space = True
                    else:
                        need_space = True

                # R3: 二元运算符前后需要空格（但 . :: 前后永远不需要）
                # 注意: < > 在 include/pragma 后由 R13 处理，这里排除
                if ttype == tokenize.OP and tstr in _BINARY_OPS and tstr not in ('<', '>'):
                    if prev_type == tokenize.OP and prev_str in ('.', '::', '(', '[', '{', '@'):
                        need_space = False
                    else:
                        need_space = True
                if prev_type == tokenize.OP and prev_str in _BINARY_OPS and prev_str not in ('<', '>'):
                    if ttype == tokenize.OP and tstr in ('.', '::', ')', ']', '}', ',', ';', ':'):
                        need_space = False
                    else:
                        need_space = True

                # R4: 逗号后通常需要空格（除非下一个是 ) ] }）
                if prev_type == tokenize.OP and prev_str == ',':
                    if ttype == tokenize.OP and tstr in (')', ']', '}'):
                        need_space = False
                    else:
                        need_space = True

                # R5: 冒号: 对象字面量 key: value 需要空格；三元运算符 ? a : b 需要空格
                if prev_type == tokenize.OP and prev_str == ':':
                    if ttype == tokenize.OP and tstr in (')', ']', '}', ','):
                        need_space = False
                    else:
                        need_space = True

                # R6: 分号后需要空格
                if prev_type == tokenize.OP and prev_str == ';':
                    need_space = True

                # R7: ) ] } 后跟 NAME/NUMBER/STRING 通常需要空格
                if prev_type == tokenize.OP and prev_str in (')', ']', '}'):
                    if ttype in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
                        if not (ttype == tokenize.OP and tstr == '.'):
                            need_space = True

                # R8: . 和 :: 前后永远不需要空格
                if ttype == tokenize.OP and tstr in ('.', '::'):
                    need_space = False
                if prev_type == tokenize.OP and prev_str in ('.', '::'):
                    need_space = False

                # R9: 链式调用: )(. )[. }(. ]( 等不需要空格
                if ttype == tokenize.OP and tstr == '(':
                    if prev_type == tokenize.OP and prev_str in (')', ']'):
                        need_space = False
                if ttype == tokenize.OP and tstr == '.':
                    if prev_type == tokenize.OP and prev_str in (')', ']'):
                        need_space = False

                # R10: NAME 后跟 ( 不需要空格（函数调用）
                if ttype == tokenize.OP and tstr == '(':
                    if prev_type == tokenize.NAME:
                        if prev_str not in _KEYWORD_NEED_SPACE_AFTER:
                            need_space = False

                # R11: @ 前需要空格（装饰器语法）
                if ttype == tokenize.OP and tstr == '@':
                    need_space = True

                # R12: # 在 C 预处理中紧跟标识符不需要空格
                if ttype == tokenize.NAME:
                    if prev_type == tokenize.OP and prev_str == '#':
                        need_space = False

                # R13: include/pragma 后的 < 和 > 不需要空格（覆盖 R3）
                # 也覆盖 #include <iostream> 中头文件名后的 >
                if ttype == tokenize.OP and tstr in ('<', '>'):
                    if prev_str and ('include' in prev_str or 'pragma' in prev_str):
                        need_space = False
                    elif tstr == '>' and prev_type == tokenize.NAME:
                        # 检查是否在 #include <...> 的上下文中（简化：> 紧跟 NAME 时不加空格）
                        need_space = False

            result.append(' ' if need_space else '')
            result.append(tstr)
            prev_type = ttype
            prev_str = tstr

        result.append('\n')

    return ''.join(result)


# ============================================================
#  统一解包
# ============================================================

def _parse_type_suffix(filename):
    """从文件名解析压缩类型后缀，如 'test_py_md.compress' -> ['py', 'md']
    
    支持的文件名格式:
      - src.compress          -> None (无后缀，导入全部)
      - src_py.compress       -> ['py']
      - src_py_md.compress    -> ['py', 'md']
      - src_all.compress      -> ['all']
      - src_part1_md.compress -> ['md'] (跳过 partN 前缀)
    """
    # 去掉 .compress 后缀
    name = filename
    if name.endswith('.compress'):
        name = name[:-9]
    
    # 查找最后一个 _ 开始的后缀部分
    # 但需要跳过 _partN 前缀
    # 策略：找到 _all, 或找到第一个已知类型前缀
    KNOWN_TYPES = {'py', 'js', 'ts', 'java', 'go', 'rs', 'cpp', 'c', 'h',
                   'sh', 'svg', 'png', 'tex', 'md', 'txt', 'all'}
    
    # 从后向前查找，找到第一个已知类型
    parts = name.split('_')
    types = []
    found_type = False
    for part in reversed(parts):
        if part in KNOWN_TYPES:
            types.append(part)
            found_type = True
        elif found_type:
            # 已经找到类型，但当前 part 不是类型，停止
            break
        # 如果还没找到类型，继续向前搜索（跳过 part1, part2 等）
    
    if not types:
        return None  # 没有找到类型后缀，导入全部
    
    types.reverse()  # 恢复原始顺序
    return types


def _import_decoder_modules(type_list):
    """根据文件类型列表按需导入解码模块"""
    modules = {'base': True, 'svg': False, 'png': False, 'latex': False, 'md': False}
    
    if type_list is None or 'all' in type_list:
        # 导入所有模块
        modules = {k: True for k in modules}
    else:
        # 按需标记
        for t in type_list:
            if t in ('py', 'js', 'ts', 'c', 'cpp', 'java', 'go', 'rs', 'sh'):
                modules['base'] = True
            elif t == 'svg':
                modules['svg'] = True
            elif t == 'png':
                modules['png'] = True
            elif t == 'tex':
                modules['latex'] = True
            elif t in ('md', 'txt'):
                modules['md'] = True
    
    result = {}
    
    # base 始终需要（read_index, decode_dict 等）
    if modules['base'] or True:
        result['base'] = True
    
    if modules['png']:
        from util_svg_png import decode_png_from_stream
        result['decode_png'] = decode_png_from_stream
    
    if modules['latex']:
        from util_latex import restore_latex_constants, decode_latex_const_table
        result['restore_latex'] = restore_latex_constants
        result['decode_latex_table'] = decode_latex_const_table
    else:
        result['restore_latex'] = lambda text, cmap: text
        result['decode_latex_table'] = lambda data: {}
    
    if modules['md']:
        from util_md_stream import decode_md_stream, _MD_BLOCK_TYPE_NAMES
        from util_md_patterns import restore_md_patterns, decode_md_pattern_table
        result['decode_md_stream'] = decode_md_stream
        result['restore_md'] = restore_md_patterns
        result['decode_md_table'] = decode_md_pattern_table
        result['md_block_names'] = _MD_BLOCK_TYPE_NAMES
    
    return result


def decode_files(compress_path):
    """
    从压缩文件解码所有源文件。
    根据文件名后缀（如 .compress_py_md）按需导入解码模块。

    Returns:
        dict: {filename: (file_type, content)}
            file_type: 'source'  -> str (Python/SVG/LaTeX source)
                       'image'   -> PIL Image (PNG)
    """
    # 1. 从文件名解析类型，按需导入模块
    filename = os.path.basename(compress_path)
    type_list = _parse_type_suffix(filename)
    modules = _import_decoder_modules(type_list)
    
    with open(compress_path, 'rb') as f:
        compressed = f.read()

    decompressed = lzma.decompress(compressed)
    pos = 0

    # 1. 代码字典
    dict_len = struct.unpack('>I', decompressed[pos:pos+4])[0]
    pos += 4

    if dict_len > 0:
        mapping = decode_dict(decompressed[pos:pos+dict_len])
    else:
        mapping = None
    pos += dict_len

    # 2. LaTeX 常量表
    latex_const_len = struct.unpack('>I', decompressed[pos:pos+4])[0]
    pos += 4

    latex_const_map = {}
    if latex_const_len > 0:
        latex_const_map = decode_latex_const_table(decompressed[pos:pos+latex_const_len])
    pos += latex_const_len

    # 3. MD 模式常量表
    md_const_len = struct.unpack('>I', decompressed[pos:pos+4])[0]
    pos += 4

    md_pattern_map = {}
    if md_const_len > 0:
        md_pattern_map = decode_md_pattern_table(decompressed[pos:pos+md_const_len])
    pos += md_const_len

    # 4. 文件数量
    num_files = struct.unpack('>H', decompressed[pos:pos+2])[0]
    pos += 2

    file_headers = []
    for i in range(num_files):
        fname_len = struct.unpack('>H', decompressed[pos:pos+2])[0]
        pos += 2
        fname = decompressed[pos:pos+fname_len].decode('utf-8')
        pos += fname_len
        file_type = chr(decompressed[pos])
        pos += 1
        stream_len = struct.unpack('>I', decompressed[pos:pos+4])[0]
        pos += 4
        file_headers.append((fname, file_type, stream_len))

    results = {}
    for fname, file_type, stream_len in file_headers:
        encoded = decompressed[pos:pos+stream_len]
        pos += stream_len

        if file_type == 'p':
            lines = decode_stream_to_lines(encoded, mapping)
            source = restore_from_lines(lines)
            results[fname] = ('source', source)
        elif file_type == 'c':
            lines = decode_stream_to_lines(encoded, mapping)
            source = restore_generic_lines(lines)
            results[fname] = ('source', source)
        elif file_type == 's':
            # SVG: 直接解码为 UTF-8 文本
            source = encoded.decode('utf-8')
            results[fname] = ('source', source)
        elif file_type == 'g':
            # PNG: 按需解码
            if 'decode_png' in modules:
                image = modules['decode_png'](encoded)
            else:
                raise ValueError("PNG解码模块未加载，请使用后缀 .compress_png")
            results[fname] = ('image', image)
        elif file_type == 'l':
            text = encoded.decode('utf-8')
            if latex_const_map:
                text = modules['restore_latex'](text, latex_const_map)
            results[fname] = ('source', text)
        elif file_type == 'm':
            # Markdown 分块压缩解码: 按需解码
            if 'decode_md_stream' in modules:
                text = modules['decode_md_stream'](
                    encoded, md_pattern_map=md_pattern_map,
                    latex_const_map=latex_const_map
                )
            else:
                text = encoded.decode('utf-8')
            results[fname] = ('source', text)
        else:
            raise ValueError(f"未知文件类型: {file_type}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Decoder V7 - 统一解压缩 (PY/JS/C++/JAVA/SVG/PNG)'
    )
    parser.add_argument('input', help='输入 .compress 文件路径')
    parser.add_argument('output', nargs='?', default='output',
                        help='输出目录（默认: ./output）')

    args = parser.parse_args()

    compress_path = Path(args.input)
    if not compress_path.exists():
        print(f"Error: 文件不存在 {compress_path}")
        sys.exit(1)

    output_base = Path(args.output)

    results = decode_files(compress_path)

    if len(results) > 1:
        output_dir = output_base / compress_path.stem
    else:
        output_dir = output_base

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print("   Decompression Results")
    print(f"{'='*50}")

    for fname, (ftype, content) in results.items():
        out_path = output_dir / fname

        if ftype == 'source':
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(content)
            ext = Path(fname).suffix.lower()
            if ext == '.py':
                tag = '[PY]'
            elif ext == '.svg':
                tag = '[SVG]'
            elif ext == '.tex':
                tag = '[TEX]'
            elif ext == '.md':
                tag = '[MD]'
            else:
                tag = f'[{ext.lstrip(".")}]'
            print(f"  {tag:<8s} {fname} -> {out_path} ({len(content)} chars)")
        elif ftype == 'image':
            content.save(out_path, optimize=True)
            print(f"  [PNG]    {fname} -> {out_path} ({content.size[0]}x{content.size[1]})")

    print(f"{'='*50}")
    print(f"Done. All files restored to: {output_dir}")


if __name__ == '__main__':
    main()

