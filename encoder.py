#!/usr/bin/env python3
"""
Encoder V7 - 统一压缩工具（.py / .js / .svg / .png / 通用代码）

自动根据文件扩展名选择压缩方法：
  .py  -> Token 级压缩 + 全局字典映射 (Python tokenizer)
  .js/.ts/.c/.cpp/.java/.go/.rs... -> Token 级压缩 + 全局字典映射 (通用 tokenizer)
  .svg -> 文本变换压缩（清理 + 简化）
  .png -> K-Means 颜色量化 + Color RLE V2 + zlib

Usage:
    python encoder.py <file1.py> [file2.js file3.svg file4.png ...]
    python encoder.py -d <directory> [-o <output_dir>]
    python encoder.py photo.png --clusters 32
"""

import sys
import os
import re
import struct
import tokenize
import io
import collections
import keyword
import builtins
import lzma
import argparse
from pathlib import Path

from util import (
    generate_short_names, get_index_bytes, pack_index,
    encode_png_to_stream, CODE_EXTENSIONS, BUILTINS_MAP, LANGUAGE_MAP,
    replace_latex_constants, encode_latex_const_table, decode_latex_const_table,
    encode_md_stream, encode_md_pattern_table,
)


# ============================================================
#  Python 编码器（Token 级压缩）
# ============================================================

def get_line_groups(source):
    """按行分组 Token，去除注释和空行"""
    all_tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    lines = []
    current_tokens = []
    indent_stack = [0]

    for tok in all_tokens:
        if tok.type == tokenize.ENCODING:
            continue
        elif tok.type == tokenize.COMMENT:
            continue
        elif tok.type in (tokenize.NEWLINE, tokenize.NL):
            if current_tokens:
                lines.append((indent_stack[-1], current_tokens))
            current_tokens = []
        elif tok.type == tokenize.INDENT:
            indent_stack.append(len(tok.string))
        elif tok.type == tokenize.DEDENT:
            if len(indent_stack) > 1:
                indent_stack.pop()
        elif tok.type == tokenize.ENDMARKER:
            if current_tokens:
                lines.append((indent_stack[-1], current_tokens))
        else:
            current_tokens.append((tok.type, tok.string))

    return lines


def build_mapping_dict(sources):
    """构建 Python Token 全局映射字典"""
    keywords_set = set()
    builtins_set = set()
    operators_set = set()
    numbers_set = set()
    idents_counter = collections.Counter()
    strings_counter = collections.Counter()

    for source in sources.values():
        lines = get_line_groups(source)
        for indent, tokens in lines:
            for ttype, tstr in tokens:
                if ttype == tokenize.NAME:
                    if tstr in keyword.kwlist:
                        keywords_set.add(tstr)
                    elif tstr in dir(builtins) or tstr in ('True', 'False', 'None'):
                        builtins_set.add(tstr)
                    else:
                        idents_counter[tstr] += 1
                elif ttype == tokenize.OP:
                    operators_set.add(tstr)
                elif ttype == tokenize.STRING:
                    strings_counter[tstr] += 1
                elif ttype == tokenize.NUMBER:
                    numbers_set.add(tstr)

    short_names = generate_short_names(len(idents_counter))
    id_map = {}
    for i, (orig_name, _) in enumerate(idents_counter.most_common()):
        id_map[short_names[i]] = orig_name

    st_map = {}
    for i, (orig_str, count) in enumerate(strings_counter.most_common()):
        if count >= 2:
            st_map[f"s{i}"] = orig_str

    return {
        "v": "6",
        "kw": sorted(keywords_set),
        "bi": sorted(builtins_set),
        "op": sorted(operators_set),
        "nu": sorted(numbers_set, key=lambda s: len(s)),
        "id": id_map,
        "st": st_map,
    }


def encode_lines(lines, mapping):
    """将行组编码为紧凑字节流"""
    kw_index = {s: i for i, s in enumerate(mapping['kw'])}
    bi_index = {s: i for i, s in enumerate(mapping['bi'])}
    op_index = {s: i for i, s in enumerate(mapping['op'])}
    nu_index = {s: i for i, s in enumerate(mapping['nu'])}
    id_items = list(mapping['id'].items())
    id_index = {orig: i for i, (short, orig) in enumerate(id_items)}
    st_items = list(mapping['st'].items())
    st_index = {orig: i for i, (short, orig) in enumerate(st_items)}

    kw_b = get_index_bytes(len(mapping['kw']))
    bi_b = get_index_bytes(len(mapping['bi']))
    op_b = get_index_bytes(len(mapping['op']))
    nu_b = get_index_bytes(len(mapping['nu']))
    id_b = get_index_bytes(len(mapping['id']))
    st_b = get_index_bytes(len(mapping['st']))

    encoded = bytearray()
    prev_indent = 0

    for indent, tokens in lines:
        if indent == prev_indent:
            encoded.append(0xFF)
        else:
            encoded.append(0x80 + min(indent, 127))
            prev_indent = indent

        for ttype, tstr in tokens:
            if ttype == tokenize.NAME:
                if tstr in kw_index:
                    encoded.append(ord('k'))
                    encoded.extend(pack_index(kw_index[tstr], kw_b))
                elif tstr in bi_index:
                    encoded.append(ord('b'))
                    encoded.extend(pack_index(bi_index[tstr], bi_b))
                elif tstr in id_index:
                    encoded.append(ord('i'))
                    encoded.extend(pack_index(id_index[tstr], id_b))
                else:
                    encoded.append(ord('r'))
                    s = tstr.encode('utf-8')
                    encoded.extend(struct.pack('>H', len(s)))
                    encoded.extend(s)
            elif ttype == tokenize.OP:
                if tstr in op_index:
                    encoded.append(ord('o'))
                    encoded.extend(pack_index(op_index[tstr], op_b))
                else:
                    encoded.append(ord('r'))
                    s = tstr.encode('utf-8')
                    encoded.extend(struct.pack('>H', len(s)))
                    encoded.extend(s)
            elif ttype == tokenize.STRING:
                if tstr in st_index:
                    encoded.append(ord('s'))
                    encoded.extend(pack_index(st_index[tstr], st_b))
                else:
                    encoded.append(ord('r'))
                    s = tstr.encode('utf-8')
                    encoded.extend(struct.pack('>H', len(s)))
                    encoded.extend(s)
            elif ttype == tokenize.NUMBER:
                if tstr in nu_index:
                    encoded.append(ord('n'))
                    encoded.extend(pack_index(nu_index[tstr], nu_b))
                else:
                    encoded.append(ord('r'))
                    s = tstr.encode('utf-8')
                    encoded.extend(struct.pack('>H', len(s)))
                    encoded.extend(s)
            else:
                encoded.append(ord('r'))
                s = tstr.encode('utf-8')
                encoded.extend(struct.pack('>H', len(s)))
                encoded.extend(s)

    return bytes(encoded)


def encode_python_files(sources, mapping):
    """编码所有 Python 文件，返回 {fname: encoded_bytes}"""
    encoded_streams = {}
    for fname, source in sources.items():
        lines = get_line_groups(source)
        encoded_streams[fname] = encode_lines(lines, mapping)
    return encoded_streams


def encode_dict(mapping):
    """将映射字典编码为紧凑二进制"""
    b = bytearray()

    for key in ['kw', 'bi', 'op', 'nu']:
        items = mapping[key]
        b.append(ord(key[0]))
        b.extend(struct.pack('>H', len(items)))
        for s in items:
            s_bytes = s.encode('utf-8')
            b.extend(struct.pack('>H', len(s_bytes)))
            b.extend(s_bytes)

    id_origs = [orig for short, orig in mapping['id'].items()]
    b.append(ord('i'))
    b.extend(struct.pack('>H', len(id_origs)))
    for orig in id_origs:
        orig_b = orig.encode('utf-8')
        b.extend(struct.pack('>H', len(orig_b)))
        b.extend(orig_b)

    st_origs = [orig for short, orig in mapping['st'].items()]
    b.append(ord('s'))
    b.extend(struct.pack('>H', len(st_origs)))
    for orig in st_origs:
        orig_b = orig.encode('utf-8')
        b.extend(struct.pack('>H', len(orig_b)))
        b.extend(orig_b)

    return bytes(b)


# ============================================================
#  通用代码编码器（类 C 语法 Token 级压缩）
# ============================================================

# Token 类型常量（复用 Python tokenize 模块值以保持兼容）
import tokenize as _tokenize_module

# 通用 tokenizer 正则表达式（优先级从高到低）
_GENERIC_TOKEN_RE = re.compile(r'''
    (?P<STRING>        (?:"(?:[^"\\]|\\.)*"          |
                           \'(?:[^\'\\]|\\.)*\'          |
                           \`(?:[^`\\]|\\.)*\` )        ) |
    (?P<TEMPLATE_HEAD> \`(?:[^`\\]|\\.)*\$\{           ) |
    (?P<LINECOMMENT>   //[^\n]*                         ) |
    (?P<BLOCKCOMMENT>  /\*[\s\S]*?\*/                   ) |
    (?P<NUMBER>        (?:0[xX][0-9a-fA-F]+             |
                           0[bB][01]+                    |
                           0[oO][0-7]+                   |
                           (?:\d+\.\d*|\.\d+|\d+)
                           (?:[eE][+-]?\d+)?[fFlLiJuUzZ]* ) ) |
    (?P<OPERATOR>      (?:\+\+|--|::|<<=?|>>>=?|>>=?|===|!==|
                           =>|\.\.\.|&&|\|\||[-+*/%&|^!=<>]=?|
                           [?~,.:;[\](){}@#]           ) ) |
    (?P<IDENTIFIER>    [a-zA-Z_][a-zA-Z0-9_]*            ) |
    (?P<WHITESPACE>    [ \t\r\f\v]+                      ) |
    (?P<NEWLINE>       \n                                 ) |
    (?P<OTHER>         .                                  )
''', re.VERBOSE)

# 需要空格分隔的前后 token 规则
_NEED_SPACE_BEFORE = {
    _tokenize_module.NAME: True,
    _tokenize_module.NUMBER: True,
    _tokenize_module.STRING: True,
}
_NEED_SPACE_AFTER = {
    _tokenize_module.NAME: True,
    _tokenize_module.NUMBER: True,
}

# 关键字后紧跟左括号不需要空格
_PARENS = {'(', '[', '{'}
# 左括号前通常不需要空格
_NO_SPACE_BEFORE_PAREN = True


def _protect_strings(source):
    """
    保护源码中的字符串和正则字面量，避免在移除注释时被误伤。
    返回 (cleaned_source, string_map)。
    """
    string_map = {}
    counter = [0]

    def replace_string(m):
        key = f"\x00STR{counter[0]}\x00"
        string_map[key] = m.group()
        counter[0] += 1
        return key

    # 保护正则表达式字面量（/pattern/flags）— 在保护字符串之后处理
    # 先保护字符串（单/双引号）
    protected = re.sub(
        r'(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',
        replace_string,
        source
    )
    # 先保护模板字符串（完整）
    protected = re.sub(
        r'\`(?:[^`\\]|\\.)*\`',
        replace_string,
        protected
    )

    return protected, string_map


def _restore_strings(source, string_map):
    """恢复被保护的字符串（支持嵌套占位符，多次恢复）。"""
    changed = True
    while changed:
        changed = False
        for key, value in string_map.items():
            if key in source:
                source = source.replace(key, value)
                changed = True
    return source


def tokenize_generic_code(source, keywords_set, builtins_set):
    """
    通用类 C 代码 Tokenizer。

    先在整个源码级别移除所有注释（处理跨行块注释，
    但先保护字符串避免误伤），再逐行正则解析。
    输出格式与 Python tokenizer 一致：
    返回 [(indent, [(ttype, tstr), ...]), ...]
    """
    # 第一步：保护字符串（避免注释移除时破坏 URL 等）
    protected, string_map = _protect_strings(source)

    # 第二步：移除所有注释
    cleaned = re.sub(r'/\*[\s\S]*?\*/', '', protected)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)

    # 第三步：恢复字符串
    cleaned = _restore_strings(cleaned, string_map)

    lines = []
    for line in cleaned.split('\n'):
        # 计算缩进
        stripped = line.lstrip(' \t')
        indent = len(line) - len(stripped)
        if not stripped:
            continue  # 跳过空行

        tokens = []
        pos = 0

        while pos < len(stripped):
            m = _GENERIC_TOKEN_RE.match(stripped, pos)
            if not m:
                pos += 1
                continue

            pos = m.end()
            kind = m.lastgroup
            value = m.group()

            if kind in ('LINECOMMENT', 'BLOCKCOMMENT'):
                continue
            elif kind == 'WHITESPACE':
                continue
            elif kind == 'NEWLINE':
                continue
            elif kind == 'IDENTIFIER':
                tokens.append((_tokenize_module.NAME, value))
            elif kind == 'OPERATOR':
                tokens.append((_tokenize_module.OP, value))
            elif kind == 'STRING':
                tokens.append((_tokenize_module.STRING, value))
            elif kind == 'TEMPLATE_HEAD':
                tokens.append((_tokenize_module.STRING, value))
            elif kind == 'NUMBER':
                tokens.append((_tokenize_module.NUMBER, value))
            elif kind == 'OTHER':
                tokens.append((_tokenize_module.NAME, value))
            else:
                tokens.append((_tokenize_module.NAME, value))

        if tokens:
            lines.append((indent, tokens))

    return lines


def build_generic_mapping_dict(sources_with_lang):
    """
    构建通用代码的全局映射字典。

    sources_with_lang: {fname: (source, keywords_set, builtins_set)}
    返回与 Python 版本兼容的 mapping dict。
    """
    keywords_set = set()
    builtins_set = set()
    operators_set = set()
    numbers_set = set()
    idents_counter = collections.Counter()
    strings_counter = collections.Counter()

    for source, kw_set, bi_set in sources_with_lang.values():
        lines = tokenize_generic_code(source, kw_set, bi_set)
        for indent, tokens in lines:
            for ttype, tstr in tokens:
                if ttype == _tokenize_module.NAME:
                    if tstr in kw_set:
                        keywords_set.add(tstr)
                    elif tstr in bi_set:
                        builtins_set.add(tstr)
                    elif tstr and tstr[0].isalpha():
                        idents_counter[tstr] += 1
                    else:
                        # 其他 NAME 类型的 token（如 $）
                        pass
                elif ttype == _tokenize_module.OP:
                    operators_set.add(tstr)
                elif ttype == _tokenize_module.STRING:
                    strings_counter[tstr] += 1
                elif ttype == _tokenize_module.NUMBER:
                    numbers_set.add(tstr)

    short_names = generate_short_names(len(idents_counter))
    id_map = {}
    for i, (orig_name, _) in enumerate(idents_counter.most_common()):
        id_map[short_names[i]] = orig_name

    st_map = {}
    for i, (orig_str, count) in enumerate(strings_counter.most_common()):
        if count >= 2:
            st_map[f"s{i}"] = orig_str

    return {
        "v": "7",
        "kw": sorted(keywords_set),
        "bi": sorted(builtins_set),
        "op": sorted(operators_set),
        "nu": sorted(numbers_set, key=lambda s: len(s)),
        "id": id_map,
        "st": st_map,
    }


def encode_generic_files(sources_with_lang, mapping):
    """
    编码所有通用代码文件，返回 {fname: encoded_bytes}。
    复用 encode_lines 函数。
    """
    encoded_streams = {}
    for fname, (source, kw_set, bi_set) in sources_with_lang.items():
        lines = tokenize_generic_code(source, kw_set, bi_set)
        encoded_streams[fname] = encode_lines(lines, mapping)
    return encoded_streams


# ============================================================
#  SVG 编码器（文本变换压缩）
# ============================================================

def compress_svg(content):
    """强力 SVG 文本变换压缩"""
    # 1. 移除 base64 字体和空 defs
    content = re.sub(r'<style class="style-fonts">.*?</style>', '', content, flags=re.DOTALL)
    content = re.sub(r'<defs>\s*</defs>', '', content, flags=re.DOTALL)

    # 2. 移除空标签
    content = re.sub(r'<mask>\s*</mask>', '', content)
    content = re.sub(r'<metadata>\s*</metadata>', '', content)

    # 3. 移除注释
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

    # 4. 移除 XML 默认值
    content = content.replace(' standalone="no"', '')

    # 5. 简化文本属性
    content = content.replace(' direction="ltr"', '')
    content = content.replace(' dominant-baseline="alphabetic"', '')
    content = content.replace(' style="white-space: pre;"', '')
    content = content.replace(
        'font-family="Comic Shanns, monospace, Segoe UI Emoji"',
        'font-family="monospace"'
    )

    # 6. 安全颜色简写
    content = content.replace('#ffffff', '#fff')

    # 7. 简化数值精度（保留 1 位小数）
    def simplify_num(m):
        v = float(m.group(0))
        if v == int(v):
            return str(int(v))
        r = round(v, 1)
        return str(int(r)) if r == int(r) else str(r)

    xml_decl = re.search(r'<\?xml[^?]*\?>', content)
    if xml_decl:
        pre_xml = content[:xml_decl.end()]
        rest = content[xml_decl.end():]
        rest = re.sub(r'-?\d+\.\d+', simplify_num, rest)
        content = pre_xml + rest
    else:
        content = re.sub(r'-?\d+\.\d+', simplify_num, content)

    # 8. 移除路径上的冗余默认属性
    content = re.sub(r'\s+stroke="none"\s+stroke-width="0"', '', content)
    content = re.sub(r'fill="none"\s+(stroke="[^"]+")', r'\1', content)
    content = re.sub(r'(stroke="[^"]+")\s+fill="none"', r'\1', content)

    # 9. 手绘矩形路径 → 标准 <rect>
    def replace_rect(m):
        transform = m.group(1)
        inner = m.group(2)

        fill = re.search(r'fill="(#[^"]+)"', inner)
        fill_color = fill.group(1) if fill else '#e7f5ff'

        stroke_match = re.search(
            r'<path[^>]*fill="none"[^>]*stroke="(#[^"]+)"[^>]*stroke-width="([^"]+)"',
            inner
        )
        if not stroke_match:
            stroke_match = re.search(
                r'<path[^>]*stroke="(#[^"]+)"[^>]*stroke-width="([^"]+)"[^>]*fill="none"',
                inner
            )

        if stroke_match:
            stroke_color = stroke_match.group(1)
            stroke_width = stroke_match.group(2)
        else:
            stroke_color = '#1971c2'
            stroke_width = '2'

        rotate_m = re.search(r'rotate\(0\s+([\d.]+)\s+([\d.]+)\)', transform)
        if rotate_m:
            cx = float(rotate_m.group(1))
            cy = float(rotate_m.group(2))
            width = cx * 2
            height = cy * 2
            rx = cy * 0.5

            width = int(width) if width == int(width) else width
            height = int(height) if height == int(height) else height
            rx = int(rx) if rx == int(rx) else round(rx, 1)

            rect_tag = (
                f'<rect x="0" y="0" width="{width}" height="{height}" '
                f'rx="{rx}" ry="{rx}" fill="{fill_color}" '
                f'stroke="{stroke_color}" stroke-width="{stroke_width}"></rect>'
            )
            return f'<g transform="{transform}">{rect_tag}</g>'

        return m.group(0)

    content = re.sub(
        r'<g stroke-linecap="round" transform="([^"]+)">(.*?)</g>',
        replace_rect,
        content,
        flags=re.DOTALL
    )

    # 10. 压缩空白
    content = re.sub(r'>\s+<', '><', content)
    content = re.sub(r'  +', ' ', content)
    content = content.strip()

    return content


def encode_svg_files(sources):
    """编码所有 SVG 文件，返回 {fname: encoded_bytes}"""
    encoded_streams = {}
    for fname, content in sources.items():
        compressed = compress_svg(content)
        encoded_streams[fname] = compressed.encode('utf-8')
    return encoded_streams


# ============================================================
#  PNG 编码器（调用 util.py）
# ============================================================

def encode_png_files(sources, clusters=16):
    """
    编码所有 PNG 文件。
    sources: {fname: filepath}
    返回: ({fname: encoded_bytes}, {fname: psnr})
    """
    encoded_streams = {}
    psnr_map = {}
    for fname, fpath in sources.items():
        data, psnr = encode_png_to_stream(fpath, clusters=clusters)
        encoded_streams[fname] = data
        psnr_map[fname] = psnr
    return encoded_streams, psnr_map


# ============================================================
#  LaTeX 编码器（常量替换）
# ============================================================

def encode_latex_files(sources):
    """
    编码 LaTeX/Markdown 文件：常量替换 + 收集使用的常量。

    Args:
        sources: {fname: text_content}

    Returns:
        (encoded_streams, used_consts)
        encoded_streams: {fname: replaced_bytes}
        used_consts: {marker: (orig, count)}
    """
    encoded_streams = {}
    all_used = {}

    for fname, text in sources.items():
        replaced, used = replace_latex_constants(text)
        encoded_streams[fname] = replaced.encode('utf-8')

        for marker, info in used.items():
            if marker not in all_used:
                all_used[marker] = info
            else:
                all_used[marker] = (info[0], all_used[marker][1] + info[1])

    return encoded_streams, all_used


# ============================================================
#  Markdown 编码器（分块压缩）
# ============================================================

def encode_md_files(sources, code_mapping=None):
    """
    编码 Markdown 文件：分块压缩（代码块用代码压缩器，Math用LaTeX替换，表格特殊压缩，纯文本用模式替换）。

    Args:
        sources: {fname: text_content}
        code_mapping: 全局代码字典（用于代码块token压缩）

    Returns:
        (encoded_streams, md_const_binary)
        encoded_streams: {fname: encoded_bytes}
        md_const_binary: MD模式常量表
    """
    encoded_streams = {}
    all_md_used = {}
    all_latex_used = {}
    dyn_idx = 0

    for fname, text in sources.items():
        encoded, md_used, latex_used, next_dyn_idx = encode_md_stream(text, code_mapping=code_mapping, dyn_start_idx=dyn_idx)
        encoded_streams[fname] = encoded
        dyn_idx = next_dyn_idx

        # md_used 是 dict {marker: (orig, count)}
        if isinstance(md_used, dict):
            for marker, info in md_used.items():
                if marker not in all_md_used:
                    all_md_used[marker] = info
                else:
                    all_md_used[marker] = (info[0], all_md_used[marker][1] + info[1])
        # 收集LaTeX常量
        if isinstance(latex_used, dict):
            for marker, info in latex_used.items():
                if marker not in all_latex_used:
                    all_latex_used[marker] = info
                else:
                    all_latex_used[marker] = (info[0], all_latex_used[marker][1] + info[1])

    md_const_binary = encode_md_pattern_table(all_md_used) if all_md_used else b''
    return encoded_streams, md_const_binary, all_latex_used


# ============================================================
#  统一打包与输出
# ============================================================

def pack_all_files(py_sources, code_sources, svg_sources, png_sources, latex_sources,
                   md_sources, output_path, png_clusters=16):
    """
    统一打包格式：
    [4] dict_len | [dict_len] dict_data |
    [4] latex_const_len | [latex_const_len] latex_const_data |
    [4] md_const_len | [md_const_len] md_const_data |
    [2] num_files |
    对于每个文件:
        [2] fname_len | [fname_len] filename | [1] file_type | [4] stream_len |
    [all streams concatenated]
    [lzma compressed]

    file_type: 'p' = Python, 'c' = Code (generic), 's' = SVG, 'g' = PNG (Graphic)
               'l' = LaTeX, 'm' = Markdown (分块压缩)
    """
    # 构建联合源数据（如果有多种代码类型）
    all_code_sources = {}
    if py_sources:
        all_code_sources.update(py_sources)
    if code_sources:
        for fname, (source, kw_set, bi_set) in code_sources.items():
            all_code_sources[fname] = (source, kw_set, bi_set)

    # 如果同时有 Python 和通用代码，需要分别构建各自的字典
    # 但统一使用一个字典（Python 的 tokenizer 优先级更高）
    mapping = None
    dict_binary = b''
    py_encoded = {}
    code_encoded = {}

    if py_sources and code_sources:
        # 两者都有：使用 Python 字典 + code-only 的额外条目
        # 先用各自的方法构建字典
        py_mapping = build_mapping_dict(py_sources)
        code_mapping = build_generic_mapping_dict(code_sources)

        # 合并字典（Python 的条目在前）
        merged = {"v": "7"}
        for key in ['kw', 'bi', 'op', 'nu']:
            combined = list(dict.fromkeys(py_mapping.get(key, []) + code_mapping.get(key, [])))
            merged[key] = combined
        # id: py 在前，code 在后
        py_ids = list(py_mapping['id'].values())
        code_ids = list(code_mapping['id'].values())
        short_names = generate_short_names(len(py_ids) + len(code_ids))
        merged['id'] = {short_names[i]: name for i, name in enumerate(py_ids + code_ids)}
        # st: py 在前，code 在后
        py_st = list(py_mapping['st'].values())
        code_st = list(code_mapping['st'].values())
        merged['st'] = {f"s{i}": s for i, s in enumerate(py_st + code_st)}

        mapping = merged
        dict_binary = encode_dict(mapping)

        # 用合并字典重新编码所有文件（Python 文件不需要偏移，因为 Python 的条目在前）
        py_encoded = encode_python_files(py_sources, mapping)
        code_encoded = encode_generic_files(code_sources, mapping)

    elif py_sources:
        # 只有 Python
        mapping = build_mapping_dict(py_sources)
        dict_binary = encode_dict(mapping)
        py_encoded = encode_python_files(py_sources, mapping)

    elif code_sources:
        # 只有通用代码
        mapping = build_generic_mapping_dict(code_sources)
        dict_binary = encode_dict(mapping)
        code_encoded = encode_generic_files(code_sources, mapping)

    # SVG 编码
    svg_encoded = encode_svg_files(svg_sources)

    # PNG 编码
    png_encoded = {}
    png_psnr = {}
    if png_sources:
        png_encoded, png_psnr = encode_png_files(png_sources, clusters=png_clusters)

    # LaTeX 编码
    latex_encoded = {}
    latex_const_binary = b''
    if latex_sources:
        latex_encoded, latex_used = encode_latex_files(latex_sources)
        if latex_used:
            latex_const_binary = encode_latex_const_table(latex_used)

    # Markdown 编码（分块压缩）
    md_encoded = {}
    md_const_binary = b''
    md_latex_used = {}
    if md_sources:
        md_encoded, md_const_binary, md_latex_used = encode_md_files(md_sources, code_mapping=mapping)
    
    # 合并 MD 文件中的 LaTeX 常量到 latex_const_binary
    if md_latex_used:
        if latex_const_binary:
            # 解码现有的 LaTeX 常量表（{marker: orig}），转换为 {marker: (orig, 1)} 格式
            existing = decode_latex_const_table(latex_const_binary)
            merged = {marker: (orig, 1) for marker, orig in existing.items()}
            for marker, info in md_latex_used.items():
                if marker not in merged:
                    merged[marker] = info
            latex_const_binary = encode_latex_const_table(merged)
        else:
            latex_const_binary = encode_latex_const_table(md_latex_used)

    # 统一打包
    all_encoded = {**py_encoded, **code_encoded, **svg_encoded, **png_encoded, **latex_encoded, **md_encoded}
    num_files = len(all_encoded)

    merged = bytearray()
    # 1. 代码字典 (4字节长度 + 数据)
    merged.extend(struct.pack('>I', len(dict_binary)))
    merged.extend(dict_binary)
    # 2. LaTeX 常量表 (4字节长度 + 数据)
    merged.extend(struct.pack('>I', len(latex_const_binary)))
    merged.extend(latex_const_binary)
    # 3. MD 模式常量表 (4字节长度 + 数据)
    merged.extend(struct.pack('>I', len(md_const_binary)))
    merged.extend(md_const_binary)
    # 4. 文件数量
    merged.extend(struct.pack('>H', num_files))

    stream_data = bytearray()
    for fname, encoded in all_encoded.items():
        fname_bytes = fname.encode('utf-8')
        merged.extend(struct.pack('>H', len(fname_bytes)))
        merged.extend(fname_bytes)

        if fname.endswith('.py'):
            merged.append(ord('p'))
        elif Path(fname).suffix.lower() in CODE_EXTENSIONS:
            merged.append(ord('c'))
        elif fname.endswith('.svg'):
            merged.append(ord('s'))
        elif fname.lower().endswith('.png'):
            merged.append(ord('g'))
        elif fname.endswith('.tex'):
            merged.append(ord('l'))
        elif fname in md_sources:
            # .md 和 .txt 都使用分块压缩
            merged.append(ord('m'))
        else:
            merged.append(ord('?'))

        merged.extend(struct.pack('>I', len(encoded)))
        stream_data.extend(encoded)

    merged.extend(stream_data)
    compressed = lzma.compress(bytes(merged), preset=9)

    with open(output_path, 'wb') as f:
        f.write(compressed)

    return compressed, png_psnr


def parse_size(size_str):
    """解析大小字符串，如 '30kb', '10mb', '5000' -> bytes"""
    size_str = size_str.strip().lower()
    multipliers = {'b': 1, 'kb': 1024, 'mb': 1024**2, 'gb': 1024**3}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            return int(float(size_str[:-len(suffix)]) * mult)
    return int(size_str)


def format_size(n):
    """格式化字节数为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def group_by_type(all_files):
    """按类型分组文件，返回各类型字典"""
    py = {k: v for k, v in all_files.items() if k.endswith('.py')}
    svg = {k: v for k, v in all_files.items() if k.endswith('.svg')}
    png = {k: v for k, v in all_files.items() if k.lower().endswith('.png')}
    latex = {k: v for k, v in all_files.items() if k.endswith('.tex')}
    md = {k: v for k, v in all_files.items() if k.endswith('.md') or k.endswith('.txt')}
    code = {}
    for fn, content in all_files.items():
        ext = Path(fn).suffix.lower()
        if ext in CODE_EXTENSIONS:
            code[fn] = (content, LANGUAGE_MAP[ext], BUILTINS_MAP.get(ext, set()))
    return py, code, svg, png, latex, md


def filter_by_filetypes(all_files, filetypes_set):
    """根据文件类型集合过滤文件
    filetypes_set 中的元素以 '.' 开头，如 {'.md', '.py'}
    如果包含 'all' 或 '.all'，则不过滤
    """
    if not filetypes_set:
        return all_files
    # 处理 'all' 或 '.all'
    if 'all' in filetypes_set or '.all' in filetypes_set:
        return all_files
    filtered = {}
    for fn, content in all_files.items():
        ext = Path(fn).suffix.lower()
        if ext in filetypes_set or ext.lstrip('.') in filetypes_set:
            filtered[fn] = content
    return filtered


def get_all_extensions(all_files):
    """获取所有文件类型列表"""
    exts = set()
    for fn in all_files:
        ext = Path(fn).suffix.lower()
        if ext:
            exts.add(ext)
    return sorted(exts)


def interactive_select_filetypes(all_files):
    """交互式询问用户选择要压缩的文件类型"""
    exts = get_all_extensions(all_files)
    print(f"\n{'='*60}")
    print("  文件类型选择")
    print(f"{'='*60}")
    print(f"\n  发现 {len(all_files)} 个文件，共 {len(exts)} 种类型:")
    for i, ext in enumerate(exts, 1):
        count = sum(1 for fn in all_files if Path(fn).suffix.lower() == ext)
        print(f"    {i}. {ext}  ({count} 个文件)")
    print(f"\n  操作说明:")
    print(f"    - 输入 'all' 压缩所有文件类型")
    print(f"    - 输入类型列表，如: py,js,md (用逗号分隔)")
    print(f"    - 可带或不带点号，如 .py 和 py 均可")
    print(f"{'='*60}")
    while True:
        choice = input("\n  请选择要压缩的文件类型: ").strip()
        if not choice:
            continue
        if choice.lower() == 'all':
            return set(exts)
        selected = set()
        valid = True
        for ft in choice.split(','):
            ft = ft.strip().lower()
            if not ft.startswith('.'):
                ft = '.' + ft
            if ft in exts:
                selected.add(ft)
            else:
                available = ', '.join(exts)
                print(f"  错误: 不支持的类型 '{ft}'。可用类型: {available}")
                valid = False
                break
        if valid and selected:
            return selected


def get_compress_suffix(needs):
    """根据压缩类型列表生成压缩文件后缀
    
    Args:
        needs: 文件类型列表，如 ['py', 'md', 'svg']
    
    Returns:
        后缀字符串，如 '_py', '_py_md', '_all'
    """
    if not needs or 'all' in needs:
        return '_all'
    # 排序保证一致性
    sorted_needs = sorted(set(needs))
    return '_' + '_'.join(sorted_needs)


def get_orig_size(all_files):
    """计算原始文件总大小"""
    total = 0
    for fn, v in all_files.items():
        if isinstance(v, str) and not v.startswith('/'):
            total += len(v.encode('utf-8'))
        elif isinstance(v, str):
            total += Path(v).stat().st_size
        else:
            total += len(str(v).encode('utf-8'))
    return total


def compress_subset(py, code, svg, png, latex, md, output_path, clusters=16):
    """压缩一组文件，返回压缩后字节和png_psnr"""
    return pack_all_files(py, code, svg, png, latex, md, output_path, png_clusters=clusters)


def main():
    parser = argparse.ArgumentParser(
        description='Encoder V7 - 统一压缩工具 (PY/JS/C++/JAVA/SVG/PNG/MD/TXT)'
    )
    parser.add_argument('inputs', nargs='*', help='输入文件路径')
    parser.add_argument('-d', '--directory', metavar='DIR',
                        help='指定目录，递归收集文件')
    parser.add_argument('-o', '--output', metavar='DIR',
                        help='指定输出目录（默认: ./compress/<输入目录名>）')
    parser.add_argument('--clusters', '-k', type=int, default=16,
                        help='PNG 聚类簇数 (默认: 16)')
    parser.add_argument('--info', action='store_true',
                        help='显示 PNG 文件信息')
    parser.add_argument('--maxsize', metavar='SIZE',
                        help='单个压缩文件大小上限，如 30kb, 5mb (超出10%容忍)')
    parser.add_argument('--filetypes', metavar='TYPES',
                        help='指定压缩的文件类型，逗号分隔，如 py,js,md (不指定则交互询问)')

    args = parser.parse_args()

    if args.info:
        if args.inputs:
            from PIL import Image
            import numpy as np
            for f in args.inputs:
                p = Path(f)
                if p.suffix.lower() == '.png':
                    img = Image.open(p).convert('RGB')
                    arr = np.array(img)
                    colors = len(np.unique(arr.reshape(-1, 3), axis=0))
                    print(f"{p.name}: {arr.shape[1]}x{arr.shape[0]}, {colors} unique colors")
                else:
                    print(f"{p.name}: {p.stat().st_size} bytes")
        return

    # ============================================================
    #  1. 收集源文件
    # ============================================================
    supported_exts = ('.py', '.svg', '.png', '.tex', '.md', '.txt') + tuple(CODE_EXTENSIONS)

    if args.directory:
        input_path = Path(args.directory)
        if not input_path.is_dir():
            print(f"Error: {args.directory} 不是有效目录")
            sys.exit(1)

        all_files = {}
        for p in sorted(input_path.rglob('*')):
            if p.is_file() and p.suffix.lower() in supported_exts:
                if p.suffix.lower() == '.png':
                    all_files[p.name] = str(p)
                else:
                    all_files[p.name] = p.read_text(encoding='utf-8')

        if not all_files:
            print(f"Error: 在 {args.directory} 中未找到支持的文件")
            sys.exit(1)

        base_name = input_path.name
    else:
        if not args.inputs:
            parser.print_help()
            sys.exit(1)

        all_files = {}
        for fpath in args.inputs:
            p = Path(fpath)
            if not p.exists() or not p.is_file():
                print(f"Warning: 跳过无效文件 {fpath}")
                continue
            if p.suffix.lower() not in supported_exts:
                print(f"Warning: 跳过不支持类型 {fpath}")
                continue
            if p.suffix.lower() == '.png':
                all_files[p.name] = str(p)
            else:
                all_files[p.name] = p.read_text(encoding='utf-8')

        if not all_files:
            print("Error: 没有有效的输入文件")
            sys.exit(1)

        base_name = Path(args.inputs[0]).stem

    # ============================================================
    #  2. 处理 --filetypes 参数（必需，不指定则交互询问）
    # ============================================================
    if args.filetypes:
        selected_types = set()
        for ft in args.filetypes.split(','):
            ft = ft.strip().lower()
            if not ft.startswith('.'):
                ft = '.' + ft
            selected_types.add(ft)
        needs = {t.lstrip('.') for t in selected_types}
        # 过滤
        before = len(all_files)
        all_files = filter_by_filetypes(all_files, selected_types)
        after = len(all_files)
        print(f"  文件类型过滤: {before} -> {after} 个文件")
        if not all_files:
            available = ', '.join(get_all_extensions({k: v for k, v in {
                fn: c for fn, c in [(fn, all_files.get(fn)) for fn in all_files]
            }.items()}))
            print(f"Error: 过滤后无文件。可用类型: {available}")
            sys.exit(1)
    else:
        # 交互式询问
        needs = interactive_select_filetypes(all_files)
        all_files = filter_by_filetypes(all_files, needs)
        print(f"  已选择类型: {', '.join(sorted(needs))}")
        print(f"  过滤后文件数: {len(all_files)}")

    # ============================================================
    #  3. 按类型分组
    # ============================================================
    py_sources, code_sources, svg_sources, png_sources, latex_sources, md_sources = group_by_type(all_files)

    # 确定输出路径
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('compress') / base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    #  4. 处理 --maxsize 参数（自动拆分压缩）
    # ============================================================
    orig_size = get_orig_size(all_files)

    if args.maxsize:
        max_bytes = parse_size(args.maxsize)
        max_with_tolerance = int(max_bytes * 1.1)  # 10% 容忍
        print(f"\n  最大压缩文件大小: {format_size(max_bytes)} (容忍上限: {format_size(max_with_tolerance)})")

        # 逐个文件尝试压缩，直到超出上限
        file_list = list(all_files.items())
        part_num = 1
        remaining_files = dict(file_list)
        total_compressed = 0

        while remaining_files:
            part_files = {}
            part_remaining = dict(remaining_files)

            # 逐个添加文件，检查大小
            for fn, content in list(remaining_files.items()):
                trial_files = dict(part_files)
                trial_files[fn] = content
                py, code, svg, png, latex, md = group_by_type(trial_files)
                suffix = get_compress_suffix(needs)
                tmp_path = output_dir / f"{base_name}_part{part_num}{suffix}.compress"
                try:
                    compressed, _ = compress_subset(py, code, svg, png, latex, md, tmp_path, args.clusters)
                except Exception as e:
                    print(f"  警告: 压缩失败 {fn}: {e}")
                    part_remaining.pop(fn, None)
                    continue

                if len(compressed) <= max_with_tolerance or not part_files:
                    # 可以容纳
                    part_files[fn] = content
                    part_remaining.pop(fn, None)
                else:
                    # 超出上限且不是第一个文件，停止添加
                    break

            if not part_files:
                # 单个文件就超出上限，只能单独压缩
                fn, content = list(part_remaining.items())[0]
                part_files[fn] = content
                part_remaining.pop(fn)

            # 压缩这一批
            py, code, svg, png, latex, md = group_by_type(part_files)
            output_path = output_dir / f"{base_name}_part{part_num}.compress"
            compressed, png_psnr = compress_subset(py, code, svg, png, latex, md, output_path, args.clusters)
            total_compressed += len(compressed)

            # 统计
            part_orig = get_orig_size(part_files)
            print(f"\n  Part {part_num}: {len(part_files)} 个文件")
            print(f"    原始: {format_size(part_orig)} -> 压缩: {format_size(len(compressed))} ({len(compressed)/part_orig*100:.1f}%)")

            remaining_files = part_remaining
            part_num += 1

        print(f"\n{'='*50}")
        print("   Compression Results (Multi-Part)")
        print(f"{'='*50}")
        print(f"  Original:   {format_size(orig_size)}")
        print(f"  Compressed: {format_size(total_compressed)} ({total_compressed/orig_size*100:.1f}%)")
        print(f"  Ratio:      {orig_size/total_compressed:.1f}x")
        print(f"  Parts:      {part_num - 1} 个压缩文件")
        print(f"  Output:     {output_dir}")
        print(f"{'='*50}")

    else:
        # 普通单文件压缩
        suffix = get_compress_suffix(needs)
        output_path = output_dir / f"{base_name}{suffix}.compress"
        compressed, png_psnr = compress_subset(
            py_sources, code_sources, svg_sources, png_sources, latex_sources, md_sources,
            output_path, args.clusters
        )

        print(f"\n{'='*50}")
        print("   Compression Results")
        print(f"{'='*50}")
        print(f"  Original:   {format_size(orig_size)}")
        print(f"  Compressed: {format_size(len(compressed))} ({len(compressed)/orig_size*100:.1f}%)")
        print(f"  Ratio:      {orig_size/len(compressed):.1f}x")
        print(f"  Output:     {output_path}")

        if py_sources:
            print(f"\n  Python: {len(py_sources)} files")
        if code_sources:
            langs = sorted(set(str(Path(k).suffix).lower() for k in code_sources))
            print(f"  Code:   {len(code_sources)} files ({', '.join(langs)})")
        if svg_sources:
            print(f"  SVG:    {len(svg_sources)} files")
        if png_sources:
            for fname, psnr in png_psnr.items():
                print(f"  PNG:    {fname} (PSNR={psnr:.1f}dB, K={args.clusters})")
        if latex_sources:
            print(f"  LaTeX:  {len(latex_sources)} files")
        if md_sources:
            print(f"  MD:     {len(md_sources)} files (block-level compression)")
        print(f"{'='*50}")


if __name__ == '__main__':
    main()

