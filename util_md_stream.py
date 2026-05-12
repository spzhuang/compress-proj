import struct
from util_base import CODE_EXTENSIONS, BUILTINS_MAP, LANGUAGE_MAP
from util_latex import replace_latex_constants, restore_latex_constants
from util_md_blocks import parse_md_blocks, _lang_to_ext
from util_md_patterns import (
    replace_md_patterns, restore_md_patterns,
    encode_md_pattern_table, decode_md_pattern_table,
    compress_md_table, decompress_md_table,
    _MD_MARKER_MAP, _MD_PATTERNS_UNIQUE,
)

# 块类型标记 (1字节)
_MD_BLOCK_TYPES = {
    'text': 0x01,
    'code': 0x02,
    'math_block': 0x03,
    'math_inline': 0x04,
    'table': 0x05,
}

_MD_BLOCK_TYPE_NAMES = {v: k for k, v in _MD_BLOCK_TYPES.items()}


def encode_md_stream(text, code_mapping=None, dyn_start_idx=0):
    """
    将 Markdown 文本编码为分块压缩字节流。

    Args:
        text: MD原始文本
        code_mapping: 代码字典（用于代码块的token压缩）
        dyn_start_idx: 动态标记起始索引（多文件压缩时递增避免冲突）

    Returns:
        (bytes, used_patterns, latex_used, next_dyn_idx)
        bytes: 编码后的字节流
        used_patterns: dict {marker: (orig, count)} 使用的MD模式
        latex_used: dict {marker: (orig, count)} 使用的LaTeX常量
        next_dyn_idx: 下一个可用的动态标记索引
    """
    blocks = parse_md_blocks(text)

    all_used_md_patterns = {}
    all_latex_used = {}
    encoded_blocks = bytearray()
    next_dyn_idx = dyn_start_idx

    for block in blocks:
        block_type = block[0]

        if block_type == 'code':
            content = block[1]
            lang = block[2] if len(block) > 2 else ''
            bt_count = block[3] if len(block) > 3 else 3

            encoded_blocks.append(_MD_BLOCK_TYPES['code'])

            # 编码反引号数量 (1字节)
            encoded_blocks.append(bt_count)

            # 编码语言标识符
            lang_b = lang.encode('utf-8')
            encoded_blocks.extend(struct.pack('>H', len(lang_b)))
            encoded_blocks.extend(lang_b)

            # 对代码内容使用token压缩（如果语言已知）
            ext = _lang_to_ext(lang)
            if ext and ext in LANGUAGE_MAP:
                try:
                    kw_set = LANGUAGE_MAP[ext]
                    bi_set = BUILTINS_MAP.get(ext, set())
                    # 导入 encoder 的函数进行token压缩
                    from encoder import tokenize_generic_code, encode_lines
                    lines = tokenize_generic_code(content, kw_set, bi_set)
                    # 如果没有提供code_mapping，直接使用原始文本
                    if code_mapping:
                        code_bytes = encode_lines(lines, code_mapping)
                    else:
                        code_bytes = content.encode('utf-8')
                except Exception:
                    code_bytes = content.encode('utf-8')
            elif lang == 'python' or lang == 'py':
                try:
                    import tokenize
                    from encoder import get_line_groups, encode_lines
                    if code_mapping:
                        lines = get_line_groups(content)
                        code_bytes = encode_lines(lines, code_mapping)
                    else:
                        code_bytes = content.encode('utf-8')
                except Exception:
                    code_bytes = content.encode('utf-8')
            else:
                code_bytes = content.encode('utf-8')

            encoded_blocks.extend(struct.pack('>I', len(code_bytes)))
            encoded_blocks.extend(code_bytes)

        elif block_type == 'math_block':
            content = block[1]
            start_prefix = block[2] if len(block) > 2 else ''
            start_suffix = block[3] if len(block) > 3 else ''
            end_prefix = block[4] if len(block) > 4 else ''
            end_suffix = block[5] if len(block) > 5 else ''
            encoded_blocks.append(_MD_BLOCK_TYPES['math_block'])
            # 编码开始和结束的前后缀
            for s in [start_prefix, start_suffix, end_prefix, end_suffix]:
                s_b = s.encode('utf-8')
                encoded_blocks.extend(struct.pack('>H', len(s_b)))
                encoded_blocks.extend(s_b)
            # LaTeX常量替换（LaTeX常量标记以\x00开头，收集到latex_used）
            replaced, used = replace_latex_constants(content)
            for marker, info in used.items():
                if marker not in all_latex_used:
                    all_latex_used[marker] = info
                else:
                    all_latex_used[marker] = (info[0], all_latex_used[marker][1] + info[1])
            content_b = replaced.encode('utf-8')
            encoded_blocks.extend(struct.pack('>I', len(content_b)))
            encoded_blocks.extend(content_b)

        elif block_type == 'table':
            content = block[1]
            encoded_blocks.append(_MD_BLOCK_TYPES['table'])
            # 表格压缩
            compressed = compress_md_table(content)
            content_b = compressed.encode('utf-8')
            encoded_blocks.extend(struct.pack('>I', len(content_b)))
            encoded_blocks.extend(content_b)

        else:  # text
            content = block[1]
            encoded_blocks.append(_MD_BLOCK_TYPES['text'])
            # MD模式替换 + LaTeX行内公式常量替换
            replaced, used_md, next_dyn_idx = replace_md_patterns(content, next_dyn_idx)
            replaced, used_latex = replace_latex_constants(replaced)

            for marker, info in used_md.items():
                if marker not in all_used_md_patterns:
                    all_used_md_patterns[marker] = info
                else:
                    all_used_md_patterns[marker] = (info[0], all_used_md_patterns[marker][1] + info[1])
            # 收集LaTeX常量
            for marker, info in used_latex.items():
                if marker not in all_latex_used:
                    all_latex_used[marker] = info
                else:
                    all_latex_used[marker] = (info[0], all_latex_used[marker][1] + info[1])

            content_b = replaced.encode('utf-8')
            encoded_blocks.extend(struct.pack('>I', len(content_b)))
            encoded_blocks.extend(content_b)

    return bytes(encoded_blocks), all_used_md_patterns, all_latex_used, next_dyn_idx


def decode_md_stream(encoded, md_pattern_map=None, latex_const_map=None):
    """
    从分块压缩字节流解码 Markdown 文本。

    Args:
        encoded: 字节流
        md_pattern_map: MD模式映射
        latex_const_map: LaTeX常量映射

    Returns:
        恢复的MD文本
    """
    result = []
    pos = 0

    while pos < len(encoded):
        block_type = encoded[pos]
        pos += 1

        if block_type not in _MD_BLOCK_TYPE_NAMES:
            raise ValueError(f"未知的MD块类型: {block_type}")

        block_name = _MD_BLOCK_TYPE_NAMES[block_type]

        if block_name == 'code':
            bt_count = encoded[pos]
            pos += 1
            lang_len = struct.unpack('>H', encoded[pos:pos+2])[0]
            pos += 2
            lang = encoded[pos:pos+lang_len].decode('utf-8')
            pos += lang_len
            content_len = struct.unpack('>I', encoded[pos:pos+4])[0]
            pos += 4
            content = encoded[pos:pos+content_len].decode('utf-8')
            pos += content_len

            fence = '`' * bt_count
            result.append(f'{fence}{lang}\n{content}\n{fence}')

        elif block_name == 'math_block':
            parts = []
            for _ in range(4):
                plen = struct.unpack('>H', encoded[pos:pos+2])[0]
                pos += 2
                parts.append(encoded[pos:pos+plen].decode('utf-8'))
                pos += plen
            start_prefix, start_suffix, end_prefix, end_suffix = parts
            content_len = struct.unpack('>I', encoded[pos:pos+4])[0]
            pos += 4
            content = encoded[pos:pos+content_len].decode('utf-8')
            pos += content_len

            # 恢复常量（先MD模式，再LaTeX，避免LaTeX的\x00标记误匹配MD标记中的子串）
            if md_pattern_map:
                content = restore_md_patterns(content, md_pattern_map)
            if latex_const_map:
                content = restore_latex_constants(content, latex_const_map)
            result.append(f'{start_prefix}$${start_suffix}\n{content}\n{end_prefix}$${end_suffix}')

        elif block_name == 'table':
            content_len = struct.unpack('>I', encoded[pos:pos+4])[0]
            pos += 4
            content = encoded[pos:pos+content_len].decode('utf-8')
            pos += content_len

            # 恢复表格
            content = decompress_md_table(content)
            result.append(content)

        else:  # text
            content_len = struct.unpack('>I', encoded[pos:pos+4])[0]
            pos += 4
            content = encoded[pos:pos+content_len].decode('utf-8')
            pos += content_len

            # 恢复常量（先MD模式，再LaTeX，避免LaTeX的\x00标记误匹配MD标记中的子串）
            if md_pattern_map:
                content = restore_md_patterns(content, md_pattern_map)
            if latex_const_map:
                content = restore_latex_constants(content, latex_const_map)
            result.append(content)

    return '\n'.join(result)
