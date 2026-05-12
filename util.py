"""
兼容层 - 统一导入所有工具子模块
保持向后兼容，同时支持动态导入
"""

# 基础工具
from util_base import (
    generate_short_names, get_index_bytes, pack_index, read_index,
    encode_png_to_stream, decode_png_from_stream,
    CODE_EXTENSIONS, BUILTINS_MAP, LANGUAGE_MAP,
    detect_language,
)

# LaTeX
from util_latex import (
    replace_latex_constants, restore_latex_constants,
    encode_latex_const_table, decode_latex_const_table,
)

# MD 分块
from util_md_blocks import (
    parse_md_blocks, _count_backticks, _detect_code_lang, _lang_to_ext,
)

# MD 模式
from util_md_patterns import (
    replace_md_patterns, restore_md_patterns,
    encode_md_pattern_table, decode_md_pattern_table,
    compress_md_table, decompress_md_table,
    _extract_dynamic_patterns,
    _MD_COMMON_WORDS, _MD_PATTERNS_UNIQUE, _MD_MARKER_MAP,
)

# MD 编码/解码
from util_md_stream import (
    encode_md_stream, decode_md_stream,
    _MD_BLOCK_TYPES, _MD_BLOCK_TYPE_NAMES,
)
# encode_md_files 在 encoder.py 中定义
