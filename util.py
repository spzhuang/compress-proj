#!/usr/bin/env python3
"""
util.py - 共享工具函数

供 encoder.py 和 decoder.py 共用，包含：
  - 通用工具：短名生成、索引打包/解包
  - PNG 压缩：K-Means 量化、Color RLE V2 编解码
"""

import struct
import zlib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# ============================================================
#  通用工具
# ============================================================

def generate_short_names(count):
    """生成短标识符序列：a-z, A-Z, _, aa, ab, ..."""
    chars = [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_']
    names = chars.copy()
    for c1 in chars:
        for c2 in chars:
            names.append(c1 + c2)
    for c1 in chars:
        for c2 in chars:
            for c3 in chars:
                names.append(c1 + c2 + c3)
    return names[:count]


def get_index_bytes(dict_size):
    """根据字典大小返回索引字节数"""
    return 1 if dict_size < 128 else 2


def pack_index(value, bytes_count):
    """打包索引值为定长字节"""
    if bytes_count == 1:
        return struct.pack('>B', value)
    return struct.pack('>H', value)


def read_index(data, pos, bytes_count):
    """从字节流中读取索引值"""
    if bytes_count == 1:
        return data[pos], pos + 1
    return struct.unpack('>H', data[pos:pos+2])[0], pos + 2


# ============================================================
#  PNG 压缩工具（K-Means + Color RLE V2）
# ============================================================

def quantize_colors(image_array, k=16, random_state=42):
    """
    使用 K-Means 对图片进行颜色量化。

    Args:
        image_array: HxWx3 numpy 数组 (RGB)
        k: 聚类簇数
        random_state: 随机种子

    Returns:
        quantized: 量化后的 HxWx3 图像
        labels: HxW 的标签矩阵
        centers: Kx3 的调色板 (RGB 聚类中心)
        psnr: 峰值信噪比
    """
    h, w, c = image_array.shape
    pixels = image_array.reshape(-1, 3).astype(np.float32)

    if k < 20:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    else:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=k, random_state=random_state,
            n_init=3, max_iter=100, batch_size=2048
        )

    labels = kmeans.fit_predict(pixels)
    centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)

    quantized = centers[labels].reshape(h, w, 3)
    labels_2d = labels.reshape(h, w)

    mse = np.mean((image_array.astype(float) - quantized.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

    return quantized, labels_2d, centers, psnr


def color_rle_encode(labels_2d, centers, w, h):
    """
    Color RLE V2 编码器。

    对每种颜色，按行存储列区间（游程编码）。
    利用量化后图片的同色区域连续性，达到高压缩率。

    Format:
      [K: 1B] + [palette: K*3B]
      For each color ci:
        [row_count: 2B]
        For each row:
          [delta_y: 1-2B] + [run_count: 1B]
          For each run:
            [start_x: 2B] + [length: 1-2B]
    """
    k = len(centers)
    data = bytearray()

    # Header: K + full palette (3 bytes per color)
    data.append(k)
    data.extend(centers.tobytes())

    for ci in range(k):
        mask = (labels_2d == ci)
        rows_with_color = np.where(mask.any(axis=1))[0]
        num_rows = len(rows_with_color)
        data.extend(struct.pack('<H', num_rows))

        prev_y = -1
        for y in rows_with_color:
            y = int(y)
            # Delta-Y encoding: small incrementals compress better
            delta_y = y - prev_y
            prev_y = y
            if delta_y < 128:
                data.append(delta_y)
            else:
                data.append(128 | (delta_y >> 8))
                data.append(delta_y & 0xFF)

            # Run-length encode columns in this row
            cols = np.where(mask[y, :])[0]
            if len(cols) == 0:
                data.append(0)
                continue

            runs = []
            start_col = prev_col = int(cols[0])
            for x in cols[1:]:
                x = int(x)
                if x == prev_col + 1:
                    prev_col = x
                else:
                    runs.append((start_col, prev_col - start_col + 1))
                    start_col = prev_col = x
            runs.append((start_col, prev_col - start_col + 1))

            data.append(len(runs))
            for s, length in runs:
                data.extend(struct.pack('<H', s))
                if length < 255:
                    data.append(length)
                else:
                    data.append(255)  # marker for 2-byte length
                    data.extend(struct.pack('<H', length))

    return bytes(data)


def color_rle_decode(rle_data, w, h):
    """
    Color RLE V2 解码器。

    Args:
        rle_data: RLE 编码的字节流
        w, h: 目标图像宽高

    Returns:
        image: HxWx3 numpy 数组 (RGB)
        palette: Kx3 numpy数组 (RGB 调色板)
    """
    idx = 0
    data = rle_data

    # Read K
    k = data[idx]
    idx += 1

    # Read full palette
    palette = np.zeros((k, 3), dtype=np.uint8)
    for i in range(k):
        palette[i] = [data[idx], data[idx + 1], data[idx + 2]]
        idx += 3

    # Initialize image
    image = np.zeros((h, w, 3), dtype=np.uint8)

    for ci in range(k):
        row_count = struct.unpack('<H', data[idx:idx + 2])[0]
        idx += 2

        color = palette[ci]
        prev_y = -1

        for _ in range(row_count):
            # Delta-Y
            delta_y = data[idx]
            idx += 1
            if delta_y & 128:
                delta_y = ((delta_y & 127) << 8) | data[idx]
                idx += 1
            y = prev_y + delta_y
            prev_y = y

            run_count = data[idx]
            idx += 1

            for _ in range(run_count):
                start_x = struct.unpack('<H', data[idx:idx + 2])[0]
                idx += 2
                length = data[idx]
                idx += 1
                if length == 255:
                    length = struct.unpack('<H', data[idx:idx + 2])[0]
                    idx += 2

                end_x = min(start_x + length, w)
                image[y, start_x:end_x] = color

    return image, palette


def encode_png_to_stream(png_path, clusters=16):
    """
    将 PNG 图片编码为 RLE 原始字节流（不压缩，交给统一 LZMA）。

    Args:
        png_path: PNG 文件路径
        clusters: K-Means 聚类簇数 (默认 16)

    Returns:
        result: bytes, 格式为 [4B width][4B height][RLE raw data]
        psnr: float, 峰值信噪比
    """
    img = Image.open(png_path).convert('RGB')
    arr = np.array(img)
    h, w, _ = arr.shape

    _, labels_2d, centers, psnr = quantize_colors(arr, k=clusters)
    rle_data = color_rle_encode(labels_2d, centers, w, h)

    # 不再使用 zlib —— RLE 原始数据交给统一的 LZMA 压缩
    result = struct.pack('<II', w, h) + rle_data
    return result, psnr


def decode_png_from_stream(png_stream):
    """
    从 RLE 字节流解码为 PIL Image。

    Args:
        png_stream: bytes, 格式为 [4B width][4B height][RLE raw data]

    Returns:
        PIL Image 对象 (RGB 模式)
    """
    w = struct.unpack('<I', png_stream[:4])[0]
    h = struct.unpack('<I', png_stream[4:8])[0]
    rle_data = png_stream[8:]

    # 不再使用 zlib —— 数据由统一 LZMA 解压后传入
    image, _ = color_rle_decode(rle_data, w, h)

    return Image.fromarray(image)


# ============================================================
#  通用代码压缩 - 语言关键字与文件扩展名映射
# ============================================================

# 各语言关键字集合
KEYWORDS_JS = {
    'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
    'default', 'delete', 'do', 'else', 'enum', 'export', 'extends', 'false',
    'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new',
    'null', 'return', 'super', 'switch', 'this', 'throw', 'true', 'try',
    'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static',
    'await', 'async', 'of', 'as', 'from', 'implements', 'interface',
    'package', 'private', 'protected', 'public', 'abstract', 'readonly',
    'declare', 'namespace', 'type', 'module', 'require', 'module',
    'get', 'set', 'constructor',
}

KEYWORDS_TS = KEYWORDS_JS | {
    'interface', 'type', 'namespace', 'declare', 'abstract', 'readonly',
    'implements', 'module', 'enum', 'out', 'override', 'satisfies',
    'infer', 'is', 'keyof', 'unique', 'using',
}

KEYWORDS_C = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'inline', 'int', 'long', 'register', 'restrict', 'return', 'short',
    'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
    'unsigned', 'void', 'volatile', 'while', '_Alignas', '_Alignof',
    '_Atomic', '_Bool', '_Complex', '_Generic', '_Imaginary',
    '_Noreturn', '_Static_assert', '_Thread_local',
    'bool', 'true', 'false', 'nullptr', 'typeof', 'alignas', 'alignof',
    'static_assert', 'thread_local',
}

KEYWORDS_CPP = KEYWORDS_C | {
    'class', 'public', 'private', 'protected', 'virtual', 'override', 'final',
    'template', 'typename', 'namespace', 'using', 'new', 'delete', 'explicit',
    'mutable', 'constexpr', 'consteval', 'constinit', 'co_await', 'co_return',
    'co_yield', 'decltype', 'noexcept', 'requires', 'concept', 'export',
    'friend', 'operator', 'private', 'protected', 'public', 'this', 'throw',
    'try', 'catch',
}

KEYWORDS_JAVA = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
    'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
    'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
    'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
    'package', 'private', 'protected', 'public', 'return', 'short', 'static',
    'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
    'transient', 'try', 'void', 'volatile', 'while', 'record', 'sealed',
    'permits', 'var', 'yield', 'non-sealed',
}

KEYWORDS_CS = {
    'abstract', 'as', 'base', 'bool', 'break', 'byte', 'case', 'catch', 'char',
    'checked', 'class', 'const', 'continue', 'decimal', 'default', 'delegate',
    'do', 'double', 'else', 'enum', 'event', 'explicit', 'extern', 'false',
    'finally', 'fixed', 'float', 'for', 'foreach', 'goto', 'if', 'implicit',
    'in', 'int', 'interface', 'internal', 'is', 'lock', 'long', 'namespace',
    'new', 'null', 'object', 'operator', 'out', 'override', 'params',
    'private', 'protected', 'public', 'readonly', 'ref', 'return', 'sbyte',
    'sealed', 'short', 'sizeof', 'stackalloc', 'static', 'string', 'struct',
    'switch', 'this', 'throw', 'true', 'try', 'typeof', 'uint', 'ulong',
    'unchecked', 'unsafe', 'ushort', 'using', 'virtual', 'void', 'volatile',
    'while', 'add', 'alias', 'ascending', 'descending', 'dynamic', 'from',
    'get', 'global', 'group', 'into', 'join', 'let', 'orderby', 'partial',
    'remove', 'select', 'set', 'value', 'var', 'where', 'yield', 'when',
    'nameof', 'init', 'record', 'required', 'scoped', 'file',
}

KEYWORDS_GO = {
    'break', 'case', 'chan', 'const', 'continue', 'default', 'defer', 'else',
    'fallthrough', 'for', 'func', 'go', 'goto', 'if', 'import', 'interface',
    'map', 'package', 'range', 'return', 'select', 'struct', 'switch', 'type',
    'var',
}

KEYWORDS_RUST = {
    'as', 'async', 'await', 'break', 'const', 'continue', 'crate', 'dyn',
    'else', 'enum', 'extern', 'false', 'fn', 'for', 'if', 'impl', 'in',
    'let', 'loop', 'match', 'mod', 'move', 'mut', 'pub', 'ref', 'return',
    'self', 'Self', 'static', 'struct', 'super', 'trait', 'true', 'type',
    'unsafe', 'use', 'where', 'while', 'abstract', 'become', 'box', 'do',
    'final', 'macro', 'override', 'priv', 'typeof', 'unsized', 'virtual',
    'yield', 'try', 'gen', 'raw', 'union',
}

KEYWORDS_SWIFT = {
    'associatedtype', 'class', 'deinit', 'enum', 'extension', 'func', 'import',
    'init', 'inout', 'internal', 'let', 'operator', 'private', 'protocol',
    'public', 'static', 'struct', 'subscript', 'typealias', 'var', 'break',
    'case', 'continue', 'default', 'defer', 'do', 'else', 'fallthrough',
    'for', 'guard', 'if', 'in', 'repeat', 'return', 'switch', 'where', 'while',
    'as', 'Any', 'catch', 'false', 'is', 'nil', 'rethrows', 'super', 'self',
    'Self', 'throw', 'throws', 'true', 'try', 'discardableResult', 'dynamic',
    'convenience', 'override', 'open', 'final', 'lazy', 'mutating',
    'nonmutating', 'prefix', 'postfix', 'indirect', 'optional', 'required',
    'weak', 'unowned', 'willSet', 'didSet', 'Protocol', 'Type',
}

KEYWORDS_KOTLIN = {
    'as', 'as?', 'break', 'by', 'catch', 'class', 'companion', 'const',
    'constructor', 'continue', 'crossinline', 'data', 'delegate', 'do',
    'dynamic', 'else', 'enum', 'external', 'false', 'field', 'file', 'finally',
    'for', 'fun', 'get', 'if', 'import', 'in', 'infix', 'init', 'inline',
    'inner', 'interface', 'internal', 'is', 'it', 'lateinit', 'noinline',
    'null', 'object', 'open', 'operator', 'out', 'override', 'package', 'param',
    'private', 'property', 'protected', 'public', 'receiver', 'reified',
    'return', 'sealed', 'set', 'setparam', 'super', 'suspend', 'tailrec',
    'this', 'throw', 'true', 'try', 'typealias', 'typeof', 'val', 'var',
    'vararg', 'when', 'where', 'while', 'value', 'annotation', 'actual',
    'expect',
}

KEYWORDS_PHP = {
    '__halt_compiler', 'abstract', 'and', 'array', 'as', 'break', 'callable',
    'case', 'catch', 'class', 'clone', 'const', 'continue', 'declare',
    'default', 'die', 'do', 'echo', 'else', 'elseif', 'empty', 'enddeclare',
    'endfor', 'endforeach', 'endif', 'endswitch', 'endwhile', 'eval', 'exit',
    'extends', 'false', 'final', 'finally', 'fn', 'for', 'foreach', 'function',
    'global', 'goto', 'if', 'implements', 'include', 'include_once',
    'instanceof', 'insteadof', 'interface', 'isset', 'list', 'match',
    'namespace', 'new', 'or', 'print', 'private', 'protected', 'public',
    'readonly', 'require', 'require_once', 'return', 'static', 'switch',
    'throw', 'trait', 'true', 'try', 'unset', 'use', 'var', 'while', 'xor',
    'yield', 'from',
}

KEYWORDS_RUBY = {
    '__ENCODING__', '__LINE__', '__FILE__', 'BEGIN', 'END', 'alias', 'and',
    'begin', 'break', 'case', 'class', 'def', 'defined?', 'do', 'else',
    'elsif', 'end', 'ensure', 'false', 'for', 'if', 'in', 'module', 'next',
    'nil', 'not', 'or', 'redo', 'rescue', 'retry', 'return', 'self', 'super',
    'then', 'true', 'undef', 'unless', 'until', 'when', 'while', 'yield',
    'begin', 'rescue', 'ensure', 'retry', 'untrace_var', 'trace_var',
}

KEYWORDS_SQL = {
    'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP',
    'ALTER', 'TABLE', 'INDEX', 'VIEW', 'JOIN', 'LEFT', 'RIGHT', 'INNER',
    'OUTER', 'ON', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET',
    'UNION', 'ALL', 'DISTINCT', 'AS', 'AND', 'OR', 'NOT', 'NULL', 'IS', 'IN',
    'BETWEEN', 'LIKE', 'EXISTS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'IF',
    'CAST', 'CONVERT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'INTO', 'VALUES',
    'SET', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'DEFAULT',
    'AUTO_INCREMENT', 'UNIQUE', 'CHECK', 'CONSTRAINT', 'DATABASE', 'SHOW',
    'USE', 'DESCRIBE', 'EXPLAIN', 'TRANSACTION', 'COMMIT', 'ROLLBACK',
    'SAVEPOINT', 'GRANT', 'REVOKE', 'PROCEDURE', 'FUNCTION', 'TRIGGER',
    'CASCADE', 'RESTRICT', 'ASC', 'DESC', 'WITH', 'RECURSIVE', 'RETURNING',
    'WINDOW', 'PARTITION', 'ROW_NUMBER', 'RANK', 'DENSE_RANK',
}

KEYWORDS_BASH = {
    'if', 'then', 'else', 'elif', 'fi', 'case', 'esac', 'for', 'select',
    'while', 'until', 'do', 'done', 'in', 'function', 'time', 'coproc',
    'declare', 'typeset', 'local', 'export', 'readonly', 'unset', 'shift',
    'exit', 'return', 'break', 'continue', 'trap', 'wait', 'eval', 'exec',
    'source', 'alias', 'unalias', 'set', 'shopt', 'getopts', 'hash', 'pwd',
    'cd', 'pushd', 'popd', 'dirs', 'echo', 'printf', 'read', 'test', '[',
    '[[', 'true', 'false',
}

# 文件扩展名 -> 关键字集合映射
LANGUAGE_MAP = {
    '.js':   KEYWORDS_JS,
    '.jsx':  KEYWORDS_JS,
    '.mjs':  KEYWORDS_JS,
    '.cjs':  KEYWORDS_JS,
    '.ts':   KEYWORDS_TS,
    '.tsx':  KEYWORDS_TS,
    '.mts':  KEYWORDS_TS,
    '.cts':  KEYWORDS_TS,
    '.c':    KEYWORDS_C,
    '.h':    KEYWORDS_C,
    '.cpp':  KEYWORDS_CPP,
    '.cc':   KEYWORDS_CPP,
    '.cxx':  KEYWORDS_CPP,
    '.hpp':  KEYWORDS_CPP,
    '.hh':   KEYWORDS_CPP,
    '.java': KEYWORDS_JAVA,
    '.cs':   KEYWORDS_CS,
    '.go':   KEYWORDS_GO,
    '.rs':   KEYWORDS_RUST,
    '.swift':KEYWORDS_SWIFT,
    '.kt':   KEYWORDS_KOTLIN,
    '.kts':  KEYWORDS_KOTLIN,
    '.php':  KEYWORDS_PHP,
    '.rb':   KEYWORDS_RUBY,
    '.rake': KEYWORDS_RUBY,
    '.sql':  KEYWORDS_SQL,
    '.sh':   KEYWORDS_BASH,
    '.bash': KEYWORDS_BASH,
    '.zsh':  KEYWORDS_BASH,
}

# 通用代码文件扩展名集合
CODE_EXTENSIONS = set(LANGUAGE_MAP.keys())

# JS/TS 内置对象（用于 builtins 分类）
BUILTINS_JS = {
    'console', 'document', 'window', 'navigator', 'location', 'history',
    'localStorage', 'sessionStorage', 'fetch', 'setTimeout', 'setInterval',
    'clearTimeout', 'clearInterval', 'JSON', 'Math', 'Date', 'RegExp',
    'Array', 'Object', 'String', 'Number', 'Boolean', 'Function', 'Symbol',
    'Map', 'Set', 'WeakMap', 'WeakSet', 'Promise', 'Proxy', 'Reflect',
    'Error', 'TypeError', 'ReferenceError', 'SyntaxError', 'RangeError',
    'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURI',
    'decodeURI', 'encodeURIComponent', 'decodeURIComponent', 'eval',
    'require', 'exports', 'module', 'global', 'process', 'Buffer',
    '__dirname', '__filename', 'arguments', 'undefined', 'NaN', 'Infinity',
    'addEventListener', 'removeEventListener', 'dispatchEvent',
    'requestAnimationFrame', 'cancelAnimationFrame',
}

# 语言 -> 内置对象映射
BUILTINS_MAP = {
    '.js': BUILTINS_JS,
    '.jsx': BUILTINS_JS,
    '.mjs': BUILTINS_JS,
    '.cjs': BUILTINS_JS,
    '.ts': BUILTINS_JS,
    '.tsx': BUILTINS_JS,
    '.mts': BUILTINS_JS,
    '.cts': BUILTINS_JS,
}


def detect_language(fname):
    """根据文件名检测编程语言，返回 (ext, keywords, builtins)"""
    ext = Path(fname).suffix.lower()
    if ext in LANGUAGE_MAP:
        keywords = LANGUAGE_MAP[ext]
        builtins = BUILTINS_MAP.get(ext, set())
        return ext, keywords, builtins
    return None, None, None


# ============================================================
#  LaTeX 常量替换压缩
# ============================================================

# 预定义 LaTeX 高频常量表: (原始文本, 紧凑标记)
# 标记格式: \x00 + 可打印字符 (\x00 在正常文本中几乎不出现)
# 按长度降序排列(长的先匹配, 避免部分匹配问题)
LATEX_CONSTANTS = [
    # 积分/求和 (6)
    (r'\int_{-\infty}^{+\infty}', '\x00A'),
    (r'\sum_{n=-\infty}^{+\infty}', '\x00H'),
    (r'\sum_{k=-\infty}^{+\infty}', '\x00I'),
    (r'\sum_{n=0}^{N-1}', '\x00T'),
    (r'_{-\infty}^{+\infty}', '\x00n'),
    (r'\frac{2\pi}{T}', '\x00G'),
    # 分数 (8)
    (r'\frac{1}{2\pi}', '\x00B'),
    (r'\frac{1}{T}', '\x00C'),
    (r'\frac{1}{2j}', '\x00D'),
    (r'\frac{w}{2\pi}', '\x00E'),
    (r'\frac{1}{T_{s}}', '\x00S'),
    (r'\frac{dx(t)}{dt}', '\x00F'),
    (r'\frac{w_{c}T}{\pi}', '\x00U'),
    (r'\frac{1}{4\pi^{2}}', '\x00Z'),
    # 指数/函数 (8)
    (r'e^{-jwt}', '\x00J'),
    (r'e^{jwt}', '\x00K'),
    (r'e^{-at}', '\x00Y'),
    (r'e^{jkw_{s}t}', '\x00a'),
    (r'\delta(t-nT)', '\x00L'),
    (r'\delta(w-kw_{s})', '\x00X'),
    (r'F^{-1}(', '\x00h'),
    (r'F(x(t))', '\x00i'),
    # 变换/箭头 (4)
    (r'\overset{F}{\longrightarrow}', '\x00M'),
    (r'\overset{F^{-1}}{\longrightarrow}', '\x00N'),
    (r'\overline{X(jw)}', '\x00O'),
    (r'\overline{x(t)}', '\x00P'),
    # 变量/符号 (11)
    (r'x(t)', '\x00b'),
    (r'X(jw)', '\x00c'),
    (r'h(t)', '\x00d'),
    (r'P(t)', '\x00e'),
    (r'x(nT)', '\x00f'),
    (r'x(\tau)', '\x00l'),
    (r'2\pi', '\x00m'),
    (r'w_{s}', '\x00V'),
    (r'X_{p}', '\x00W'),
    (r'w_{c}', '\x00o'),
    (r'w_{M}', '\x00p'),
    # 其他 (7)
    (r'F^{-1}', '\x00g'),
    (r'jwX(jw)', '\x00j'),
    (r'x(t)P(t)', '\x00k'),
    (r'kw_{s}', '\x00q'),
    (r'\begin{align}', '\x00Q'),
    (r'\end{align}', '\x00R'),
    (r'\mathbb{R}', '\x00r'),
]

# 去重并按长度降序（长的先匹配，避免部分匹配）
_seen_consts = set()
LATEX_CONSTANTS_UNIQUE = []
for orig, marker in LATEX_CONSTANTS:
    if orig not in _seen_consts:
        _seen_consts.add(orig)
        LATEX_CONSTANTS_UNIQUE.append((orig, marker))
LATEX_CONSTANTS_UNIQUE.sort(key=lambda x: -len(x[0]))


def replace_latex_constants(text):
    """
    对文本中的 LaTeX 常量进行替换。

    Args:
        text: 原始文本字符串

    Returns:
        replaced_text: 替换后的文本
        used: dict {marker: (orig, count)} 实际使用的常量
    """
    used = {}
    result = text
    for orig, marker in LATEX_CONSTANTS_UNIQUE:
        if orig in result:
            count = result.count(orig)
            if count > 0:
                result = result.replace(orig, marker)
                if marker not in used:
                    used[marker] = (orig, 0)
                used[marker] = (used[marker][0], used[marker][1] + count)
    return result, used


def restore_latex_constants(text, const_map):
    """
    从常量映射恢复原始 LaTeX 文本。

    Args:
        text: 替换后的文本
        const_map: dict {marker: original}

    Returns:
        restored_text: 恢复后的原始文本
    """
    result = text
    for marker, orig in sorted(const_map.items(), key=lambda x: -len(x[1])):
        result = result.replace(marker, orig)
    return result


def encode_latex_const_table(used_consts):
    """
    将使用的常量表编码为紧凑二进制。

    Format:
      [2] count | for each: [2]marker_len | marker | [2]orig_len | orig

    Args:
        used_consts: dict {marker: (orig, count)}

    Returns:
        bytes: 编码后的常量表
    """
    b = bytearray()
    items = list(used_consts.items())
    b.extend(struct.pack('>H', len(items)))
    for marker, (orig, _count) in items:
        mb = marker.encode('utf-8')
        b.extend(struct.pack('>H', len(mb)))
        b.extend(mb)
        ob = orig.encode('utf-8')
        b.extend(struct.pack('>H', len(ob)))
        b.extend(ob)
    return bytes(b)


def decode_latex_const_table(data):
    """
    从二进制解码常量表。

    Args:
        data: bytes, 编码后的常量表

    Returns:
        dict {marker: orig}
    """
    pos = 0
    num_consts = struct.unpack('>H', data[pos:pos+2])[0]
    pos += 2
    const_map = {}
    for _ in range(num_consts):
        mlen = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2
        marker = data[pos:pos+mlen].decode('utf-8')
        pos += mlen
        olen = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2
        orig = data[pos:pos+olen].decode('utf-8')
        pos += olen
        const_map[marker] = orig
    return const_map


# ============================================================
#  Markdown 专用压缩/解压缩
# ============================================================

# MD代码块语言到文件扩展名的映射
_MD_LANG_TO_EXT = {
    'python': '.py', 'py': '.py', 'py3': '.py',
    'javascript': '.js', 'js': '.js', 'jsx': '.jsx', 'mjs': '.mjs',
    'typescript': '.ts', 'ts': '.ts', 'tsx': '.tsx',
    'html': '.js', 'css': '.js', 'scss': '.js', 'sass': '.js',
    'c': '.c', 'cpp': '.cpp', 'cc': '.cpp', 'cxx': '.cpp', 'h': '.c', 'hpp': '.cpp',
    'java': '.java',
    'go': '.go',
    'rust': '.rs', 'rs': '.rs',
    'swift': '.swift',
    'kotlin': '.kt', 'kt': '.kt',
    'php': '.php',
    'ruby': '.rb', 'rb': '.rb',
    'sql': '.sql',
    'bash': '.sh', 'sh': '.sh', 'zsh': '.zsh', 'shell': '.sh', 'powershell': '.sh',
    'json': '.js', 'yaml': '.py', 'yml': '.py', 'xml': '.js',
    'dockerfile': '.sh', 'makefile': '.sh',
    'c++': '.cpp', 'objectivec': '.c', 'objc': '.c',
}


def _detect_code_lang(info_line):
    """从 ```language 行中提取语言标识（保留原始大小写用于显示）"""
    lang = info_line.strip()
    # 处理类似 ```python filename.py 的情况
    lang = lang.split()[0] if lang else ''
    return lang


def _lang_to_ext(lang):
    """将语言标识转换为文件扩展名（大小写不敏感）"""
    return _MD_LANG_TO_EXT.get(lang.lower(), None)


def _count_backticks(line):
    """计算行开头的连续反引号数量"""
    stripped = line.strip()
    count = 0
    for c in stripped:
        if c == '`':
            count += 1
        else:
            break
    return count, stripped[count:].strip()


def parse_md_blocks(text):
    """
    将 Markdown 文本解析为不同类型的块序列。

    返回 [(block_type, content), ...]
    block_type: 'code', 'math_block', 'math_inline', 'table', 'text'
    """
    lines = text.split('\n')
    blocks = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # 1. 代码块 ```language (至少3个反引号，匹配相同数量的结束)
        bt_count, rest = _count_backticks(line)
        if bt_count >= 3:
            fence = '`' * bt_count
            lang = rest if rest else ''
            i += 1
            code_lines = []
            while i < n:
                next_bt_count, _ = _count_backticks(lines[i])
                if next_bt_count == bt_count:
                    break
                code_lines.append(lines[i])
                i += 1
            # 跳过结束 fence
            if i < n:
                i += 1
            blocks.append(('code', '\n'.join(code_lines), lang, bt_count))
            continue

        # 2. 块级 Math $$...$$ (支持行首/行尾空格)
        stripped = line.strip()
        if stripped == '$$':
            # 保留开始 $$ 的前后缀
            idx = line.index('$$')
            start_prefix = line[:idx]
            start_suffix = line[idx+2:]
            i += 1
            math_lines = []
            while i < n and lines[i].strip() != '$$':
                math_lines.append(lines[i])
                i += 1
            # 保存结束 $$ 的前后缀
            end_prefix = ''
            end_suffix = ''
            if i < n:
                end_line = lines[i]
                idx2 = end_line.index('$$')
                end_prefix = end_line[:idx2]
                end_suffix = end_line[idx2+2:]
                i += 1  # 跳过结束 $$
            blocks.append(('math_block', '\n'.join(math_lines), start_prefix, start_suffix, end_prefix, end_suffix))
            continue

        # 3. 表格 (以 | 开头的行)
        if line.strip().startswith('|') and '|' in line.strip()[1:]:
            table_lines = []
            while i < n and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            blocks.append(('table', '\n'.join(table_lines)))
            continue

        # 4. 收集连续的非特殊行作为文本块
        text_lines = []
        while i < n:
            line = lines[i]
            # 检查是否是特殊块的开始
            stripped = line.strip()
            if stripped.startswith('```'):
                break
            if stripped == '$$':
                break
            if stripped.startswith('|') and '|' in stripped[1:]:
                break
            text_lines.append(line)
            i += 1

        if text_lines:
            blocks.append(('text', '\n'.join(text_lines)))

    return blocks


# ============================================================
#  MD文本压缩 - 常见词/重复模式替换
# ============================================================

# Markdown 高频词汇和模式（用于文本块压缩）
# 使用自动分配的4字节标记: \x02\x00 + 2字节索引
# 固定长度标记确保解码时不会跨越边界重叠
_MD_COMMON_WORDS = [
    # 中文高频词组 (长度长的优先匹配，所以排前面)
    '如下所示', '需要注意的是', '在这种情况下', '综上所述',
    '构造函数', '命名空间', 'getElementsByClassName', 'getElementsByTagName',
    'addEventListener', 'previousSibling', 'removeAttribute', 'stopPropagation',
    'createAttribute', 'getElementById', 'sessionStorage', 'preventDefault',
    'getAttribute', 'setAttribute', 'removeChild', 'insertBefore',
    'createTextNode', 'cloneNode', 'childNodes', 'parentNode',
    'nextSibling', 'firstChild', 'lastChild', 'classList',
    'localStorage', 'innerHTML', 'innerText', 'className',
    'appendChild', 'createElement', 'querySelector', 'console.log',
    'getElementsByTagName',  # duplicate, will be deduped
    # 中文词
    '字符串', '返回值', '浏览器', '控制台',
    '函数', '方法', '对象', '数组', '变量', '参数', '循环',
    '条件', '事件', '节点', '元素', '属性', '标签',
    '文档', '页面', '按钮', '表单', '样式', '类名', '实例',
    '定义', '调用', '执行', '触发', '绑定', '修改',
    '获取', '设置', '删除', '添加', '创建', '使用',
    '通过', '可以', '表示', '例如', '如下',
    '因此', '所以', '因为', '如果', '否则', '或者', '并且',
    '对于', '关于', '根据', '由于', '同时', '此外', '另外',
    '最后', '首先', '然后', '接下来', '最终', '结果',
    '过程', '操作', '功能', '作用', '目的', '方式',
    '类型', '格式', '结构', '数据', '信息', '内容',
    '部分', '步骤', '阶段', '示例', '案例', '说明',
    '注释', '注意', '警告', '重要', '参考', '链接',
    '图片', '表格', '列表', '代码', '程序', '模块',
    '组件', '接口', '框架', '工具', '系统', '应用',
    '开发', '设计', '实现', '测试', '运行', '输出',
    '输入', '打印', '显示', '验证', '计算', '处理',
    '转换', '比较', '判断', '选择', '包含', '等于',
    '大于', '小于', '之间', '开始', '结束', '完成',
    '继续', '返回', '退出', '错误', '异常', '问题',
    '解决', '优化', '配置', '安装', '导入', '导出',
    '加载', '保存', '读取', '写入', '更新', '初始化',
    '继承', '封装', '多态', '抽象', '静态', '常量', '全局',
    '局部', '作用域', '闭包', '回调', '异步', '同步',
    '线程', '进程', '服务器', '客户端', '请求', '响应',
    '协议', '连接', '网络', '数据库', '查询', '索引',
    '表', '字段', '记录', '事务', '缓存', '存储',
    '文件', '目录', '路径', '权限', '安全', '加密',
    '解密', '认证', '授权', '会话',
    # JS 英文
    'prototype', 'constructor', 'length', 'push', 'pop',
    'shift', 'unshift', 'splice', 'slice', 'concat', 'join',
    'reverse', 'sort', 'indexOf', 'forEach', 'map', 'filter',
    'reduce', 'find', 'includes', 'split', 'replace', 'match',
    'search', 'trim', 'toString', 'parseInt', 'parseFloat',
    'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
    'fetch', 'then', 'resolve', 'reject',
    'require', 'module', 'exports', '__dirname', '__filename',
    'Event', 'Target', 'preventDefault', 'stopPropagation',
    'createTextNode', 'createAttribute',
    'keydown', 'keyup', 'keypress', 'click', 'mousedown',
    'mouseup', 'mousemove', 'mouseenter', 'mouseleave',
    'mouseover', 'mouseout', 'focus', 'blur', 'change',
    'input', 'select', 'submit', 'load', 'resize', 'scroll',
    'error', 'Cookie', 'Token', 'API', 'URL',
    'JSON', 'HTML', 'CSS', 'DOM', 'BOM', 'AJAX',
    'Promise', 'async', 'await', 'import', 'export', 'from',
    'default', 'class', 'extends', 'function', 'return',
    'const', 'let', 'var', 'this', 'super', 'new',
    'null', 'undefined', 'true', 'false', 'if', 'else',
    'for', 'while', 'switch', 'case', 'break', 'continue',
    'try', 'catch', 'throw', 'finally', 'yield', 'typeof',
    'instanceof', 'void', 'delete', 'in', 'of', 'with',
    'debugger', 'document', 'window', 'Math', 'Date',
    'obsidian', 'excalidraw',
    '>[!note]', '>[!tip]', '>[!warning]', '>[!important]',
    '>[!quote]', '>[!info]', '>[!success]', '>[!danger]',
    '>[!bug]', '>[!abstract]', '>[!todo]', '>[!example]',
    '>[!question]', '>[!failure]',
    # 中文文章高频词（职场/生活/情感类）
    '王者荣耀', '英雄联盟', '英雄联盟', '我的世界', '和平精英',
    '工作', '生活', '人生', '时间', '自己', '老板', '同事', '朋友',
    '公司', '工资', '辞职', '加班', '上班', '下班', '职场', '职业',
    '努力', '奋斗', '目标', '成功', '失败', '压力', '痛苦', '快乐',
    '游戏', '打游', '视频', '手机', '电脑', '网络', '世界', '问题',
    '事情', '感觉', '想法', '看法', '想法', '观点', '态度', '行为',
    '思考', '认为', '觉得', '知道', '明白', '理解', '意识', '认识',
    '经历', '经验', '教训', '启示', '感悟', '体会', '感受', '感想',
    '坚持', '放弃', '选择', '决定', '改变', '成长', '进步', '提升',
    '积累', '沉淀', '磨练', '锻炼', '培养', '发展', '提高', '增强',
    '需要', '想要', '希望', '期待', '渴望', '追求', '梦想', '理想',
    '现实', '实际', '真实', '真正', '确实', '实在', '其实', '本来',
    '可能', '也许', '大概', '应该', '必须', '一定', '肯定', '绝对',
    '不得不', '实际上', '事实上', '本质上', '根本上', '总体来说',
    '某种程度上', '某种意义上', '一定程度上', '一方面', '另一方面',
    '不仅没有', '不但没有', '不仅没有', '不但没有',
    '每个人', '所有人', '有些人', '很多人', '大多数人', '不少人',
    '任何事情', '所有事情', '有些事情', '很多事情', '一切事情',
    '这种情况', '那种情况', '各种情况', '任何情况', '所有情况',
    '这种方式', '那种方式', '各种方式', '任何方式', '所有方式',
    '这个问题', '那个问题', '各种问题', '任何问题', '所有问题',
    '没有', '不是', '不能', '不会', '不要', '不该', '不必', '不可',
    '知乎', '微博', '微信', '朋友', '家人', '父母', '孩子', '家庭',
    '社会', '环境', '时代', '行业', '领域', '市场', '商业', '经济',
    '领导', '下属', '团队', '项目', '产品', '客户', '用户', '服务',
    '价值', '价格', '成本', '利润', '收益', '回报', '代价', '风险',
    '机会', '机遇', '挑战', '困难', '障碍', '瓶颈', '困境', '窘境',
    '思维', '逻辑', '方法', '策略', '计划', '方案', '措施', '手段',
    '能力', '实力', '水平', '素质', '素养', '修养', '格局', '视野',
    '情绪', '心态', '心理', '精神', '精力', '体力', '身体', '健康',
    '关系', '联系', '沟通', '交流', '合作', '协作', '配合', '支持',
    '动力', '激情', '热情', '兴趣', '爱好', '习惯', '性格', '个性',
    '意义', '意思', '趣味', '味道', '色彩', '光芒', '温度', '厚度',
    '年轻', '年少', '青春', '岁月', '时光', '日子', '生活', '活着',
    '非常', '特别', '十分', '极其', '相当', '比较', '稍微', '有点',
    '一直', '永远', '总是', '经常', '常常', '有时', '偶尔', '很少',
    '开始', '后来', '最后', '最终', '终于', '起初', '最初', '原先',
    '越来越', '渐渐地', '慢慢地', '逐步地', '不断地', '持续地',
    '只不过', '无非是', '不过是', '仅仅是', '只不过是', '只不过是',
    '想想看', '想想看', '想想看', '想想看', '想想看',
    '第一段', '第二段', '第三段', '第四段', '第五段',
    '第一次', '第二次', '第三次', '第四次', '第五次',
    '星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期天',
    '想办法', '找方法', '找出路', '找机会', '找时间', '找借口',
    '没有办法', '没有方法', '没有出路', '没有机会', '没有时间',
    '到什么程度', '到什么程度', '到什么程度', '到什么程度',
    # URL/链接模式
    'https://www.zhihu.com/question/', 'https://www.zhihu.com/',
    '作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。',
    '在所难免', '理所当然', '无可厚非', '不言而喻', '显而易见',
    '众所周知', '不可否认', '毫无疑问', '毋庸置疑', '不言而喻',
]

# 去重并排序（长的优先）
_seen_md = set()
_MD_PATTERNS_UNIQUE = []
for w in _MD_COMMON_WORDS:
    if w not in _seen_md:
        _seen_md.add(w)
        _MD_PATTERNS_UNIQUE.append(w)
_MD_PATTERNS_UNIQUE.sort(key=lambda x: -len(x))

# 分配标记：
# - 前255个2字词用2字节标记: \x03 + 1字节索引 (索引1-255)
# - 其余(溢出的2字词和所有3字+词)用4字节标记: \x02\x00 + 2字节索引
# 中文2字词=6 bytes UTF-8，2字节标记节省4 bytes/次，4字节标记节省2 bytes/次
_md_2char_words = [w for w in _MD_PATTERNS_UNIQUE if len(w) <= 2]
_md_long_words = [w for w in _MD_PATTERNS_UNIQUE if len(w) > 2]
_MD_MARKER_MAP = {}
_2BYTE_LIMIT = 255
long_idx = 0
for i, w in enumerate(_md_2char_words):
    if i < _2BYTE_LIMIT:
        _MD_MARKER_MAP[w] = '\x03' + chr(i + 1)  # 2字节标记
    else:
        _MD_MARKER_MAP[w] = '\x02\x00' + struct.pack('>H', long_idx).decode('latin-1')  # 4字节
        long_idx += 1
for w in _md_long_words:
    _MD_MARKER_MAP[w] = '\x02\x00' + struct.pack('>H', long_idx).decode('latin-1')  # 4字节标记
    long_idx += 1


def _extract_dynamic_patterns(text, max_patterns=100):
    """
    从文本中动态提取高频中文模式（3-6字词组）。
    只提取出现3次以上且不在静态词表中的模式。
    
    Returns:
        list of (pattern, count, saving) sorted by saving desc
    """
    from collections import Counter
    counter = Counter()
    # 只扫描纯中文区域
    for n in range(3, 7):
        for i in range(len(text) - n + 1):
            seq = text[i:i+n]
            if all('\u4e00' <= c <= '\u9fff' for c in seq):
                counter[seq] += 1
    
    candidates = []
    for seq, count in counter.items():
        if count >= 3 and seq not in _MD_MARKER_MAP:
            utf8_len = len(seq.encode('utf-8'))
            marker_cost = 4  # 4字节标记
            saving = (utf8_len - marker_cost) * count - marker_cost  # 减去常量表存储成本
            if saving > 0:
                candidates.append((seq, count, saving))
    
    # 按节省大小排序，取前N个
    candidates.sort(key=lambda x: -x[2])
    return candidates[:max_patterns]


def replace_md_patterns(text, dyn_start_idx=0):
    """
    替换 Markdown 文本中的高频模式。
    先使用静态词表，再使用动态提取的高频模式。

    Args:
        text: 输入文本
        dyn_start_idx: 动态标记起始索引（多文件压缩时使用不同起始值避免冲突）

    Returns:
        (replaced_text, used_patterns, next_dyn_idx)
        replaced_text: 替换后的文本
        used_patterns: dict {marker: (orig, count)}
        next_dyn_idx: 下一个可用的动态标记索引
    """
    used = {}
    result = text
    
    # 1. 使用静态词表
    for orig in _MD_PATTERNS_UNIQUE:
        if orig in result:
            count = result.count(orig)
            if count > 0:
                marker = _MD_MARKER_MAP[orig]
                result = result.replace(orig, marker)
                if marker not in used:
                    used[marker] = (orig, 0)
                used[marker] = (used[marker][0], used[marker][1] + count)
    
    # 2. 使用动态提取的高频模式
    dynamic = _extract_dynamic_patterns(result, max_patterns=500)
    dyn_idx = dyn_start_idx
    for seq, count, saving in dynamic:
        if seq in result:
            # 分配动态标记（使用4字节标记），确保全局唯一
            marker = '\x02\x01' + struct.pack('>H', dyn_idx).decode('latin-1')
            dyn_idx += 1
            result = result.replace(seq, marker)
            if marker not in used:
                used[marker] = (seq, 0)
            used[marker] = (used[marker][0], used[marker][1] + count)
    
    return result, used, dyn_idx


def restore_md_patterns(text, pattern_map):
    """
    从模式映射恢复原始 MD 文本。
    使用安全的从左到右扫描替换，正确区分多种标记格式：
    - 2字节标记: \x03前缀
    - 4字节静态标记: \x02\x00前缀
    - 4字节动态标记: \x02\x01前缀
    """
    if not pattern_map:
        return text

    result = []
    i = 0
    n = len(text)

    while i < n:
        matched = False
        # 检查4字节动态标记 (以 \x02\x01 开头)
        if not matched and i + 3 < n and text[i] == '\x02' and text[i+1] == '\x01':
            marker4 = text[i:i+4]
            if marker4 in pattern_map:
                result.append(pattern_map[marker4])
                i += 4
                matched = True
        # 检查4字节静态标记 (以 \x02\x00 开头)
        if not matched and i + 3 < n and text[i] == '\x02' and text[i+1] == '\x00':
            marker4 = text[i:i+4]
            if marker4 in pattern_map:
                result.append(pattern_map[marker4])
                i += 4
                matched = True
        # 检查2字节标记 (以 \x03 开头)
        if not matched and i + 1 < n and text[i] == '\x03':
            marker2 = text[i:i+2]
            if marker2 in pattern_map:
                result.append(pattern_map[marker2])
                i += 2
                matched = True
        if not matched:
            result.append(text[i])
            i += 1

    return ''.join(result)


def encode_md_pattern_table(used_patterns):
    """将使用的MD模式表编码为紧凑二进制。
    
    格式:
    [2] num_items
    [1] type_flags for each item (0=2byte \x03, 1=4byte \x02\x00, 2=4byte \x02\x01)
    [N] markers (fixed length based on type)
    [2] orig_len + orig_bytes for each item
    """
    b = bytearray()
    items = list(used_patterns.items())
    b.extend(struct.pack('>H', len(items)))
    
    # Type flags (1 byte per item)
    TYPE_MAP = {'\x03': 0, '\x02\x00': 1, '\x02\x01': 2}
    for marker, _ in items:
        prefix = marker[:2] if marker[0] == '\x02' else marker[0]
        b.append(TYPE_MAP.get(prefix, 3))
    
    # Markers (fixed length based on type: 2, 4, or 4 bytes)
    for marker, _ in items:
        b.extend(marker.encode('latin-1'))
    
    # Original strings
    for _marker, (orig, _count) in items:
        ob = orig.encode('utf-8')
        b.extend(struct.pack('>H', len(ob)))
        b.extend(ob)
    
    return bytes(b)


def decode_md_pattern_table(data):
    """从二进制解码MD模式表。"""
    if not data:
        return {}
    pos = 0
    num = struct.unpack('>H', data[pos:pos+2])[0]
    pos += 2
    
    if num == 0:
        return {}
    
    TYPE_REVERSE = {0: '\x03', 1: '\x02\x00', 2: '\x02\x01'}
    types = [data[pos + i] for i in range(num)]
    pos += num
    
    # Read markers
    markers = []
    for t in types:
        if t == 0:  # 2-byte \x03 prefix
            mlen = 2
        elif t in (1, 2):  # 4-byte \x02\x00 or \x02\x01 prefix
            mlen = 4
        else:
            # Unknown type, treat as 4-byte marker
            mlen = 4
        if pos + mlen > len(data):
            break
        marker = data[pos:pos+mlen].decode('latin-1')
        pos += mlen
        markers.append(marker)
    
    # Read originals
    pattern_map = {}
    for i in range(min(num, len(markers))):
        if pos + 2 > len(data):
            break
        olen = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2
        if pos + olen > len(data):
            break
        orig = data[pos:pos+olen].decode('utf-8')
        pos += olen
        pattern_map[markers[i]] = orig
    
    return pattern_map


# ============================================================
#  MD表格压缩
# ============================================================

def compress_md_table(table_text):
    """
    压缩 Markdown 表格：记录对齐行的模式（l/c/r/默认）和原始 - 数量。
    格式: |n:l:n:c...| 其中 n 是原始 - 的数量，l/c/r 是对齐方式。
    """
    lines = table_text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        # 检测对齐行（只包含 |, -, :, 空格）
        if stripped.startswith('|') and all(c in '|-: ' for c in stripped):
            cells = stripped.split('|')[1:-1]  # 去掉首尾空
            compressed_cells = []
            for cell in cells:
                orig = cell.strip()
                dash_count = len(orig.replace(':', '').replace(' ', ''))
                if orig.startswith(':') and orig.endswith(':'):
                    align = 'c'
                elif orig.startswith(':'):
                    align = 'l'
                elif orig.endswith(':'):
                    align = 'r'
                else:
                    align = 'd'  # default
                compressed_cells.append(f'{dash_count}:{align}')
            result.append('|' + '|'.join(compressed_cells) + '|')
        else:
            result.append(line)
    return '\n'.join(result)


def decompress_md_table(compressed_text):
    """解压缩 Markdown 表格（恢复对齐行的 - 标记）。"""
    lines = compressed_text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        # 检测压缩的对齐行
        if stripped.startswith('|') and all(c in '|0123456789lcd: ' for c in stripped):
            cells = stripped.split('|')[1:-1]
            restored = []
            for cell in cells:
                cell = cell.strip()
                if ':' in cell:
                    count_str, align = cell.split(':', 1)
                    dashes = '-' * max(3, int(count_str))
                    if align == 'c':
                        restored.append(f' :{dashes}: ')
                    elif align == 'l':
                        restored.append(f' :{dashes} ')
                    elif align == 'r':
                        restored.append(f' {dashes}: ')
                    else:
                        restored.append(f' {dashes} ')
                else:
                    restored.append(' --- ')
            result.append('|' + '|'.join(restored) + '|')
        else:
            result.append(line)
    return '\n'.join(result)


# ============================================================
#  MD分块编码/解码
# ============================================================

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
