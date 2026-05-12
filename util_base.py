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
