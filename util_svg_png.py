"""
SVG 和 PNG 图像处理模块
"""
import struct
import zlib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans

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
