import struct

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
