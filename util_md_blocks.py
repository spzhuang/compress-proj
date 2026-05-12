import struct
from util_base import CODE_EXTENSIONS, BUILTINS_MAP, LANGUAGE_MAP

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

