import struct
from collections import Counter

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

