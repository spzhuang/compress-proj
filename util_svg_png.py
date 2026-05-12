import struct
import zlib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

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