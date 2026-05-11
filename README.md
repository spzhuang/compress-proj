# Compress-Proj V7 - Unified Compression Tool

A unified compression tool for Python, JavaScript, C/C++, Java, SVG, PNG, and many more file formats. Uses token-level compression for code and K-Means + Color RLE V2 for images.

---

## Features

| File Type | Compression Method | File Extensions |
|-----------|-------------------|-----------------|
| Python | Token-level + Global Dictionary Mapping | `.py` |
| Generic Code | Token-level + Generic Tokenizer | `.js`, `.ts`, `.c`, `.cpp`, `.java`, `.go`, `.rs`, `.swift`, `.kt`, `.cs`, `.php`, `.rb`, `.sql`, `.sh`, ... |
| SVG | Text Transformation (cleanup + simplify) | `.svg` |
| PNG | K-Means Quantization + Color RLE V2 + LZMA | `.png` |

**Supported languages**: JavaScript, TypeScript, C, C++, Java, C#, Go, Rust, Swift, Kotlin, PHP, Ruby, SQL, Bash, and more.

---

## Architecture

```
+---------------+     +---------------+     +---------------+
|  encoder.py   |<--->|   util.py     |<--->|  decoder.py   |
|   (压缩器)    |     |  (共享工具)   |     |   (解压器)    |
+---------------+     +---------------+     +---------------+
       |                       |                     |
       v                       v                     v
  .compress file          K-Means + RLE         Restore files
```

- **encoder.py** — Unified encoder, auto-detects file type and applies the best compression method
- **decoder.py** — Unified decoder, reverses the compression process
- **util.py** — Shared utilities: short-name generator, index packing, PNG compression (K-Means + Color RLE V2), multi-language keyword maps
- **restore.py** — One-click recovery script (embeds all files for easy distribution)

---

## Installation

### Requirements

```bash
pip install numpy pillow scikit-learn
```

### Clone

```bash
git clone https://github.com/spzhuang/compress-proj.git
cd compress-proj
```

---

## Usage

### Encode (Compress)

```bash
# Single file
python encoder.py example.py

# Multiple files
python encoder.py file1.py file2.js file3.svg photo.png

# Directory (recursive)
python encoder.py -d ./my_project -o ./output

# PNG with custom cluster count (default: 16)
python encoder.py photo.png --clusters 32
```

### Decode (Decompress)

```bash
# Extract to default output directory
python decoder.py project.compress

# Extract to specific directory
python decoder.py project.compress ./extracted
```

### Restore (One-click recovery)

```bash
python restore.py
```

This will automatically restore `util.py`, `encoder.py`, and `decoder.py` from embedded data.

---

## Compression Format

The `.compress` file structure:

```
[4 bytes]  dict_len      -- Dictionary binary length
[N bytes]  dict_data     -- Global mapping dictionary
[2 bytes]  num_files     -- Number of files
For each file:
  [2 bytes]  fname_len   -- Filename length
[N bytes]  filename      -- Filename
[1 byte]   file_type     -- 'p'=Python, 'c'=Code, 's'=SVG, 'g'=PNG
[4 bytes]  stream_len    -- Stream length
[All streams concatenated]
[LZMA compressed]
```

---

## API

### util.py

| Function | Description |
|----------|-------------|
| `generate_short_names(count)` | Generate short identifier sequences (a-z, A-Z, _...) |
| `get_index_bytes(dict_size)` | Return index byte count based on dictionary size |
| `pack_index(value, bytes_count)` | Pack index value to fixed-length bytes |
| `read_index(data, pos, bytes_count)` | Read index value from byte stream |
| `quantize_colors(image_array, k)` | K-Means color quantization |
| `color_rle_encode(labels_2d, centers, w, h)` | Color RLE V2 encoder |
| `color_rle_decode(rle_data, w, h)` | Color RLE V2 decoder |
| `encode_png_to_stream(png_path, clusters)` | Encode PNG to RLE byte stream |
| `decode_png_from_stream(png_stream)` | Decode RLE byte stream to PIL Image |
| `detect_language(fname)` | Detect programming language from filename |

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Author

[@spzhuang](https://github.com/spzhuang)
