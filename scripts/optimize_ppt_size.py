"""
PPT 字号缩小 15%
映射表 (原 -> 新):
- 8100 (81.0pt) -> 6900 (69.0pt)  封面大标题
- 5180 (51.8pt) -> 4400 (44.0pt)  主标题
- 3600 (36.0pt) -> 3060 (30.6pt)
- 3530 (35.3pt) -> 3000 (30.0pt)
- 3450 (34.5pt) -> 2930 (29.3pt)
- 3300 (33.0pt) -> 2800 (28.0pt)
- 3230 (32.3pt) -> 2750 (27.5pt)
- 3150 (31.5pt) -> 2680 (26.8pt)
- 3080 (30.8pt) -> 2620 (26.2pt)
- 3000 (30.0pt) -> 2550 (25.5pt)
- 2930 (29.3pt) -> 2490 (24.9pt)
- 2700 (27.0pt) -> 2300 (23.0pt)
- 2620 (26.2pt) -> 2230 (22.3pt)
- 2550 (25.5pt) -> 2170 (21.7pt)
- 2400 (24.0pt) -> 2040 (20.4pt)
- 2320 (23.2pt) -> 1970 (19.7pt)
- 2170 (21.7pt) -> 1840 (18.4pt)
- 2100 (21.0pt) -> 1785 (17.85pt)
- 2020 (20.2pt) -> 1720 (17.2pt)
- 1950 (19.5pt) -> 1660 (16.6pt)
- 1870 (18.7pt) -> 1590 (15.9pt)
- 1800 (18.0pt) -> 1530 (15.3pt)
- 1730 (17.3pt) -> 1470 (14.7pt)
- 1650 (16.5pt) -> 1400 (14.0pt)
- 1600 (16.0pt) -> 1360 (13.6pt)
- 1580 (15.8pt) -> 1340 (13.4pt)
- 1500 (15.0pt) -> 1275 (12.75pt)
- 1430 (14.3pt) -> 1215 (12.15pt)
- 1350 (13.5pt) -> 1150 (11.5pt)
- 1280 (12.8pt) -> 1090 (10.9pt)
"""
import re
import zipfile
import shutil
from pathlib import Path

WORK_DIR = Path("D:/Studying/LLM/Project/.workbuddy/pptx_work")
SRC = Path("D:/Studying/LLM/Project/3C电商智能客服Agent系统-答辩.pptx")
OUT = Path("D:/Studying/LLM/Project/3C电商智能客服Agent系统-答辩.pptx")

SCALE = 0.85

# 字号映射 (整数化到 10)
def scale_size(sz):
    return round(sz * SCALE / 10) * 10

# 提取所有 sz 值
import zipfile
sizes_seen = set()
with zipfile.ZipFile(SRC, 'r') as z:
    for i in range(1, 17):
        data = z.read(f'ppt/slides/slide{i}.xml').decode('utf-8', errors='ignore')
        for s in re.findall(r'<a:rPr[^>]*?sz="(\d+)"', data):
            sizes_seen.add(int(s))

print("=== 字号缩放映射 ===")
size_map = {}
for sz in sorted(sizes_seen, reverse=True):
    new = scale_size(sz)
    size_map[sz] = new
    print(f"  {sz:>5} ({sz/100:>5.1f}pt) -> {new:>5} ({new/100:>5.1f}pt)")

# 解压
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True)
with zipfile.ZipFile(SRC, 'r') as z:
    z.extractall(WORK_DIR)

# 修改所有 slide 和 layout / master 中的 sz
slides_dir = WORK_DIR / "ppt/slides"
layout_dir = WORK_DIR / "ppt/slideLayouts"
master_dir = WORK_DIR / "ppt/slideMasters"

modified_count = 0
for d in [slides_dir, layout_dir, master_dir]:
    for f in d.glob("*.xml"):
        xml = f.read_text(encoding='utf-8')
        original = xml
        # 替换 sz="N"
        def replace_sz(m):
            old = int(m.group(1))
            new = size_map.get(old, old)
            return f'sz="{new}"'
        xml = re.sub(r'sz="(\d+)"', replace_sz, xml)
        if xml != original:
            f.write_text(xml, encoding='utf-8')
            modified_count += 1
            print(f"  ✓ {f.relative_to(WORK_DIR)}")

print(f"\n共修改 {modified_count} 个文件")

# 重打包
OUT.unlink(missing_ok=True)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root in WORK_DIR.rglob('*'):
        if root.is_file():
            arcname = root.relative_to(WORK_DIR).as_posix()
            zf.write(root, arcname)
print(f"✓ 保存到 {OUT}")

# 验证
import zipfile, re
print("\n=== 验证 (新字号) ===")
all_sizes = set()
with zipfile.ZipFile(OUT, 'r') as z:
    for i in range(1, 17):
        data = z.read(f'ppt/slides/slide{i}.xml').decode('utf-8', errors='ignore')
        for s in re.findall(r'<a:rPr[^>]*?sz="(\d+)"', data):
            all_sizes.add(int(s))
print(f"字号: {sorted(all_sizes, reverse=True)}")
