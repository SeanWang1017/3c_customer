"""
PPT 优化 V2: 进一步统一剩余的深蓝色 (00-1F 范围) 到主色 PRIMARY
"""
import re
import zipfile
from pathlib import Path

WORK_DIR = Path("D:/Studying/LLM/Project/.workbuddy/pptx_work")
OUT = Path("D:/Studying/LLM/Project/3C电商智能客服Agent系统-答辩.pptx")

PRIMARY = "0A1F5C"

DARK_PATTERN = re.compile(r'^([0-1][0-9A-F])', re.IGNORECASE)

slides_dir = WORK_DIR / "ppt/slides"

for sf in sorted(slides_dir.glob("slide*.xml")):
    xml = sf.read_text(encoding='utf-8')
    original = xml

    def replace_dark(m):
        color = m.group(1).upper()
        if color == PRIMARY:
            return m.group(0)
        if DARK_PATTERN.match(color):
            return f'<a:srgbClr val="{PRIMARY}"/>'
        return m.group(0)

    xml = re.sub(r'<a:srgbClr val="([0-9A-Fa-f]{6})"/>', replace_dark, xml)

    if xml != original:
        sf.write_text(xml, encoding='utf-8')
        print(f"  ✓ {sf.name}: unified dark colors")

print("\nRepacking...")
OUT.unlink(missing_ok=True)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root in WORK_DIR.rglob('*'):
        if root.is_file():
            arcname = root.relative_to(WORK_DIR).as_posix()
            zf.write(root, arcname)
print(f"✓ Saved to {OUT}")
