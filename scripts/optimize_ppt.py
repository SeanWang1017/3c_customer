"""
PPT 视觉风格统一优化脚本
目标:
1. 统一标题色为深蓝主调 (#0A1F5C)
2. 统一正文字色为深灰蓝 (#1F2937)
3. 修复"颜色=背景色"的隐形文字
4. 修复文字错误 ("|效果展示", "邮件已达", 占位符 [你的名字] 等)
5. 优化字号层级 (标题/小标题/正文/标注)
6. 提升标题与正文的对比度

策略: 直接操作 slide{N}.xml, 替换 srgbClr val="XXXXXX" 和文本
"""
import re
import shutil
import zipfile
from pathlib import Path

SRC = Path("D:/Studying/LLM/Project/3C电商智能客服Agent系统-答辩.pptx")
WORK_DIR = Path("D:/Studying/LLM/Project/.workbuddy/pptx_work")
OUT = Path("D:/Studying/LLM/Project/3C电商智能客服Agent系统-答辩.pptx")

# 统一后的色板
PRIMARY = "0A1F5C"        # 主标题色 - 深蓝
PRIMARY_LIGHT = "1E3A8A"  # 副标题色 - 中蓝
BODY = "1F2937"          # 正文字色 - 深灰
BODY_LIGHT = "4B5563"    # 辅助正文
ACCENT = "DC2626"        # 强调色 - 红 (关键数据)
ACCENT_WARN = "F59E0B"   # 警告色 - 橙

# 隐形文字修复: F2F6F9 / F3F6F9 (背景色) -> 改为深色
INVISIBLE_COLORS = {"F2F6F9", "F3F6F9", "F4F7FA", "F1F5F9"}

# 标题字号 (EMU) - 在原基础上统一
# 原: 657860, 448310, 419100, 410210, 400050, 372110  -> 统一到 457200
# 小标题: 275590, 294640, 342900 -> 统一到 304800
# 正文: 200660, 209550, 219710, 228600 -> 统一到 209550
# 辅助: 181610, 190500, 200660 -> 统一到 190500

# 文本替换: 错误修复
TEXT_FIXES = {
    "|效果展示": "效果展示",
    "邮件已达": "邮件已送达",
    "[你的名字]": "Sean",
    "[老师姓名]": "指导老师",
    "2026年X月X日": "2026年6月",
    "技术方案:Qwen2.5-0.5B + QLoRA int4 + 关键词规则": "技术方案:Qwen2.5-0.5B + QLoRA int4 + 关键词规则",
}


def extract_pptx(src, dest):
    """解压 pptx"""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(src, 'r') as z:
        z.extractall(dest)


def repack_pptx(work_dir, out):
    """重新打包"""
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root in work_dir.rglob('*'):
            if root.is_file():
                arcname = root.relative_to(work_dir).as_posix()
                zf.write(root, arcname)


def fix_text(xml):
    """修复文字错误"""
    for old, new in TEXT_FIXES.items():
        xml = xml.replace(old, new)
    return xml


def unify_title_color(xml):
    """统一所有标题颜色 (大字 + bold) 为 PRIMARY"""
    # 找出所有 <a:rPr ... sz="..." b="1" ...> 的 run 颜色
    # 这是大标题 run, 把它们的颜色统一
    def fix_run(m):
        attrs = m.group(1)
        # 检查是否是粗体大字 (>= 280000 EMU ~= 22pt)
        sz_match = re.search(r'sz="(\d+)"', attrs)
        b_match = re.search(r'b="1"', attrs)
        if sz_match and b_match and int(sz_match.group(1)) >= 280000:
            # 替换颜色 - 移除现有颜色, 改为 PRIMARY
            attrs = re.sub(r'<a:solidFill>.*?</a:solidFill>', '', attrs, flags=re.DOTALL)
            return f'<a:rPr{attrs}>'
        return m.group(0)

    # 在 rPr 中重新插入 solidFill
    # 简化: 直接扫描所有 srgbClr, 看上下文
    return xml


def fix_run_colors(xml):
    """修复 run 颜色: 统一主标题色 + 修复隐形色"""
    # 策略: 使用正则找到 <a:rPr ...><a:solidFill><a:srgbClr val="..."/></a:solidFill></a:rPr>
    # 替换颜色
    
    # 1. 修复隐形色 (背景色) -> 改为 BODY
    for inv in INVISIBLE_COLORS:
        pattern = f'<a:solidFill><a:srgbClr val="{inv}"/></a:solidFill>'
        xml = xml.replace(pattern, f'<a:solidFill><a:srgbClr val="{BODY}"/></a:solidFill>')
    
    # 2. 找出所有 run 的颜色, 如果是 "深蓝系" (06/07/08/09 开头), 统一为 PRIMARY
    def replace_dark_blue(m):
        color = m.group(1)
        # 主标题色系 (深蓝): 04-0F 开头 (浅) 或 06-09 开头 (中深)
        if re.match(r'^(0[4-9]|1[0-9])', color):
            return f'<a:srgbClr val="{PRIMARY}"/>'
        return m.group(0)
    
    xml = re.sub(r'<a:srgbClr val="([0-9A-Fa-f]{6})"/>', replace_dark_blue, xml)
    
    return xml


def fix_title_colors_in_slide(slide_path):
    """对每张 slide, 把大字号 (>= 350000) 的标题 run 颜色统一为 PRIMARY"""
    xml = slide_path.read_text(encoding='utf-8')
    
    # 找到 <a:rPr ... sz="N" b="1" ...>...</a:rPr> 块 (含 solidFill)
    # 替换其中的 srgbClr
    pattern = re.compile(
        r'(<a:rPr\s[^>]*?sz="(\d+)"[^>]*?b="1"[^>]*?>)'
        r'(<a:solidFill><a:srgbClr val=")([0-9A-Fa-f]{6})("/></a:solidFill>)?',
        re.DOTALL
    )
    
    def replace_title(m):
        prefix = m.group(1)
        sz = int(m.group(2))
        if sz >= 280000:  # 大字标题
            if m.group(3):
                # 已有 solidFill, 替换颜色
                return f'{prefix}<a:solidFill><a:srgbClr val="{PRIMARY}"/></a:solidFill>'
            else:
                # 插入 solidFill
                return f'{prefix}<a:solidFill><a:srgbClr val="{PRIMARY}"/></a:solidFill>'
        return m.group(0)
    
    # 处理带 solidFill 的情况
    def replace_title_full(m):
        full = m.group(0)
        prefix = m.group(1)
        sz = int(m.group(2))
        if sz >= 280000:
            return f'{prefix}<a:solidFill><a:srgbClr val="{PRIMARY}"/></a:solidFill></a:rPr>'
        return full
    
    # 找到 rPr 节点 (含 sz 和 b=1)
    rpr_pattern = re.compile(
        r'<a:rPr\s+([^>]*?)\s+b="1"([^>]*?)>(.*?)</a:rPr>',
        re.DOTALL
    )
    
    def process_rpr(m):
        attrs_before = m.group(1)
        attrs_after = m.group(2)
        inner = m.group(3)
        # 找 sz
        sz_match = re.search(r'sz="(\d+)"', attrs_before + " " + attrs_after)
        if not sz_match or int(sz_match.group(1)) < 280000:
            return m.group(0)
        # 替换 inner 中的 srgbClr
        new_inner = re.sub(
            r'<a:srgbClr val="[0-9A-Fa-f]{6}"/>',
            f'<a:srgbClr val="{PRIMARY}"/>',
            inner
        )
        return f'<a:rPr {attrs_before} b="1"{attrs_after}>{new_inner}</a:rPr>'
    
    # 同时处理 rPr 的多种写法
    for rpr_pat in [
        r'<a:rPr\s+([^>]*?)\s+b="1"([^>]*?)>(.*?)</a:rPr>',
        r'<a:rPr\s+b="1"([^>]*?)>(.*?)</a:rPr>',
        r'<a:rPr\s+([^>]*?)\s+b="1"\s*/>',
        r'<a:rPr\s+b="1"\s*/>',
    ]:
        def make_replace(pat_str):
            pat = re.compile(pat_str, re.DOTALL)
            def rep(m):
                groups = m.groups()
                # 找到 sz
                all_attrs = ' '.join(g for g in groups if g)
                sz_match = re.search(r'sz="(\d+)"', all_attrs)
                if not sz_match or int(sz_match.group(1)) < 280000:
                    return m.group(0)
                # 自闭合
                if pat_str.endswith('/>'):
                    return re.sub(
                        r'(<a:rPr[^>]*?)(/?>)$',
                        lambda mm: f'{mm.group(1)}><a:solidFill><a:srgbClr val="{PRIMARY}"/></a:solidFill></a:rPr>' if mm.group(2) == '/>' else mm.group(0),
                        m.group(0)
                    )
                # 非自闭合: 处理 inner
                inner = m.group(2) if len(groups) >= 2 else ''
                if not inner:
                    inner = groups[-1] if groups else ''
                new_inner = re.sub(
                    r'<a:srgbClr val="[0-9A-Fa-f]{6}"/>',
                    f'<a:srgbClr val="{PRIMARY}"/>',
                    inner
                )
                # 重建
                full = m.group(0)
                # 把 inner 替换掉
                return full.replace(inner, new_inner, 1)
            return rep
        
        xml = re.sub(pat, make_replace(pat_str), xml)
    
    return xml


def main():
    print("Extracting pptx...")
    extract_pptx(SRC, WORK_DIR)
    
    slides_dir = WORK_DIR / "ppt/slides"
    slide_files = sorted(slides_dir.glob("slide*.xml"))
    print(f"Found {len(slide_files)} slide XML files")
    
    for sf in slide_files:
        original = sf.read_text(encoding='utf-8')
        
        # Step 1: 修复文字错误
        xml = fix_text(original)
        
        # Step 2: 修复隐形色 (F2F6F9 系列)
        for inv in INVISIBLE_COLORS:
            xml = xml.replace(
                f'<a:srgbClr val="{inv}"/>',
                f'<a:srgbClr val="{BODY}"/>'
            )
        
        # Step 3: 统一深蓝色系到主色 PRIMARY
        def replace_dark_blue(m):
            color = m.group(1).upper()
            # 主标题色系 (深蓝): 06-0F 开头
            if re.match(r'^(0[6-9]|1[0-5])', color):
                return f'<a:srgbClr val="{PRIMARY}"/>'
            return m.group(0)
        xml = re.sub(r'<a:srgbClr val="([0-9A-Fa-f]{6})"/>', replace_dark_blue, xml)
        
        # Step 4: 单独把 000000 黑色标题色 (slide 5) 改为 PRIMARY
        # 已包含在 step 3 之外, 因为 000000 不在 06-15 范围
        
        if xml != original:
            sf.write_text(xml, encoding='utf-8')
            print(f"  ✓ {sf.name} updated")
    
    # Step 5: 修改 slide 5 的标题色 (它是 000000 黑色, 单独处理)
    slide5 = slides_dir / "slide5.xml"
    if slide5.exists():
        xml = slide5.read_text(encoding='utf-8')
        # 找 "4 重保险与工具协作" 标题 run, 把颜色改为 PRIMARY
        # 简单做法: 在 <a:rPr b="1" sz="..."><a:solidFill> 后改色
        # 但因为是黑色不是深蓝, 我们特殊处理
        # 把 sz=419100 的大字 000000 替换为 PRIMARY
        # 这是一个 run 属性 模式
        pattern = r'(<a:rPr\s+lang="zh-CN"\s+altLang="en-US"\s+sz="419100"\s+b="1"[^>]*?>)(\s*<a:solidFill>\s*<a:srgbClr val=")000000("/>\s*</a:solidFill>)'
        new_xml = re.sub(pattern, lambda m: f'{m.group(1)}{m.group(2)}{PRIMARY}{m.group(3)}', xml)
        if new_xml != xml:
            slide5.write_text(new_xml, encoding='utf-8')
            print(f"  ✓ slide5: title color 000000 -> {PRIMARY}")
    
    # Step 6: 修复黑色文字 (正文中有几处用了 000000)
    for sf in slide_files:
        xml = sf.read_text(encoding='utf-8')
        # 黑色 000000 -> BODY (深灰) 但保留重要强调
        # 这里只把 000000 替换为 BODY, 因为正文中用纯黑不够柔和
        new_xml = re.sub(
            r'<a:srgbClr val="000000"/>',
            f'<a:srgbClr val="{BODY}"/>',
            xml
        )
        if new_xml != xml:
            sf.write_text(new_xml, encoding='utf-8')
    
    # Step 7: 修复文本错误 (F2F6F9 之外)
    for sf in slide_files:
        xml = sf.read_text(encoding='utf-8')
        new_xml = fix_text(xml)
        if new_xml != xml:
            sf.write_text(new_xml, encoding='utf-8')
            print(f"  ✓ {sf.name}: text fixes")
    
    # Repack
    print("\nRepacking pptx...")
    repack_pptx(WORK_DIR, OUT)
    print(f"✓ Saved to {OUT}")


if __name__ == "__main__":
    main()
