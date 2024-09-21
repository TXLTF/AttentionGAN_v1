from pdfminer.high_level import extract_text
from markdown import markdown
import re


def pdf_to_markdown(pdf_path, md_path):
    # 提取PDF文本
    text = extract_text(pdf_path)

    # 简单的文本处理
    lines = text.split('\n')
    markdown_lines = []

    for line in lines:
        line = line.strip()
        if line:
            # 检测标题
            if re.match(r'^[A-Z]', line) and len(line) < 50:
                markdown_lines.append(f"## {line}\n")
            else:
                markdown_lines.append(f"{line}\n\n")

    markdown_text = ''.join(markdown_lines)

    # 保存为Markdown文件
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_text)

    print(f"已将PDF转换为Markdown并保存到 {md_path}")


# 使用示例
pdf_to_markdown('docs/基于残差注意力网络的多针迹刺绣图像生成算法研究_杨辰 (1).pdf',
                'docs/基于残差注意力网络的多针迹刺绣图像生成算法研究_杨辰 (1).md')
