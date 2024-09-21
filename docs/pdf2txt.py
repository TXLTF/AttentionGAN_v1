import PyPDF2
import os


def pdf_to_txt(pdf_path, txt_path):
    try:
        # 检查PDF文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        # 打开PDF文件
        with open(pdf_path, 'rb') as pdf_file:
            try:
                # 创建PDF阅读器对象
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # 打开要写入的TXT文件
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    # 遍历PDF的每一页
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            # 提取文本并写入TXT文件
                            text = page.extract_text()
                            txt_file.write(text)
                            txt_file.write('\n\n')  # 每页后添加两个换行
                            print(f"已处理第 {page_num} 页")
                        except Exception as e:
                            print(f"处理第 {page_num} 页时出错: {str(e)}")

                print(f"PDF已成功转换为TXT: {txt_path}")

            except PyPDF2.errors.PdfReadError:
                print("无法读取PDF文件，可能是加密或损坏的文件。")

    except Exception as e:
        print(f"转换过程中出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    pdf_path = 'docs/基于残差注意力网络的多针迹刺绣图像生成算法研究_杨辰 (1).pdf'
    txt_path = 'docs/基于残差注意力网络的多针迹刺绣图像生成算法研究_杨辰 (1).txt'

    pdf_to_txt(pdf_path, txt_path)
