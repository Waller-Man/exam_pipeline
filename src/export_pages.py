import sys
from pathlib import Path

def export_pdf(pdf_path: Path, out_dir: Path, dpi: int = 200):
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72  # PDF 默认 72dpi
    mat = fitz.Matrix(zoom, zoom)

    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out_dir / f"{i + 1}.png"))

    doc.close()

def docx_to_pdf(docx_path: Path) -> Path:
    from docx2pdf import convert
    pdf_path = docx_path.with_suffix(".pdf")
    convert(str(docx_path), str(pdf_path))
    return pdf_path

def main():
    if len(sys.argv) < 2:
        print('用法: python export_pages.py "文件路径" [dpi]')
        sys.exit(1)

    in_path = Path(sys.argv[1]).expanduser().resolve()
    dpi = int(sys.argv[2]) if len(sys.argv) >= 3 else 200

    if not in_path.exists():
        print("文件不存在:", in_path)
        sys.exit(2)

    # 输出目录：与输入文件同目录下，创建同名文件夹
    out_dir = in_path.parent / in_path.stem

    if in_path.suffix.lower() == ".pdf":
        export_pdf(in_path, out_dir, dpi=dpi)
        print("完成:", out_dir)

    elif in_path.suffix.lower() in (".docx", ".doc"):
        # 先转 PDF 再导出
        pdf_path = docx_to_pdf(in_path)
        export_pdf(pdf_path, out_dir, dpi=dpi)
        print("完成:", out_dir)
        print("已生成中间 PDF:", pdf_path)

    else:
        print("不支持的格式:", in_path.suffix)
        sys.exit(3)

if __name__ == "__main__":
    main()