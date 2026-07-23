"""
PipeSight AI — Phase 1: PDF to Image Conversion
================================================
Converts P&ID PDF files to high-resolution images ready for
YOLO-based symbol detection.

Usage:
    python pdf_to_image.py --input <pdf_path> --output <output_folder> [--dpi 300] [--format png]

Output:
    One image per PDF page, named: <pdf_name>_page_01.png, _page_02.png, ...

Author: Sadhana Devarajan | PipeSight AI | Schneider Electric Internship
"""

import fitz  # PyMuPDF
import os
import argparse
from pathlib import Path


def convert_pdf_to_images(pdf_path, output_dir, dpi=300, fmt="png"):
    """
    Convert all pages of a PDF to images.

    Args:
        pdf_path  : Path to input PDF file
        output_dir: Folder where images will be saved
        dpi       : Resolution in DPI (300 recommended for P&IDs)
        fmt       : Output format — 'png' or 'jpg'

    Returns:
        List of saved image paths
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"[INFO] Opened: {pdf_path.name}")
    print(f"[INFO] Pages : {total_pages}")
    print(f"[INFO] DPI   : {dpi}")
    print(f"[INFO] Format: {fmt.upper()}")
    print(f"[INFO] Output: {output_dir}")
    print("-" * 50)

    # Scale matrix: PDF points are 72 DPI by default
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)

    saved_paths = []
    pdf_stem = pdf_path.stem  # filename without extension

    for page_num in range(total_pages):
        page = doc[page_num]

        # Render page to image
        pix = page.get_pixmap(matrix=mat)

        # Build output filename with zero-padded page number
        page_label = str(page_num + 1).zfill(len(str(total_pages)))
        out_filename = f"{pdf_stem}_page_{page_label}.{fmt}"
        out_path = output_dir / out_filename

        # Save image
        if fmt == "png":
            pix.save(str(out_path))
        elif fmt in ("jpg", "jpeg"):
            pix.save(str(out_path), jpg_quality=95)
        else:
            raise ValueError(f"Unsupported format: {fmt}. Use 'png' or 'jpg'.")

        size_kb = out_path.stat().st_size // 1024
        print(f"  Page {page_num + 1:>3}/{total_pages} → {out_filename}  ({pix.width}x{pix.height}px, {size_kb} KB)")
        saved_paths.append(str(out_path))

    doc.close()
    print("-" * 50)
    print(f"[DONE] {total_pages} images saved to: {output_dir}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="PipeSight AI — PDF to Image Converter")
    parser.add_argument("--input",  required=True,  help="Path to input PDF file")
    parser.add_argument("--output", required=True,  help="Output folder for images")
    parser.add_argument("--dpi",    type=int, default=300, help="Resolution in DPI (default: 300)")
    parser.add_argument("--format", default="png", choices=["png", "jpg"], help="Output image format (default: png)")
    args = parser.parse_args()

    convert_pdf_to_images(
        pdf_path=args.input,
        output_dir=args.output,
        dpi=args.dpi,
        fmt=args.format
    )


if __name__ == "__main__":
    main()