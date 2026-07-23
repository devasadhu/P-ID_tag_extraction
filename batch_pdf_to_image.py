"""
PipeSight AI - batch_pdf_to_image.py

Batch wrapper around pdf_to_image.py - converts every PDF in an input
folder to images, one subfolder per PDF (keeps pages from different
P&ID sets from colliding/overwriting each other).

Usage:
    python batch_pdf_to_image.py --input_dir data\haifa_real_pids\pdfs --output_dir data\haifa_real_pids\images
"""

import argparse
from pathlib import Path
from pdf_to_image import convert_pdf_to_images


def main():
    parser = argparse.ArgumentParser(description="PipeSight AI - Batch PDF to Image Converter")
    parser.add_argument("--input_dir", required=True, help="Folder containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Base output folder (one subfolder per PDF)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution in DPI (default: 300)")
    parser.add_argument("--format", default="png", choices=["png", "jpg"], help="Output format (default: png)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}\n")

    total_pages = 0
    for pdf_path in pdf_files:
        # one subfolder per PDF, named after the PDF stem, to avoid page-number collisions
        # across different sets (e.g. SET_1_page_01.png vs SET_2_page_01.png would otherwise
        # both be "page_01" if dumped into the same flat folder)
        pdf_output_dir = output_dir / pdf_path.stem
        print(f"=== {pdf_path.name} -> {pdf_output_dir} ===")
        saved = convert_pdf_to_images(
            pdf_path=str(pdf_path),
            output_dir=str(pdf_output_dir),
            dpi=args.dpi,
            fmt=args.format,
        )
        total_pages += len(saved)
        print()

    print(f"ALL DONE. Total pages converted across {len(pdf_files)} PDFs: {total_pages}")


if __name__ == "__main__":
    main()
