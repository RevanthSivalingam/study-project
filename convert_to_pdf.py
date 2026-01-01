#!/usr/bin/env python3
"""
Script to convert text policy documents to PDF format
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import os

def text_to_pdf(text_file, pdf_file):
    """Convert text file to PDF"""
    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Create PDF
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter

    # Set font
    c.setFont("Courier", 9)

    # Starting position
    y = height - 0.5 * inch
    x = 0.5 * inch
    line_height = 11

    for line in lines:
        # Remove trailing newline
        line = line.rstrip('\n')

        # Check if we need a new page
        if y < 0.5 * inch:
            c.showPage()
            c.setFont("Courier", 9)
            y = height - 0.5 * inch

        # Handle long lines by truncating
        if len(line) > 85:
            line = line[:85]

        # Draw the line
        c.drawString(x, y, line)
        y -= line_height

    # Save the PDF
    c.save()
    print(f"✅ Created: {pdf_file}")

def main():
    """Main function to convert all text files to PDF"""
    txt_files = [
        "data/pdfs/employee_leave_policy.txt",
        "data/pdfs/employee_benefits_policy.txt",
        "data/pdfs/remote_work_policy.txt",
        "data/pdfs/performance_review_policy.txt"
    ]

    print("=" * 70)
    print("Converting Policy Documents to PDF")
    print("=" * 70)

    for txt_file in txt_files:
        if os.path.exists(txt_file):
            pdf_file = txt_file.replace('.txt', '.pdf')
            try:
                text_to_pdf(txt_file, pdf_file)
            except Exception as e:
                print(f"❌ Error converting {txt_file}: {e}")
        else:
            print(f"⚠️  File not found: {txt_file}")

    print("\n" + "=" * 70)
    print("✅ Conversion Complete!")
    print("=" * 70)
    print("\nPDF files created in data/pdfs/ directory:")
    for txt_file in txt_files:
        pdf_file = txt_file.replace('.txt', '.pdf')
        if os.path.exists(pdf_file):
            size = os.path.getsize(pdf_file) / 1024
            print(f"  - {os.path.basename(pdf_file)} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
