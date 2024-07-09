import fitz
import PyPDF2

def extract_images(doc, start_page_number, output_path):
    page = doc.load_page(start_page_number)
    pix = page.get_pixmap()
    pix.save(output_path)