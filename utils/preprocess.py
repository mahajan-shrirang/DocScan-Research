import fitz
import PyPDF2
import os

from utils.merge import find_paths, merge_results, copy_images

def extract_images(doc, start_page_number, output_path="output.png"):
    page = doc.load_page(start_page_number)
    pix = page.get_pixmap()
    pix.save(output_path)

def extract_images_from_pdf_file_object(doc, start_page_number):
    page = doc.load_page(start_page_number)
    pix = page.get_pixmap()
    return pix

    

def preprocess_pdf(file):
    input_pdf = fitz.open(stream=file, filetype="pdf")
        
    images = []
    for i in range(input_pdf.page_count):
        image = extract_images_from_pdf_file_object(input_pdf, i)
        images.append(image)

    return images