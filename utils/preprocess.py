import fitz


def extract_images_from_pdf_file_object(doc, page_number) -> fitz.Pixmap:
    """Extract images from a PDF file object.

    Args:
        doc (fitz.Document): The PDF document.
        page_number (int): The page number to extract the image from.

    Returns:
        fitz.Pixmap: The extracted image
    """
    page = doc.load_page(page_number)
    pix = page.get_pixmap()
    return pix


def preprocess_pdf(file) -> list:
    """Preprocess a PDF file.

    Args:
        file (str): The path to the PDF file.

    Returns:
        list: List of images extracted from the PDF file.
    """
    input_pdf = fitz.open(stream=file, filetype="pdf")
    images = []
    for i in range(input_pdf.page_count):
        image = extract_images_from_pdf_file_object(input_pdf, i)
        images.append(image)

    return images
