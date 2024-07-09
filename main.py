from utils.preprocess import extract_images
import fitz
import os

def main():
    input_pdf = fitz.open("D:\Data Science\DocScan-Research\input\Object_detection_180-1-91.pdf")
    output_folder = r'D:\Data Science\DocScan-Research\output'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for i in range(45,92):
        output_path = os.path.join(output_folder, f"output_folder{i + 1}.png")
        extract_images(input_pdf, i, output_path)
        
if __name__ == "__main__":
    main()