from utils.preprocess import extract_images
from utils.merge import find_paths, merge_results, copy_images
import fitz
import os

def main():
    input_pdf = fitz.open("D:\Data Science\DocScan-Research\input\Object_detection_180.pdf")
    output_folder = r'D:\Data Science\DocScan-Research\Extracted Images'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for i in range(0,145):
        output_path = os.path.join(output_folder, f"output_folder{i + 1}.png")
        extract_images(input_pdf, i, output_path)
        
    # base_paths = [
    #     r'D:\Data Science\DocScan-Research\output\coco files\set 1',
    #     r'D:\Data Science\DocScan-Research\output\coco files\set 2',
    #     r'D:\Data Science\DocScan-Research\output\coco files\set 3',
    #     r'D:\Data Science\DocScan-Research\output\coco files\set 4'
    # ]

    # result_json_paths = find_paths("result.json", base_paths)
    # output_path = r"D:\Data Science\DocScan-Research\output\annotated\images\result.json"
    # if not os.path.exists(os.path.dirname(output_path)):
    #     os.makedirs(os.path.dirname(output_path))
    # output_images_folder = r"D:\Data Science\DocScan-Research\output\annotated\images"

    # merge_results(result_json_paths, output_path)
    # copy_images(base_paths, output_images_folder)
        
if __name__ == "__main__":
    main()