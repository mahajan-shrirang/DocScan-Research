import os
import json
import shutil
import numpy as np
from PIL import Image


def find_paths(file_name, base_paths) -> list:
    """Find the paths of the file with the given name in the base paths.

    Args:
        file_name (str): The name of the file to search.
        base_paths (list): List of base paths to search for the file.

    Returns:
        list: List of paths where the file is found.
    """
    file_paths = []
    for path in base_paths:
        for root, dirs, files in os.walk(path):
            if file_name in files:
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)
    return file_paths


def merge_results(paths, output_path) -> None:
    """ Merge the results from multiple COCO files into a single COCO file.

    Args:
        paths (list): List of paths to the COCO files.
        output_path (str): Path to save the combined COCO file.

    Returns:
        None
    """
    combined_coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    image_id_offset = 0
    annotation_id_offset = 0
    image_id_mapping = {}

    for path in paths:
        with open(path, 'r') as f:
            coco_data = json.load(f)

            if not combined_coco['categories']:
                combined_coco['categories'] = coco_data['categories']

            for image in coco_data['images']:
                new_image_id = image['id'] + image_id_offset
                image_id_mapping[image['id']] = new_image_id
                image['id'] = new_image_id
                combined_coco['images'].append(image)

            for annotation in coco_data['annotations']:
                new_annotation_id = annotation['id'] + annotation_id_offset
                new_image_id = image_id_mapping[annotation['image_id']]
                annotation['id'] = new_annotation_id
                annotation['image_id'] = new_image_id
                combined_coco['annotations'].append(annotation)

            image_id_offset += len(coco_data['images'])
            annotation_id_offset += len(coco_data['annotations'])

    with open(output_path, 'w') as f:
        f.writelines(json.dumps(combined_coco, indent=4))


def copy_images(base_paths, output_images_folder) -> list:
    """Copy images from multiple folders into a single folder.

    Args:
        base_paths (list): List of base paths to search for the images.
        output_images_folder (str): Path to save the combined images.

    Returns:
        list: List of paths where the file is found.
    """
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    for path in base_paths:
        images_folder = os.path.join(path, 'Images')
        for image_filename in os.listdir(images_folder):
            src_image_path = os.path.join(images_folder, image_filename)
            dst_image_path = os.path.join(output_images_folder, image_filename)
            shutil.copy(src_image_path, dst_image_path)


def merge_images_and_save_pdf(images: list[np.ndarray],
                              save_path: str) -> None:
    """Merge images and save as PDF.

    Args:
        images (list[np.ndarray]): List of images to merge.
        save_path (str): Path to save the PDF.

    Returns:
        None
    """
    images = [Image.fromarray(image) for image in images]
    return images[0].save(save_path, "PDF", save_all=True,
                          append_images=images[1:], resolution=100.0)
