import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import ntpath
from pathlib import Path
from functools import partial
from concurrent.futures.thread import ThreadPoolExecutor


def load_yaml_file(file_path):
    """
    Load YAML file
    :param file_path: file path
    :return: dict data
    """
    try:
        with open(file_path, 'r') as stream:
            data = yaml.safe_load(stream)
        return data
    except:
        return {}


def check_directory(path):
    """
    Check Existing Path if present then delete the directory and create a new directory
    :param path: directory path
    :return: None
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def input_file_path_info(input_file=None):
    """
    Extract file path info
    :param input_file: input file path
    :return: dictionary of file path info
    """
    file_dir, file_name = ntpath.split(input_file)
    file_stem, file_extension = Path(file_name).stem, Path(file_name).suffix
    return {'file_dir': file_dir, 'file_name': file_name, 'file_stem': file_stem, 'file_extension': file_extension}


# read annotation data from xml
def read_xml_data(xml_file):
    """

    :param xml_file:
    :return:
    """
    file = open(xml_file, 'r')
    tree = ET.parse(file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []

    for obj in root.iter('object'):
        cls = obj.find('name').text
        bbox = [int(x.text) for x in obj.find("bndbox")]
        objects.append([cls, bbox])

    return width, height, objects


def xml_to_yolo(bbox, w, h):
    """

    :param bbox:
    :param w:
    :param h:
    :return:
    """
    """
    Yolo coordinates are calculated by normalizing the Bounding Box coordinates.
    It is important to know the Width and Height of the image whose normalized coordinates are to be calculated
    for YOLO format.
    bbox = Only xmin, ymin, xmax, ymax are passed in the list. These are the actual values read from the xml file
    """
    x_c = ((bbox[2] + bbox[0]) / 2) / w
    y_c = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_c, y_c, width, height]


def rotateYoloBB(rotate_image, xml_data, angle=135):
    """

    :param rotate_image:
    :param xml_data:
    :param angle:
    :return:
    """
    """
    This function focuses on finding the new coordinates of the BB after Image Rotation.
    """
    new_ht, new_wd = rotate_image.shape[:2]

    # Read XML from data
    W, H, objects = xml_data

    # Store Bounding Box coordinates from an image
    new_bbox = []
    rotation_angle = angle * np.pi / 180
    # To store the differnt object classes, later used to generate the classes.txt file used for YOLO training.
    global classes
    rot_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                           [np.sin(rotation_angle), np.cos(rotation_angle)]])
    # Convert XML data to YOLO format
    for i, obj in enumerate(objects):
        if obj[0] not in classes:
            classes.append(obj[0])

        # From XML
        xmin = obj[1][0]
        ymin = obj[1][1]
        xmax = obj[1][2]
        ymax = obj[1][3]

        # Calculating coordinates of corners of the Bounding Box
        top_left = (xmin - W / 2, -H / 2 + ymin)
        top_right = (xmax - W / 2, -H / 2 + ymin)
        bottom_left = (xmin - W / 2, -H / 2 + ymax)
        bottom_right = (xmax - W / 2, -H / 2 + ymax)

        # Calculate new coordinates (after rotation)
        new_top_left = []
        new_bottom_right = [-1, -1]

        for j in (top_left, top_right, bottom_left, bottom_right):
            # Generate new corner coords by multiplying it with the transformation matrix (2,2)
            new_coords = np.matmul(rot_matrix, np.array((j[0], -j[1])))

            x_prime, y_prime = new_wd / 2 + new_coords[0], new_ht / 2 - new_coords[1]

            # Finding the new top-left coords, by finding the minimum of calculated x and y-values
            if len(new_top_left) > 0:
                if new_top_left[0] > x_prime:
                    new_top_left[0] = x_prime
                if new_top_left[1] > y_prime:
                    new_top_left[1] = y_prime
            else:  # for first iteration, lists are empty, therefore directly append
                new_top_left.append(x_prime)
                new_top_left.append(y_prime)

            # Finding the new bottom-right coords, by finding the maximum of calculated x and y-values
            if new_bottom_right[0] < x_prime:
                new_bottom_right[0] = x_prime
            if new_bottom_right[1] < y_prime:
                new_bottom_right[1] = y_prime

        # i(th) index of the object
        new_bbox.append([obj[0], [new_top_left[0], new_top_left[1], new_bottom_right[0], new_bottom_right[1]]])

    return new_bbox


def cv_to_yolo(image_bboxes, ht, wd):
    """

    :param image_bboxes:
    :param ht:
    :param wd:
    :return:
    """
    """
    This function works similar to the above xml_to_yolo(). The only point of difference is the bbox size.
    main_bbox = The BB cords along with the index is passed. This is the output of the rotated images.
    """
    yolo_boxes = []
    for main_bbox in image_bboxes:
        class_name = main_bbox[0]
        main_bbox = main_bbox[1]
        bb_width = main_bbox[2] - main_bbox[0]
        bb_height = main_bbox[3] - main_bbox[1]
        bb_cx = (main_bbox[0] + main_bbox[2]) / 2
        bb_cy = (main_bbox[1] + main_bbox[3]) / 2
        yolo_boxes.append(
            [classes.index(class_name), round(bb_cx / wd, 6), round(bb_cy / ht, 6), round(bb_width / wd, 6),
             round(bb_height / ht, 6)])
    return yolo_boxes


def save_yolo_boxes(output_path, yolo_boxes):
    """

    :param output_path:
    :param yolo_boxes:
    :return:
    """
    res = []
    for i, yolo_bbox in enumerate(yolo_boxes):
        # convert data to string
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        res.append(f"{bbox_string}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(res))


def process_data_with_multi_threading(calling_function_object, function_arguments, list_data, num_threads=3):
    """
    Process Data with Multi-Threading
    :param calling_function_object: calling function object
    :param function_arguments: function arguments
    :param list_data: list
    :param num_threads: number of threads
    :return:
    """
    try:
        calling_function = partial(calling_function_object, function_arguments)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            result = list(executor.map(calling_function, list_data))
        return result
    except Exception as e:
        print(f"Error : {e}")
        return []

def check_and_create_directory(path):
    """
    Check if directory exists, if not create it
    :param path: directory path
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"Directory {path} already exists.")