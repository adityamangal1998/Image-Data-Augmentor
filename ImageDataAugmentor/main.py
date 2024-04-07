import os
import yaml


from . import utils, core
from position_augmentor import PositionAugmentor
from noise_augmentor import NoiseAugmentor
from color_augmentor import ColorAugmentor
from image_data_augmentation import ImageDataAugmentation

image_data_aug = ImageDataAugmentation()


def main(image_list=None, image_dir_path=None, configuration_file_path=''):
    if image_dir_path is not None:
        image_data_aug.read_images_from_dir(dir_path=image_dir_path)
    elif image_list is not None:
        image_data_aug.read_images(image_list)
    configuration_data = utils.load_yaml_file(configuration_file_path)
    print(configuration_data)
    image_data_aug.read_images()
    # @TODO: Make default augmentation_types dict
    augmentation_types = {}
    if configuration_data.get('augmentation_types'):
        augmentation_types_config = configuration_data['augmentation_types']
    if 'color_augmentation' in augmentation_types_config:
        color_augmentation_config = augmentation_types_config['color_augmentation']
    if 'color_augmentation' in augmentation_types_config:
        color_augmentation_config = augmentation_types_config['color_augmentation']
    if 'color_augmentation' in augmentation_types_config:
        color_augmentation_config = augmentation_types_config['color_augmentation']
