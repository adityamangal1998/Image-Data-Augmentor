import ConcurrentImageRead as CIR
import glob
import os

import cv2

from . import utils, config
from .augmentors.position_augmentor import PositionAugmentor
from .augmentors.noise_augmentor import NoiseAugmentor
from .augmentors.color_augmentor import ColorAugmentor

color_augmentor = ColorAugmentor()
position_augmentor = PositionAugmentor()
noise_augmentor = NoiseAugmentor()


class ImageDataAugmentation:
    def __init__(self):
        self.original_images = []
        self.position_augmented_images = []
        self.color_augmented_images = []
        self.noise_augmented_images = []
        self.augmented_images = []

    def read_images(self, image_list, num_threads=3, channel_type='BGR', root_path='data'):
        """

        :param image_list:
        :param num_threads:
        :param channel_type:
        :param root_path:
        :return:
        """
        self.original_images = CIR.read(image_list, num_threads=num_threads, channel_type=channel_type,
                                        root_path=root_path)
        return self.original_images

    def read_images_from_dir(self, dir_path, file_type='png', num_threads=3, channel_type='BGR', sub_dir=False):
        """

        :param dir_path:
        :param file_type:
        :param num_threads:
        :param channel_type:
        :param sub_dir:
        :return:
        """
        self.original_images = CIR.read_dir(dir_path, file_type=file_type, num_threads=num_threads,
                                            channel_type=channel_type,
                                            sub_dir=sub_dir)
        return self.original_images

    def data_augmentation(self, image_list=None, image_file_list=None, image_dir_path=None,
                          configuration_file_path='config.yaml', file_type='png', output_dir=None, is_save_image=False):
        """

        :param image_list:
        :param image_file_list:
        :param image_dir_path:
        :param configuration_file_path:
        :param file_type:
        :return:
        """
        configuration_data = utils.load_yaml_file(configuration_file_path)
        image_file_type = configuration_data['file_type'] if configuration_data.get(
            'file_type') else config.image_file_type
        if image_dir_path is not None:
            self.read_images_from_dir(dir_path=image_dir_path, file_type=image_file_type)
            image_files = glob.glob(os.path.join(image_dir_path, "*." + image_file_type))
            file_names = []
            for image_file in image_files:
                file_names.append(utils.input_file_path_info(image_file)['file_name'])
        elif image_file_list is not None:
            self.read_images(image_file_list)
            file_names = []
            for image_file in image_file_list:
                file_names.append(utils.input_file_path_info(image_file)['file_name'])
        elif image_list is not None:
            self.original_images = image_list
            file_names = []
            for index in range(len(image_list)):
                file_names.append(str(index) + "." + image_file_type)
        # @TODO: Make default augmentation_types dict
        output_images = []
        augmentation_types_config = configuration_data['augmentation_types'] if configuration_data.get(
            'augmentation_types') else {}
        if 'color_augmentation' in augmentation_types_config:
            color_augmentation_config = augmentation_types_config['color_augmentation']
            self.color_augmented_images = color_augmentor.image_augmentor(self.original_images,
                                                                          color_augmentation_config, file_names)
            output_images.extend(self.color_augmented_images)
        if 'position_augmentation' in augmentation_types_config:
            position_augmentation_config = augmentation_types_config['position_augmentation']
            self.position_augmented_images = position_augmentor.image_augmentor(self.original_images,
                                                                                position_augmentation_config,
                                                                                file_names)
            output_images.extend(self.position_augmented_images)
        if 'noise_augmentation' in augmentation_types_config:
            noise_augmentation_config = augmentation_types_config['noise_augmentation']
            self.noise_augmented_images = noise_augmentor.image_augmentor(self.original_images,
                                                                          noise_augmentation_config, file_names)
            output_images.extend(self.noise_augmented_images)

        if is_save_image:
            if output_dir is None:
                output_dir = 'output'
            for output_image in output_images:
                cv2.imwrite(os.path.join(output_dir, output_image[-1]), output_image[0])
            return "Augmented Images Saved Successfully"
        else:
            return output_images
