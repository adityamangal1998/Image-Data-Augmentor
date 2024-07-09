import ConcurrentImageRead as CIR

import glob
import os

import cv2
import yaml

from . import utils
from .configs import color_config, noise_config, position_config
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

    # # def data_augmentation(self, num_threads=3, scale_factor=None,saturation_factor=1.5,brightness_factor=1.5):
    # #         """Complete Data Augmentation"""
    # #         self.augmented_images.extend(self.position_augmentor(scale_factor=scale_factor))
    # #         self.augmented_images.extend(self.color_augmentor(saturation_factor=saturation_factor,brightness_factor=brightness_factor))
    # #         # self.augmented_images.extend(self.noise_augmentor())
    # #         return self.augmented_images
    # #
    # def position_augmentor(self, num_threads=3, scale_factor=None):
    #     """Scaling,Cropping,Flipping,Rotation,Translation"""
    #     PositionAugmentor.scale(self.original_images)
    #     self.position_augmented_images.extend(self.scale_augmentor(scale_factor=scale_factor))
    #     # self.position_augmented_images.extend(core.crop())
    #     # self.position_augmented_images.extend(core.flipping())
    #     # self.position_augmented_images.extend(core.rotate())
    #     # self.position_augmented_images.extend(core.translation())
    #     return self.position_augmented_images
    #
    # def color_augmentor(self, num_threads=3, saturation_factor=1.5, brightness_factor=1.5):
    #     """Brightness,Contrast,Saturation,Hue,Salt And Pepper"""
    #     self.color_augmented_images.extend(self.saturation_augmentor(saturation_factor=saturation_factor))
    #     self.color_augmented_images.extend(self.brightness_augmentor(brightness_factor=brightness_factor))
    #     # self.color_augmented_images.extend(core.blur())
    #     # self.color_augmented_images.extend(core.contrast())
    #     # self.color_augmented_images.extend(core.sharpen())
    #     return self.color_augmented_images
    #
    # def noise_augmentor(self, num_threads=3):
    #     """Gaussian,Salt and Pepper,Speckle,Poisson,Periodic, Shadow"""
    #     self.noise_augmented_images.extend(core.gaussian_noise())
    #     self.noise_augmented_images.extend(core.salt_and_pepper_noise())
    #     self.noise_augmented_images.extend(core.speckle_noise())
    #     self.noise_augmented_images.extend(core.poisson_noise())
    #     self.noise_augmented_images.extend(core.periodic_noise())
    #     self.noise_augmented_images.extend(core.shadow_noise())
    #     return self.noise_augmented_images
    def data_augmentation(self, image_list=None, image_file_list=None, image_dir_path=None,
                          configuration_file_path='config.yaml', file_type='png'):
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
        augmentation_types = {}
        output_images = []
        if configuration_data.get('augmentation_types'):
            augmentation_types_config = configuration_data['augmentation_types']
        if 'color_augmentation' in augmentation_types_config:
            color_augmentation_config = augmentation_types_config['color_augmentation']
            output_images.extend(color_augmentor.image_augmentor(color_augmentation_config, file_names))
        if 'position_augmentation' in augmentation_types_config:
            position_augmentation_config = augmentation_types_config['position_augmentation']
            output_images.extend(position_augmentor.image_augmentor(position_augmentation_config, file_names))
        if 'noise_augmentation' in augmentation_types_config:
            noise_augmentation_config = augmentation_types_config['noise_augmentation']
            output_images.extend(noise_augmentor.image_augmentor(noise_augmentation_config, file_names))
