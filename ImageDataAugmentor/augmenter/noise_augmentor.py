import cv2
import numpy as np

from ImageDataAugmentor import utils
from ImageDataAugmentor.configs import noise_config


class NoiseAugmentor:
    def __init__(self):
        pass

    def gaussian_noise(self, image, mean=0, std=25):
        """
        Gaussian Noise
        :param image: image matrix
        :param mean:
        :param std:
        :return: gaussian noised image matrix
        """
        gaussian_noise = np.random.normal(mean, std, image.shape)
        noisy_image = np.clip(image + gaussian_noise, 0, 255)
        return noisy_image.astype(np.uint8)

    def salt_and_pepper_noise(self, image, salt_prob=0.02, pepper_prob=0.02):
        """
        Salt and Pepper Noise
        :param image:
        :param salt_prob:
        :param pepper_prob:
        :return: salt and pepper noised image matrix
        """
        noisy_image = image.copy()
        # Salt noise (white pixels)
        salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
        noisy_image[salt_mask] = 255
        # Pepper noise (black pixels)
        pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
        noisy_image[pepper_mask] = 0
        return noisy_image

    def speckle_noise(self, image, scale=0.1):
        """
        Speckle Noise
        :param image:
        :param scale:
        :return:
        """
        speckle_noise = np.random.normal(0, scale, image.shape)
        noisy_image = np.clip(image + image * speckle_noise, 0, 255)
        return noisy_image.astype(np.uint8)

    def poisson_noise(self, image):
        """
        Poisson Noise
        :param image:
        :return:
        """
        noisy_image = np.random.poisson(image / 255.0) * 255
        return noisy_image.astype(np.uint8)

    def periodic_noise(self, image, frequency=10):
        """
        Periodic Noise
        :param image:
        :param frequency:
        :return:
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        noise = 128 * np.sin(2 * np.pi * frequency * y / image.shape[0])
        noisy_image = np.clip(image + noise, 0, 255)
        return noisy_image.astype(np.uint8)

    def shadow_noise(self, image, shadow_prob=0.1):
        """
        Shadow Noise
        :param image:
        :param shadow_prob:
        :return:
        """
        noisy_image = image.copy()
        # Add shadow noise
        shadow_mask = np.random.rand(*image.shape[:2]) < shadow_prob
        noisy_image[shadow_mask] = np.clip(image[shadow_mask] * 0.5, 0, 255)
        return noisy_image

    def image_augmentor(self, input_images, augmentation_config, file_names):
        """

        :param input_images:
        :param augmentation_config:
        :param file_names:
        :return:
        """
        output_images = []
        for image_index, image in enumerate(input_images):
            file_stem = utils.input_file_path_info(file_names[image_index])['file_stem']
            file_extension = utils.input_file_path_info(file_names[image_index])['file_extension']
            if 'gaussian_noise' in augmentation_config:
                gaussian_noise_config = augmentation_config['gaussian_noise']
                mean = gaussian_noise_config['mean'] if gaussian_noise_config.get('mean') else noise_config.mean
                std = gaussian_noise_config['std'] if gaussian_noise_config.get('std') else noise_config.std
                output_images.append([self.gaussian_noise(image.copy(), mean=mean, std=std),
                                      file_stem + '_gaussian_noise_image'+file_extension])
            if 'salt_and_pepper_noise' in augmentation_config:
                salt_and_pepper_noise_config = augmentation_config['salt_and_pepper_noise']
                salt_prob = salt_and_pepper_noise_config['salt_prob'] if salt_and_pepper_noise_config.get(
                    'salt_prob') else noise_config.salt_prob
                pepper_prob = salt_and_pepper_noise_config['pepper_prob'] if salt_and_pepper_noise_config.get(
                    'pepper_prob') else noise_config.pepper_prob
                output_images.append(
                    [self.salt_and_pepper_noise(image.copy(), salt_prob=salt_prob, pepper_prob=pepper_prob),
                     file_stem + '_salt_and_pepper_noise_image'+file_extension])
            if 'speckle_noise' in augmentation_config:
                speckle_noise_config = augmentation_config['speckle_noise']
                scale = speckle_noise_config['scale'] if speckle_noise_config.get(
                    'scale') else noise_config.scale
                output_images.append(
                    [self.speckle_noise(image.copy(), scale=scale), file_stem + '_speckle_noise_image'+file_extension])
            if 'poisson_noise' in augmentation_config:
                output_images.append([self.poisson_noise(image.copy()), file_stem +'_poisson_noise_image'+file_extension])
            if 'periodic_noise' in augmentation_config:
                periodic_noise_config = augmentation_config['periodic_noise']
                frequency = periodic_noise_config['frequency'] if periodic_noise_config.get(
                    'scale') else noise_config.frequency
                output_images.append([self.periodic_noise(image.copy(), frequency=frequency),file_stem + '_periodic_noise_image'+file_extension])
            if 'shadow_noise' in augmentation_config:
                shadow_noise_config = augmentation_config['shadow_noise']
                shadow_prob = np.array(shadow_noise_config['shadow_prob']) if shadow_noise_config.get(
                    'shadow_prob') else noise_config.shadow_prob
                output_images.append([self.shadow_noise(image.copy(), shadow_prob=shadow_prob),
                                      file_stem + '_shadow_noise_image'+file_extension])
        return output_images
