import cv2
import numpy as np


class NoiseAugmentor:
    def __init__(self):
        pass

    @staticmethod
    def gaussian_noise(image, mean=0, std=25):
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

    @staticmethod
    def salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
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

    @staticmethod
    def speckle_noise(image, scale=0.1):
        """
        Speckle Noise
        :param image:
        :param scale:
        :return:
        """
        speckle_noise = np.random.normal(0, scale, image.shape)
        noisy_image = np.clip(image + image * speckle_noise, 0, 255)
        return noisy_image.astype(np.uint8)

    @staticmethod
    def poisson_noise(image):
        """
        Poisson Noise
        :param image:
        :return:
        """
        noisy_image = np.random.poisson(image / 255.0) * 255
        return noisy_image.astype(np.uint8)

    @staticmethod
    def periodic_noise(image, frequency=10):
        """
        Periodic Noise
        :param image:
        :param frequency:
        :return:
        """
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        noise = 128 * np.sin(2 * np.pi * frequency * y / image.shape[0])
        noisy_image = np.clip(image + noise, 0, 255)
        return noisy_image.astype(np.uint8)

    @staticmethod
    def shadow_noise(image, shadow_prob=0.1):
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

    def image_augmentor(self, image_data_aug, augmentation_config, file_names):
        input_images = image_data_aug.original_images
        output_images = []
        for image_index, image in enumerate(input_images):
            if 'gaussian_noise' in augmentation_config:
                saturation_config = augmentation_config['saturation']
                saturation_factor = saturation_config['saturation_factor'] if saturation_config.get(
                    'saturation_factor') else config.saturation_factor
                output_images.append(
                    [self.saturation(image.copy(), saturation_factor=saturation_factor),
                     file_names[image_index] + '_saturated_image'])
            if 'salt_and_pepper_noise' in augmentation_config:
                brightness_config = augmentation_config['brightness']
                brightness_factor = brightness_config['brightness_factor'] if brightness_config.get(
                    'brightness_factor') else config.brightness_factor
                output_images.append(
                    [self.brightness(image.copy(), brightness_factor=brightness_factor),
                     file_names[image_index] + 'brightened_image'])
            if 'speckle_noise' in augmentation_config:
                blur_config = augmentation_config['blur']
                blur_kernel_size = blur_config['kernel_size'] if blur_config.get(
                    'kernel_size') else config.blur_kernel_size
                output_images.append(
                    [self.blur(image.copy(), kernel_size=blur_kernel_size), file_names[image_index] + 'blurred_image'])
            if 'poisson_noise' in augmentation_config:
                output_images.append([self.contrast(image.copy()), 'contrast_image'])
            if 'periodic_noise' in augmentation_config:
                output_images.append([self.contrast(image.copy()), 'contrast_image'])
            if 'shadow_noise' in augmentation_config:
                sharpness_config = augmentation_config['sharpness']
                sharpness_kernel_size = np.array(sharpness_config['kernel']) if sharpness_config.get(
                    'kernel') else config.sharpness_kernel
                output_images.append([self.sharpness(image.copy(), kernel=sharpness_kernel_size),
                                      file_names[image_index] + 'sharp_image'])
        return output_images