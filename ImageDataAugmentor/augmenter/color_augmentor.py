import cv2
import numpy as np

from ImageDataAugmentor import utils
from ImageDataAugmentor.configs import color_config


class ColorAugmentor:
    def __init__(self):
        pass

    @staticmethod
    def saturation(image, saturation_factor=1.5):
        """
        To Change the saturation of image
        :param image: image matrix
        :param saturation_factor: factor of saturation
        :return: saturated image matrix
        """
        try:
            # Convert the image to the HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Multiply the saturation channel by the saturation factor
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
            # Convert the image back to the BGR color space
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        except Exception as e:
            print(f"Error in saturation function : {e}")
            return image

    @staticmethod
    def brightness(image=None, brightness_factor=1.5):
        """
        To Change the brightness of image
        :param image: image matrix
        :param brightness_factor: factor of brightness
        :return: brightened image matrix
        """
        try:
            # Convert the image to the HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Multiply the brightness channel by the brightness factor
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255)
            # Convert the image back to the BGR color space
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        except Exception as e:
            print(f"Error in brightness function : {e}")
            return image

    @staticmethod
    def blur(image, kernel_size=(5, 5)):
        """
        To Change the blur of image
        :param image: image matrix
        :param kernel_size: kernel size
        :return: blurred image matrix
        """
        print(f"kernel_size : {kernel_size}")
        print(f"image size : {image.shape}")
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(image, kernel_size, 0)

    @staticmethod
    def contrast(image):
        """
        To Change the contrast of image
        :param image: image matrix
        :return: contrasted image matrix
        """
        try:
            gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            return cv2.equalizeHist(gray)
        except Exception as e:
            print(f"Error in contrast function: {e}")
            return image

    @staticmethod
    def sharpness(image, kernel=None):
        """
        To Change the sharpness of image
        :param image: image matrix
        :param kernel: kernel to change sharpness
        :return: sharpened image matrix
        """
        try:
            if kernel is None:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(image.copy(), -1, kernel)
        except Exception as e:
            print(f"Error in sharpness function : {e}")
            return image

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
            if 'saturation' in augmentation_config:
                saturation_config = augmentation_config['saturation']
                saturation_factor = saturation_config['saturation_factor'] if saturation_config.get('saturation_factor') else color_config.saturation_factor
                output_images.append([self.saturation(image.copy(), saturation_factor=saturation_factor), file_stem + '_saturated_image'+file_extension])
            if 'brightness' in augmentation_config:
                brightness_config = augmentation_config['brightness']
                brightness_factor = brightness_config['brightness_factor'] if brightness_config.get('brightness_factor') else color_config.brightness_factor
                output_images.append([self.brightness(image.copy(), brightness_factor=brightness_factor),file_stem + '_brightened_image'+file_extension])
            if 'blur' in augmentation_config:
                blur_config = augmentation_config['blur']
                blur_kernel_size = tuple(blur_config['kernel_size']) if blur_config.get('kernel_size') else color_config.blur_kernel_size
                output_images.append([self.blur(image.copy(), kernel_size=blur_kernel_size), file_stem + '_blurred_image'+file_extension])
            if 'contrast' in augmentation_config:
                output_images.append([self.contrast(image.copy()), file_stem + '_contrast_image'+file_extension])
            if 'sharpness' in augmentation_config:
                sharpness_config = augmentation_config['sharpness']
                sharpness_kernel_size = np.array(sharpness_config['kernel']) if sharpness_config.get('kernel') else color_config.sharpness_kernel
                output_images.append([self.sharpness(image.copy(), kernel=sharpness_kernel_size),file_stem + '_sharp_image'+file_extension])
        return output_images
