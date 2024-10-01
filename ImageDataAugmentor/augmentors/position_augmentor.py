import cv2
import numpy as np

from ImageDataAugmentor import utils
from ImageDataAugmentor.configs import position_config


class PositionAugmentor:
    def __init__(self):
        pass

    @staticmethod
    def scale(image=None, new_width=300, new_height=300, scale_factor=None):
        """
        To Change the scale of image
        :param image: image matrix
        :param new_width: new width
        :param new_height: new height
        :param scale_factor: scale factor
        :return: saturated image matrix
        """
        try:
            height, width = image.shape[:2]
            if scale_factor is not None:
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            return resized_image
        except Exception as e:
            print(f"Error in scale function : {e}")
            return image

    @staticmethod
    def crop(image=None, is_center_crop=True, crop_width=None, crop_height=None, x_start=None, x_end=None,
             y_start=None, y_end=None):
        """

        :param image:
        :param is_center_crop:
        :param crop_width:
        :param crop_height:
        :param x_start:
        :param x_end:
        :param y_start:
        :param y_end:
        :return:
        """
        try:
            # Get the dimensions of the image
            height, width = image.shape[:2]
            if x_start is None:
                x_start = 0
            if y_start is None:
                y_start = 0
            if x_end is None:
                x_end = width // 2
            if y_end is None:
                y_end = height // 2
            if is_center_crop:
                # Calculate the center of the image
                center_x, center_y = width // 2, height // 2
                # Calculate the region of interest (ROI) for center cropping
                if crop_width is None:
                    crop_width = width // 4
                if crop_height is None:
                    crop_height = height//4
                print(f"crop_width : {crop_width} and crop_height : {crop_height}")
                x_start = center_x - crop_width
                y_start = center_y - crop_height
                x_end = center_x + crop_width
                y_end = center_y + crop_height

                # Crop the image using array slicing
                center_cropped_image = image[y_start:y_end, x_start:x_end]
            else:
                # Crop the image using array slicing
                center_cropped_image = image[y_start:y_end, x_start:x_end]
            return center_cropped_image
        except Exception as e:
            print(f"Error : {e}")
            return image

    @staticmethod
    def flip(image, is_horizontal=True, is_vertical=False, is_vertical_and_horizontal_flip=False):
        try:
            flipped_image = image.copy()
            if is_horizontal:
                # Flip the image horizontally
                flipped_image = cv2.flip(image, 1)
            elif is_vertical:
                # Flip the image vertically
                flipped_image = cv2.flip(image, 0)
            elif is_vertical_and_horizontal_flip:
                # Flip the image both horizontally and vertically
                flipped_image = cv2.flip(image, -1)
            return flipped_image
        except Exception as e:
            print(f"Error : {e}")
            return image

    @staticmethod
    def rotate(image, angle=0):
        """
        To Change the blur of image
        :param image: image matrix
        :param angle: kernel size
        :return: rotate image matrix
        """
        try:
            height, width = image.shape[:2]
            img_c = (width / 2, height / 2)  # Image Center Coordinates

            rotation_matrix = cv2.getRotationMatrix2D(img_c, angle, 1.)  # Rotating Image along the actual center

            abs_cos = abs(rotation_matrix[0, 0])  # Cos(angle)
            abs_sin = abs(rotation_matrix[0, 1])  # sin(angle)

            # New Width and Height of Image after rotation
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            # subtract the old image center and add the new center coordinates
            rotation_matrix[0, 2] += bound_w / 2 - img_c[0]
            rotation_matrix[1, 2] += bound_h / 2 - img_c[1]

            # rotating image with transformed matrix and new center coordinates
            rotated_matrix = cv2.warpAffine(image.copy(), rotation_matrix, (bound_w, bound_h),
                                            borderValue=(255, 255, 255))
            return rotated_matrix
        except Exception as e:
            print(f"Error : {e}")
            return image

    @staticmethod
    def translate(image, x_translation=50, y_translation=0):
        """
        Translate
        :param image:
        :param x_translation:
        :param y_translation:
        :return:
        """
        try:
            # Get the dimensions of the image
            height, width = image.shape[:2]
            if x_translation != 0:
                # Create separate translation matrices for x and y axes
                translation_matrix_x = np.float32([[1, 0, x_translation], [0, 1, 0]])
                image = cv2.warpAffine(image, translation_matrix_x, (width, height))
            if y_translation != 0:
                translation_matrix_y = np.float32([[1, 0, 0], [0, 1, y_translation]])
                image = cv2.warpAffine(image, translation_matrix_y, (width, height))
            return image
        except Exception as e:
            print(f"Error : {e}")
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
            if 'scale' in augmentation_config:
                scale_config = augmentation_config['scale']
                new_width = scale_config['new_width'] if scale_config.get('new_width') else position_config.new_width
                new_height = scale_config['new_height'] if scale_config.get(
                    'new_height') else position_config.new_height
                scale_factor = scale_config['scale_factor'] if scale_config.get(
                    'scale_factor') else position_config.scale_factor
                output_images.append(
                    [self.scale(image.copy(), new_width=new_width, new_height=new_height, scale_factor=scale_factor),
                     file_stem + '_scale_image'+file_extension])
            if 'crop' in augmentation_config:
                crop_config = augmentation_config['crop']
                is_center_crop = crop_config['is_center_crop'] if crop_config.get('is_center_crop') else position_config.is_center_crop
                crop_width = crop_config['crop_width'] if crop_config.get('crop_width') else position_config.crop_width
                crop_height = crop_config['crop_height'] if crop_config.get('crop_height') else position_config.crop_height
                x_start = crop_config['x_start'] if crop_config.get('x_start') else position_config.x_start
                x_end = crop_config['x_end'] if crop_config.get('x_end') else position_config.x_end
                y_start = crop_config['y_start'] if crop_config.get('y_start') else position_config.y_start
                y_end = crop_config['y_end'] if crop_config.get('y_end') else position_config.y_end
                output_images.append([self.crop(image.copy(),is_center_crop=is_center_crop,crop_width=crop_width,
                                                crop_height=crop_height,
                                                x_start=x_start,x_end=x_end,y_start=y_start,y_end=y_end),
                                      file_stem + '_crop_image'+file_extension])
            if 'horizontal_flip' in augmentation_config:
                output_images.append([self.flip(image.copy(), is_horizontal=True,is_vertical=False,
                                                is_vertical_and_horizontal_flip=False),
                                      file_stem + '_horizontal_flip_image'+file_extension])
            if 'vertical_flip' in augmentation_config:
                output_images.append([self.flip(image.copy(), is_horizontal=False,is_vertical=True,
                                                is_vertical_and_horizontal_flip=False),
                                      file_stem + '_vertical_flip_image'+file_extension])
            if 'vertical_and_horizontal_flip' in augmentation_config:
                output_images.append([self.flip(image.copy(), is_horizontal=False,is_vertical=False,
                                                is_vertical_and_horizontal_flip=True),
                                      file_stem + '_vertical_and_horizontal_flip_image'+file_extension])
            if 'rotate' in augmentation_config:
                rotate_config = augmentation_config['rotate']
                angle = rotate_config['angle'] if rotate_config.get('angle') else position_config.angle
                output_images.append([self.rotate(image.copy(),angle=angle), file_stem +'_rotate_image'+file_extension])
            if 'x_translate' in augmentation_config:
                x_translate_config = augmentation_config['x_translate']
                x_translate_value = x_translate_config['x_translate_value'] if x_translate_config.get('x_translate_value') else position_config.x_translate_value
                output_images.append([self.translate(image.copy(),x_translation=x_translate_value,y_translation=0),file_stem + '_x_translate_image'+file_extension])
            if 'y_translate' in augmentation_config:
                y_translate_config = augmentation_config['y_translate']
                y_translate_value = y_translate_config['y_translate_value'] if y_translate_config.get(
                    'y_translate_value') else position_config.y_translate_value
                output_images.append([self.translate(image.copy(), x_translation=0, y_translation=y_translate_value),
                                      file_stem +'_y_translate_image'+file_extension])
            if 'xy_translate' in augmentation_config:
                xy_translate_config = augmentation_config['xy_translate']
                y_translate_value = xy_translate_config['y_translate_value'] if xy_translate_config.get(
                    'y_translate_value') else position_config.y_translate_value
                x_translate_value = xy_translate_config['x_translate_value'] if xy_translate_config.get(
                    'x_translate_value') else position_config.x_translate_value
                output_images.append([self.translate(image.copy(), x_translation=x_translate_value, y_translation=y_translate_value),
                                      file_stem +'_xy_translate_image'+file_extension])
        return output_images
