import cv2


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
    def crop(image=None, is_center_crop=True, crop_width=10, crop_height=10, x_start=None, x_end=None,
                       y_start=None, y_end=None):
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
                x_start = center_x - crop_width // 2
                y_start = center_y - crop_height // 2
                x_end = center_x + crop_width // 2
                y_end = center_y + crop_height // 2

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
    def flip(image, is_horizontal=True, is_vertical=False):
        try:
            if is_horizontal:
                # Flip the image horizontally
                flipped_image = cv2.flip(image, 1)
            elif is_vertical:
                # Flip the image vertically
                flipped_image = cv2.flip(image, 0)
            else:
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

    def image_augmentor(self, image_data_aug, augmentation_config, file_names):
        input_images = image_data_aug.original_images
        output_images = []
        for image_index, image in enumerate(input_images):
            if 'scale' in augmentation_config:
                scale_config = augmentation_config['saturation']
                new_width = scale_config['new_width'] if scale_config.get('new_width') else config.new_width
                new_height = scale_config['new_height'] if scale_config.get('new_height') else config.new_height
                output_images.append(
                    [self.saturation(image.copy(), saturation_factor=saturation_factor),
                     file_names[image_index] + '_saturated_image'])
            if 'crop' in augmentation_config:
                brightness_config = augmentation_config['brightness']
                brightness_factor = brightness_config['brightness_factor'] if brightness_config.get(
                    'brightness_factor') else config.brightness_factor
                output_images.append(
                    [self.brightness(image.copy(), brightness_factor=brightness_factor),
                     file_names[image_index] + 'brightened_image'])
            if 'horizontal_flip' in augmentation_config:
                blur_config = augmentation_config['blur']
                blur_kernel_size = blur_config['kernel_size'] if blur_config.get(
                    'kernel_size') else config.blur_kernel_size
                output_images.append(
                    [self.blur(image.copy(), kernel_size=blur_kernel_size), file_names[image_index] + 'blurred_image'])
            if 'vertical_flip' in augmentation_config:
                output_images.append([self.contrast(image.copy()), 'contrast_image'])
            if 'rotate' in augmentation_config:
                output_images.append([self.contrast(image.copy()), 'contrast_image'])
            if 'x_translate' in augmentation_config:
                output_images.append([self.contrast(image.copy()), 'contrast_image'])
            if 'y_translate' in augmentation_config:
                output_images.append([self.contrast(image.copy()), 'contrast_image'])
        return output_images