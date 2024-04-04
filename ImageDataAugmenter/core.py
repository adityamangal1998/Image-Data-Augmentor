import cv2
import numpy as np


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


def blur(image, kernel_size=(15, 15)):
    """
    To Change the blur of image
    :param image: image matrix
    :param kernel_size: kernel size
    :return: blurred image matrix
    """
    try:
        return cv2.GaussianBlur(image.copy(), kernel_size, 0)
    except Exception as e:
        print(f"Error in blur function: {e}")
        return image


def rotate(image, angle=135):
    """
    To Change the blur of image
    :param image: image matrix
    :param kernel_size: kernel size
    :return: blurred image matrix
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
        rotated_matrix = cv2.warpAffine(image.copy(), rotation_matrix, (bound_w, bound_h), borderValue=(255, 255, 255))
        return rotated_matrix
    except Exception as e:
        print(f"Error : {e}")
        return image


def crop(image=None, is_center_crop=True, crop_width=10, crop_height=10, x_start=None, x_end=None, y_start=None,
         y_end=None):
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


# Scaling
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
        print(f"scale_factor : {scale_factor}")
        height, width = image.shape[:2]
        print(f"width : {width} ad height : {height}")
        if scale_factor is not None:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        print(f"new_width : {new_width} ad new_height : {new_height}")
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image
    except Exception as e:
        print(f"Error in scale function : {e}")
        return image


# Flipping
def flipping(image=None, is_horizontal=True, is_vertical=False):
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


# Translation
def translation(image, x_translation=50, y_translation=0):
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


def gaussian_noise(image, mean=0, std=25):
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + gaussian_noise, 0, 255)
    return noisy_image.astype(np.uint8)


def salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = image.copy()

    # Salt noise (white pixels)
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255

    # Pepper noise (black pixels)
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0

    return noisy_image


def speckle_noise(image, scale=0.1):
    speckle_noise = np.random.normal(0, scale, image.shape)
    noisy_image = np.clip(image + image * speckle_noise, 0, 255)
    return noisy_image.astype(np.uint8)


def poisson_noise(image):
    noisy_image = np.random.poisson(image / 255.0) * 255
    return noisy_image.astype(np.uint8)


def periodic_noise(image, frequency=10):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    noise = 128 * np.sin(2 * np.pi * frequency * y / image.shape[0])
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)


def shadow_noise(image, shadow_prob=0.1):
    noisy_image = image.copy()

    # Add shadow noise
    shadow_mask = np.random.rand(*image.shape[:2]) < shadow_prob
    noisy_image[shadow_mask] = np.clip(image[shadow_mask] * 0.5, 0, 255)

    return noisy_image
