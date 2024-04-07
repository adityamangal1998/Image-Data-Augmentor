import cv2


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
    def sharpness(image):
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
