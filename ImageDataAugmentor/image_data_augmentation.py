import ConcurrentImageRead as CIR


class ImageDataAugmentation:
    def __init__(self):
        self.original_images = []
        self.position_augmented_images = []
        self.color_augmented_images = []
        self.noise_augmented_images = []
        self.augmented_images = []

    def read_images(self, image_list, num_threads=3, channel_type='BGR', root_path='data'):
        self.original_images = CIR.read(image_list, num_threads=num_threads, channel_type=channel_type,
                                        root_path=root_path)
        return self.original_images

    def read_images_from_dir(self, dir_path, file_type='png', num_threads=3, channel_type='BGR', sub_dir=False):
        self.original_images = CIR.read_dir(dir_path, file_type=file_type, num_threads=num_threads,
                                            channel_type=channel_type,
                                            sub_dir=sub_dir)
        return self.original_images

    # def data_augmentation(self, num_threads=3, scale_factor=None,saturation_factor=1.5,brightness_factor=1.5):
    #         """Complete Data Augmentation"""
    #         self.augmented_images.extend(self.position_augmentor(scale_factor=scale_factor))
    #         self.augmented_images.extend(self.color_augmentor(saturation_factor=saturation_factor,brightness_factor=brightness_factor))
    #         # self.augmented_images.extend(self.noise_augmentor())
    #         return self.augmented_images
    #
    def position_augmentor(self, num_threads=3, scale_factor=None):
        """Scaling,Cropping,Flipping,Rotation,Translation"""
        PositionAugmentor.scale(self.original_images)
        self.position_augmented_images.extend(self.scale_augmentor(scale_factor=scale_factor))
        # self.position_augmented_images.extend(core.crop())
        # self.position_augmented_images.extend(core.flipping())
        # self.position_augmented_images.extend(core.rotate())
        # self.position_augmented_images.extend(core.translation())
        return self.position_augmented_images

    def color_augmentor(self, num_threads=3, saturation_factor=1.5, brightness_factor=1.5):
        """Brightness,Contrast,Saturation,Hue,Salt And Pepper"""
        self.color_augmented_images.extend(self.saturation_augmentor(saturation_factor=saturation_factor))
        self.color_augmented_images.extend(self.brightness_augmentor(brightness_factor=brightness_factor))
        # self.color_augmented_images.extend(core.blur())
        # self.color_augmented_images.extend(core.contrast())
        # self.color_augmented_images.extend(core.sharpen())
        return self.color_augmented_images

    def noise_augmentor(self, num_threads=3):
        """Gaussian,Salt and Pepper,Speckle,Poisson,Periodic, Shadow"""
        self.noise_augmented_images.extend(core.gaussian_noise())
        self.noise_augmented_images.extend(core.salt_and_pepper_noise())
        self.noise_augmented_images.extend(core.speckle_noise())
        self.noise_augmented_images.extend(core.poisson_noise())
        self.noise_augmented_images.extend(core.periodic_noise())
        self.noise_augmented_images.extend(core.shadow_noise())
        return self.noise_augmented_images
