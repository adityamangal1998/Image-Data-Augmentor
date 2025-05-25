from ImageDataAugmentor import ImageDataAugmentation as IDA
image_data_augmentor = IDA()
output = image_data_augmentor.data_augmentation(image_dir_path='input',is_save_image=True,output_dir='output')