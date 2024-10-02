# Image-Data-Augmentor
[![PyPI Status](https://img.shields.io/badge/pypi%20package-1.0.0-orange)](https://pypi.org/project/ImageDataAugmentor/)
[![PyPI Status](https://img.shields.io/github/stars/adityamangal1998/Image-Data-Augmentor)](https://img.shields.io/github/stars/adityamangal1998/Image-Data-Augmentor)
[![PyPI Status](https://img.shields.io/github/license/adityamangal1998/Image-Data-Augmentor)](https://img.shields.io/github/license/adityamangal1998/Image-Data-Augmentor)
<br><br>
<b>Author : Aditya Mangal </b>[![PyPI Status](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aditya-mangal/)


<br>
 <b>Image-Data-Augmentor</b> Image-Data-Augmentor is a Python module designed to enhance the preprocessing pipeline for machine learning models by providing a robust set of image augmentation techniques. 
<br>It focuses on modifying the input image data in various controlled ways, allowing developers and researchers to generate a more diverse dataset, improving the model's ability to generalize and handle unseen data. 
<h1>Installation</h1>

<h2>Dependencies</h2>
<ul>
<li>Python (>= 3.7)</li>
<li>cv2 (>= 4.5)</li>
<li>NumPy (>= 1.17)</li>
<li>glob (>= 0.7)</li>
<li>future (>= 0.18.2)</li>
<li>ConcurrentImageRead (>= 0.0.10)</li>
</ul>

<h1>User installation</h1>
<pre><code>pip install ImageDataAugmenter
</code></pre>

<h1>Usage</h1>
<pre><code>from ImageDataAugmentor import ImageDataAugmentation as IDA

image_data_augmentor = IDA()
output = image_data_augmentor.data_augmentation(image_dir_path='input', configuration_file_path='config.yaml')
</code></pre>

<h1>To Do List</h1>
<ul>
<li>Integrate Data Iterator</li>
<li>Integrate Parallel Processing Pipeline</li>
<li>Add yolo and xml boxes augmenter</li>
<li>Generative Adversarial Networks will be used to generate new samples of images</li>
</ul>



