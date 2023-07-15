# Neural Style Transfer

Neural Style Transfer is a technique that combines the content of one image with the style of another image, creating a new image that retains the content but adopts the style. This repository provides a Python implementation of Neural Style Transfer using TensorFlow and the VGG19 model.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Examples](#examples)
- [Acknowledgements](#acknowledgements)

## Introduction

Neural Style Transfer is a popular technique in the field of deep learning and computer vision. It allows you to transfer the style of one image onto the content of another image. For example, you can take the content of a photograph and apply the style of a famous painting to create a unique artwork.

This implementation uses the VGG19 model, a deep convolutional neural network, to extract high-level features from images. The content and style losses are computed based on the features extracted from the content and style images. The generated image is then optimized using gradient descent to minimize the total loss, resulting in a stylized image.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Numpy
- Matplotlib

## Usage

1. Clone the repository:
git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer

2. Install the dependencies:

# pip install -r requirements.txt

3. Prepare your content and style images by placing them in the `images` directory.

4. Open the `style_transfer.py` file and modify the `content_image_path` and `style_image_path` variables to point to your content and style images, respectively.

5. Run the script:

# python style_transfer.py

6. The generated image will be saved in the `output` directory.

## Examples

Here are some examples of stylized images generated using this implementation:

- Content Image: ![Content Image](images/content.jpg)
Style Image: ![Style Image](images/style.jpg)
Generated Image: ![Generated Image](output/generated.jpg)

- Content Image: ![Content Image](images/content2.jpg)
Style Image: ![Style Image](images/style2.jpg)
Generated Image: ![Generated Image](output/generated2.jpg)

Feel free to experiment with different content and style images to create your own stylized artworks!

## Acknowledgements

This implementation is based on the research paper "A Neural Algorithm of Artistic Style" by Gatys et al. (https://arxiv.org/abs/1508.06576).

The VGG19 model is a pre-trained model available in TensorFlow Keras Applications (https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG19).
