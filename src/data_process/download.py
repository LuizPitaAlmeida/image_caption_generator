"""data_process
Useful classes, method, definitions for image captioning data preprocessing.

Download and Unzip data

All this code is based on the paper:

Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation
with visual attention." International conference on machine learning. 2015.

And Github implementations provided in:

- <https://github.com/ajamjoom/Image-Captions>
- <https://github.com/yunjey/pytorch-tutorial>
- <https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning>

Thanks Abdulrahman Jamjoom, Yunjey Choi, and Sagar Vinodababu
"""
import os


class Download():
    def __init__(self):
        super(Download, self).__init__()

    def load_urls_from_txt_file(self, urls_txt_file):
        with open(os.path.abspath(urls_txt_file)) as file:
            urls = file.read().splitlines()
