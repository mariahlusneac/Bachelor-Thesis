import numpy as np
from PIL import Image
import os

folder_path_in = 'flickr_photos4'
folder_path_out = r'D:\Faculta\Licenta\conv\autoencoder\flickr_photos4_resized_rgb'


def resize_and_save(img, width, height, save_to):
    img_resized = img.resize((width, height), Image.NEAREST)
    img_resized.save(save_to)


# def turn_bw():
#     obs = Image.fromarray(observation.astype(np.uint8))
#     obs_resized = obs.resize((80, 105), Image.NEAREST).crop((0, 0, 80, 86))
#     obs_array = np.asarray(obs_resized)
#     obs_array = rgb2gray(obs_array)


width_new = 256
height_new = 256

for i, path in enumerate(os.listdir(folder_path_in)):
    path = os.path.join(folder_path_in, path)
    img = Image.open(path)
    width, height = img.size
    if width < height:
        x1 = 0; y1 = height-width
        x2 = width; y2 = height
        cropped = img.crop((x1, y1, x2, y2))
        resize_and_save(cropped, width_new, height_new, f'{folder_path_out}\{i}.jpg')
    elif width > height:
        x1 = (width-height) / 2; y1 = 0
        x2 = x1 + height; y2 = height
        cropped = img.crop((x1, y1, x2, y2))
        resize_and_save(cropped, width_new, height_new, f'{folder_path_out}\{i}.jpg')
    elif width == height:
        resize_and_save(img, width_new, height_new, f'{folder_path_out}\{i}.jpg')