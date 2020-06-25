import zipfile
from PIL import Image
import numpy as np
import os

path_dataset = r'D:\Faculta\Licenta\data_sets\Landscapes\256x256'
path_y_train = path_dataset + r'\y_train\\'
path_y_test = path_dataset + r'\y_test\\'
path_y_val = path_dataset + r'\y_val_all_small\\'
path_big_dataset = path_dataset + r'\big_dataset'
path_big_dataset_test = path_big_dataset + r'\test'
path_big_dataset_train = path_big_dataset + r'\train'
path_big_dataset_val = path_big_dataset + r'\val'

arr = np.load(r'D:\Faculta\Licenta\data_sets\Landscapes\256x256\big_dataset\test\x_test_1.npz')['x_test']
for a in arr:
    print(a.shape)
print('nu')

idx_arch = 1
list_arr_rgb = []
list_arr_bw = []
for j, folder in enumerate(os.listdir(path_y_test)):
    list_arr_rgb = []
    list_arr_bw = []
    for i, file in enumerate(os.listdir(path_y_test + folder)):
        path_im = path_y_test + folder + r'\\' + file
        im_rgb = Image.open(path_im)
        im_bw = im_rgb.convert('L')
        arr_rgb = np.array(im_rgb)
        arr_bw = np.array(im_bw)
        arr_bw = np.expand_dims(arr_bw, axis=2)
        list_arr_rgb.append(arr_rgb)
        list_arr_bw.append(arr_bw)
        if (i+1) % 1024 == 0 and i != 0:
            print(j, 'da', len(list_arr_rgb), len(list_arr_bw))
            list_arr_rgb = np.array(list_arr_rgb)
            list_arr_bw = np.array(list_arr_bw)
            path_rgb = path_big_dataset_test + r'\y_test_{}'.format(idx_arch)
            path_bw = path_big_dataset_test + r'\x_test_{}'.format(idx_arch)
            np.savez(path_rgb, y_test=list_arr_rgb)
            np.savez(path_bw, x_test=list_arr_bw)
            list_arr_rgb = []
            list_arr_bw = []
            idx_arch += 1

print('over 1')

idx_arch = 1
list_arr_rgb = []
list_arr_bw = []
for j, folder in enumerate(os.listdir(path_y_train)):
    list_arr_rgb = []
    list_arr_bw = []
    for i, file in enumerate(os.listdir(path_y_train + folder)):
        path_im = path_y_train + folder + r'\\' + file
        im_rgb = Image.open(path_im)
        im_bw = im_rgb.convert('L')
        arr_rgb = np.array(im_rgb)
        arr_bw = np.array(im_bw)
        arr_bw = np.expand_dims(arr_bw, axis=2)
        list_arr_rgb.append(arr_rgb)
        list_arr_bw.append(arr_bw)
        if (i+1) % 1024 == 0 and i != 0:
            print(j, 'daaa', len(list_arr_rgb), len(list_arr_bw))
            list_arr_rgb = np.array(list_arr_rgb)
            list_arr_bw = np.array(list_arr_bw)
            path_rgb = path_big_dataset_train + r'\y_train_{}'.format(idx_arch)
            path_bw = path_big_dataset_train + r'\x_train_{}'.format(idx_arch)
            np.savez(path_rgb, y_train=list_arr_rgb)
            np.savez(path_bw, x_train=list_arr_bw)
            list_arr_rgb = []
            list_arr_bw = []
            idx_arch += 1

print('over 2')

idx_arch = 1
list_arr_rgb = []
list_arr_bw = []
# for j, folder in enumerate(os.listdir(path_y_val)):
#     list_arr_rgb = []
#     list_arr_bw = []
for i, file in enumerate(os.listdir(path_y_val)):
    path_im = path_y_val + r'\\' + file
    im_rgb = Image.open(path_im)
    im_bw = im_rgb.convert('L')
    arr_rgb = np.array(im_rgb)
    arr_bw = np.array(im_bw)
    arr_bw = np.expand_dims(arr_bw, axis=2)
    list_arr_rgb.append(arr_rgb)
    list_arr_bw.append(arr_bw)
    # if (i+1) % 1024 == 0 and i != 0:
    print(i, len(list_arr_rgb), len(list_arr_bw))


    # list_arr_rgb = []
    # list_arr_bw = []
    # idx_arch += 1

list_arr_rgb = np.array(list_arr_rgb)
list_arr_bw = np.array(list_arr_bw)
path_rgb = path_big_dataset_val + r'\y_val_all_small'
path_bw = path_big_dataset_val + r'\x_val_all_small'
np.savez(path_rgb, y_val=list_arr_rgb)
np.savez(path_bw, x_val=list_arr_bw)
print('over 3')
