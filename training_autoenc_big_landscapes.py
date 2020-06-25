from keras.models import Model, Input
from keras.models import load_model
from keras.layers import Input as Inp
from keras.layers import Conv2D, Conv2DTranspose, Activation, Concatenate,\
  Dropout, BatchNormalization, LeakyReLU, Lambda, MaxPooling2D, Dense, UpSampling2D
from keras.optimizers import Adam, SGD
from keras.utils import Sequence
from keras import metrics
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import random
from random import randint, sample
import math
from PIL import Image
# from skimage.color import rgb2gray
import zipfile
import base64
import io
import os

# path_landscapes = '/content/drive/My Drive/Licenta/landscapes/'
path_landscapes = 'landscapes/'
path_archives = path_landscapes + 'dataset/big_dataset/all/all/'
# path_archives = 'archives/'
path_mr = path_landscapes + 'models+results/'
# path_cleancode = path_mr + 'cleancode/'
path_autoencoder = path_mr + 'autoencoder/'
path_gan = path_mr + 'gan/'
# print(path_gan)

path_models = path_autoencoder + 'models2/'
path_history = path_autoencoder + 'history2/'
if not os.path.isdir(path_models):
  os.makedirs(path_models)
if not os.path.isdir(path_history):
  os.makedirs(path_history)

"""## **1. Autoencoder approach**"""

# Encode layers
def encode_layers_autoenc(layer_in, downscale, batchnorm):
  if batchnorm:
    layer = BatchNormalization()(layer_in)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(downscale)(layer)
  else:
    layer = Activation('relu')(layer_in)
    layer = MaxPooling2D(downscale)(layer)
  return layer


# Decode layers
def decode_layers_autoenc(layer_in, n_filters, kernel_size, upscale, upscale_method, batchnorm, skip_conn, *to_concat_with):
  if upscale_method == 'Conv2DTranspose':
    deconv = Conv2DTranspose(n_filters, kernel_size, strides=upscale, padding='same')(layer_in)
  elif upscale_method == 'UpSampling2D':
    deconv = UpSampling2D(size=upscale)(layer_in)
    deconv = Conv2D(n_filters, kernel_size, padding='same')(deconv)
  if skip_conn:
    deconv = Concatenate()([deconv, to_concat_with[0]])
  if batchnorm:
    deconv = BatchNormalization()(deconv)
    deconv = Activation('relu')(deconv)
  return deconv


# Autoencoder
def autoencoder(input_shape, skip_conn, conv_batchnorm, deconv_batchnorm, deconv_upscale_method, model_loss):
  inp = Input(input_shape)

  conv1 = Conv2D(64, (3, 3), padding='same')(inp)
  layer_set1 = encode_layers_autoenc(conv1, (2,2), conv_batchnorm)
  conv2 = Conv2D(128, (3, 3), padding='same')(layer_set1)
  layer_set2 = encode_layers_autoenc(conv2, (2,2), conv_batchnorm)
  conv3 = Conv2D(256, (3, 3), padding='same')(layer_set2)
  layer_set3 = encode_layers_autoenc(conv3, (2,2), conv_batchnorm)
  # conv4 = Conv2D(512, (3,3), padding='same')(layer_set3)
  # layer_set4 = encode_layers_autoenc(conv4, (1,1), conv_batchnorm)

  # deconv4 = decode_layers_autoenc(layer_set4, 512, (3,3), (1,1), deconv_upscale_method, deconv_batchnorm, skip_conn, conv4)
  deconv3 = decode_layers_autoenc(layer_set3, 256, (3, 3), (2,2), deconv_upscale_method, deconv_batchnorm, skip_conn, conv3)
  deconv2 = decode_layers_autoenc(deconv3, 128, (3, 3), (2,2), deconv_upscale_method, deconv_batchnorm, skip_conn, conv2)
  deconv1 = decode_layers_autoenc(deconv2, 64, (3, 3), (2,2), deconv_upscale_method, deconv_batchnorm, skip_conn, conv1)

  out = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(deconv1)

  autoenc = Model(inp, out)
  autoenc.summary()
  autoenc.compile(optimizer="adam", loss=model_loss)
  return autoenc

"""*   Create autoencoder models"""

# params: input_shape,  skip_conn,  conv_batchnorm,  deconv_batchnorm, 
# deconv_upscale_method (Conv2DTranspose or UpSampling2D),  model_loss
input_shape = (256, 256, 1)
skip_conn = [True, False]
conv_batchnorm = True
deconv_batchnorm = True
deconv_upscale_method = ['Conv2DTranspose', 'UpSampling2D']
model_loss = 'mse'
autoenc_models = []
for sc in skip_conn:
  for upscale_method in deconv_upscale_method:
    autoenc_models.append(autoencoder(input_shape, sc, conv_batchnorm, 
                                      deconv_batchnorm,upscale_method,
                                      model_loss))
# autoenc_models contain 4 models

"""*   Train autoencoder"""

class CelebASequenceT(Sequence):
  def __init__(self, batch_size, len_dataset, nr_archives, path_archives):
    self.batch_size = int(batch_size)
    self.len_dataset = int(len_dataset)
    self.nr_archives = int(nr_archives)
    self.path_archives = path_archives
    self.nr_batches = int(self.len_dataset / self.batch_size)
    self.nr_batches_per_zip = int(self.nr_batches / self.nr_archives)
    self.nr_im_in_zip = int(self.len_dataset / self.nr_archives)
    self.x_archive = None
    self.y_archive = None
    self.batch_index = 0
    self.buckets = [[i for i in range(1, int(self.nr_im_in_zip+1))] for _ in range(int(self.nr_archives))]


  def __len__(self):
      return int(self.len_dataset)
    

  def __getitem__(self, idx):
    archive_idx = int(self.batch_index / self.nr_batches_per_zip)
    part_of_archive = int(self.batch_index % self.nr_batches_per_zip)
    archive_bucket = self.buckets[archive_idx]

    batch_x = []
    batch_y = []

    if self.batch_index % self.nr_batches_per_zip == 0:
      self.x_archive = np.load(path_archives + 'x_train_{}.npz'.format(int(archive_idx+1)))
      self.x_archive = self.x_archive['x_train']
      self.y_archive = np.load(path_archives + 'y_train_{}.npz'.format(int(archive_idx+1)))
      self.y_archive = self.y_archive['y_train']
      # preprocessing
      self.x_archive = np.asarray(self.x_archive, dtype=np.float64)
      self.x_archive = self.x_archive / 255
      self.y_archive = np.asarray(self.y_archive, dtype=np.float64)
      self.y_archive = self.y_archive / 255

    if self.batch_index == self.nr_batches-1:
      self.batch_index = 0
      for i in range(len(self.buckets)):
        random.shuffle(self.buckets[i])
    
      if self.x_archive is not None and len(self.x_archive.shape) != 4:
        x_train_transformed = []
        for i in range(len(self.x_archive)):
          x_train_transformed.append(np.expand_dims(self.x_archive[i], axis=2))
        self.x_archive = np.array(x_train_transformed)

    slice_archive = archive_bucket[int(part_of_archive * self.batch_size) : int((int(part_of_archive+1)) * self.batch_size)]

    batch_x = []
    for i in range(len(slice_archive)):
        batch_x.append(self.x_archive[slice_archive[i]-1])

    batch_y = []
    for i in range(len(slice_archive)):
        batch_y.append(self.y_archive[slice_archive[i]-1])

    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(batch_y)

    self.batch_index += 1
    # batch 1, index 0 apoi 1
    # batch 1024, index 1023 apoi 
    return batch_x, batch_y

class CelebASequenceV(Sequence):
  def __init__(self, batch_size, x_set, y_set):
      self.batch_size = batch_size
      self.x, self.y = x_set, y_set

  def __len__(self):
      return math.ceil(len(self.x) / self.batch_size)

  def __getitem__(self, idx):
      batch_x = self.x[idx * self.batch_size:(idx + 1) *
      self.batch_size]
      batch_y = self.y[idx * self.batch_size:(idx + 1) *
      self.batch_size]

      return np.array(batch_x), np.array(batch_y)

def path_models_results_autoenc(model_index):
  path = ''
  if model_index == 3:
    path = 'skip_false_Conv2DTranspose/'
  elif model_index == 4:
    path = 'skip_false_UpSampling2D/'
  elif model_index == 1:
    path = 'skip_true_Conv2DTranspose/'
  elif model_index == 2:
    path = 'skip_true_UpSampling2D/'
  path = path_autoencoder + path
  return path


def train_autoenc(train_number, batch_size, nr_epochs, period, nr_archives_train, len_dataset, model_index, path_archives):
  path_to_save = path_models_results_autoenc(model_index)
  autoenc = autoenc_models[model_index-1]
  # os.makedirss
  path_models = path_to_save + 'models' + train_number + '/'
  path_results = path_to_save + 'results' + train_number + '/'
  path_history = path_to_save + 'history' + train_number + '/'

  model_checkpoint = ModelCheckpoint(path_models + 
                                    'weights.{epoch:02d}-{loss:.2f}.h5', 
                                    monitor='val_loss', period=period)
  model_reduce_lr = ReduceLROnPlateau(monitor='val_loss')

  x_val = np.load(path_archives + 'x_val_all.npz')
  x_val = x_val['x_val']/255
  y_val = np.load(path_archives + 'y_val_all.npz')
  y_val = y_val['y_val']/255

  train_celeba_sequence = CelebASequenceT(batch_size, len_dataset, 
                                          nr_archives_train, path_archives)
  val_celeba_sequence = CelebASequenceV(batch_size, x_val, y_val)

  steps_per_epoch = int(len_dataset / batch_size)
  history = autoenc.fit_generator(train_celeba_sequence,
                                  steps_per_epoch=steps_per_epoch, epochs=nr_epochs, 
                                  validation_data=val_celeba_sequence, 
                                  callbacks=[model_reduce_lr, model_checkpoint])
  np.save(path_history + 'history.npy', history.history)

nr_archives_train = 80
batch_size = 32
nr_epochs = 30
period = 5
len_dataset = 81920

model_index = 1  # skip_conn = True, deconv_upscale_method='Conv2DTranspose'
# model_index = 2  # skip_conn = True, deconv_upscale_method='UpSampling2D'
# model_index = 3  # skip_conn = False, deconv_upscale_method='Conv2DTranspose'
# model_index = 4  # skip_conn = False, deconv_upscale_method='UpSampling2D'

train_number = '2'

# path_landscapes = '/content/drive/My Drive/Licenta/portraits_celeba/'
# path_archives = path_landscapes + 'dataset/dataset_arrays_archives/'

train_autoenc(train_number, batch_size, nr_epochs, period, nr_archives_train, len_dataset, model_index, path_archives)

"""## **2. GAN approach**

*   Generator
"""

def encode_layers_gan(prev_layer, nr_filters, kernel_size, stride, batchnorm):
  g = Conv2D(nr_filters, kernel_size, strides=stride, padding='same')(prev_layer)
  if batchnorm:
      g = BatchNormalization()(g, training=True)
  g = LeakyReLU(alpha=0.2)(g)
  return g

def decode_layers_gan(prev_layer, nr_filters, kernel_size, stride, dropout, skip_conn, upsample_method, *to_concat_with):
  if upsample_method == 'Conv2DTranspose':
    g = Conv2DTranspose(nr_filters, kernel_size, strides=stride, padding='same')(prev_layer)
  else:
    g = UpSampling2D(size=stride)(prev_layer)
    g = Conv2D(nr_filters, kernel_size, padding='same')(g)
  g = BatchNormalization()(g, training=True)
  if dropout:
      g = Dropout(0.2)(g, training=True)
  if skip_conn is True:
    g = Concatenate()([g, to_concat_with[0]])
  g = Activation('relu')(g)
  return g

def generator(image_shape, skip_conn, deconv_upsample_method):
    in_image = Input(shape=image_shape)

    e1 = encode_layers_gan(in_image, 64, kernel_size=(3,3), stride=(2,2), batchnorm=False)
    e2 = encode_layers_gan(e1, 128, kernel_size=(3,3), stride=(1,1), batchnorm=False)
    e3 = encode_layers_gan(e2, 128, kernel_size=(1,1), stride=(2,2), batchnorm=True)
    e4 = encode_layers_gan(e3, 256, kernel_size=(3,3), stride=(1,1), batchnorm=True)
    e5 = encode_layers_gan(e4, 256, kernel_size=(1,1), stride=(2,2), batchnorm=True)
    e6 = encode_layers_gan(e5, 512, kernel_size=(3,3), stride=(1,1), batchnorm=True)

    b = Conv2D(512, (4, 4), strides=(1, 1), padding='same')(e6)
    b = Activation('relu')(b)
                           
    d6 = decode_layers_gan(b, 512, (1,1), (1,1), False, skip_conn, deconv_upsample_method, e6)
    d5 = decode_layers_gan(d6, 256, (3,3), (1,1), False, skip_conn, deconv_upsample_method, e5)
    d4 = decode_layers_gan(d5, 256, (1,1), (2,2), True, skip_conn, deconv_upsample_method, e4)
    d3 = decode_layers_gan(d4, 128, (3,3), (1,1), True, skip_conn, deconv_upsample_method, e3)
    d2 = decode_layers_gan(d3, 128, (1,1), (2,2), True, skip_conn, deconv_upsample_method, e2)
    d1 = decode_layers_gan(d2, 64, (3,3), (1,1), True, skip_conn, deconv_upsample_method, e1)

    g = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same')(d1)
    out_image = Activation('tanh')(g)

    model = Model(in_image, out_image)
    return model

"""*   Discriminator (basic)"""

def conv_bn_activ_block(prev_layer, nr_filters, kernel_size, stride, batchnorm=True):
  g = Conv2D(nr_filters, kernel_size, strides=stride, padding='same')(prev_layer)
  if batchnorm:
      g = BatchNormalization()(g, training=True)
  g = LeakyReLU(alpha=0.2)(g)
  return g

def discriminator_basic(image_shape_bw, image_shape_rgb, loss_weights):
  rgb_true_image = Input(shape=image_shape_rgb)
  bw_image = Input(shape=image_shape_bw)
  merged = Concatenate()([rgb_true_image, bw_image])

  l1 = conv_bn_activ_block(merged, 64, (3,3), (2,2), batchnorm=False)
  l2 = conv_bn_activ_block(l1, 128, (1,1), (2,2))
  l3 = conv_bn_activ_block(l2, 256, (3,3), (2,2))
  l4 = conv_bn_activ_block(l3, 512, (1,1), (1,1))
  l5 = conv_bn_activ_block(l4, 512, (3,3), (1,1))
  
   # patch output
  d = Conv2D(1, (1, 1), padding='same')(l5)
  patch_out = Activation('sigmoid')(d)

  model = Model([bw_image, rgb_true_image], patch_out)
  model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=loss_weights)
  return model

image_shape_bw = (256, 256, 1)
image_shape_rgb = (256, 256, 3)
loss_weights = [0.5]
model = discriminator_basic(image_shape_bw, image_shape_rgb, loss_weights)
model.summary()

"""*   Discriminator (convolved)"""

def apply_layers(input_layer, layers):
  l = input_layer
  for layer in layers:
    l = layer(l)
  return l

def discriminator_convolved(image_shape_bw, image_shape_rgb, scale_factor):
  width = image_shape_bw[1]
  height = image_shape_bw[0]
  layers = [Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2)]  
  for i in range(int(np.ceil(np.log2(height/scale_factor))-3)):
    layers += [Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
              BatchNormalization(),
              LeakyReLU(alpha=0.2)]
  layers += [Conv2D(512, (2, 2), strides=(2, 2), padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(1, (1, 1), strides=(2, 2), padding='same')]

  rgb_true_image = Input(shape=image_shape_rgb)
  bw_image = Input(shape=image_shape_bw)
  merged = Concatenate()([rgb_true_image, bw_image])

  image_tensor = []
  row_tensor = []
  
  for i in range(scale_factor):
    row_tensor = []
    for j in range(scale_factor):
      patch = Lambda(lambda x: x[:, 
                                  int(height/scale_factor*i):int(height/scale_factor*(i+1)), 
                                  int(width/scale_factor*j):int(width/scale_factor*(j+1)), 
                                  :])(merged)
      row_tensor.append(apply_layers(patch, layers))        
    image_tensor.append(Concatenate(axis=2)(row_tensor))
  big_patch = Concatenate(axis=1)(image_tensor)
      
  patch_out = Activation('sigmoid')(big_patch)
  model = Model([bw_image, rgb_true_image], patch_out)
  model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
  return model

image_shape_bw = (256, 256, 1)
image_shape_rgb = (256, 256, 3)
scale_factor = 4
width = image_shape_bw[1]
height = image_shape_bw[0]
discr_conv = discriminator_convolved(image_shape_bw, image_shape_rgb, scale_factor=scale_factor)
discr_conv.summary()

def discriminator(discr_type, image_shape_bw, image_shape_rgb, *scale_f):
  if discr_type == 'basic':
    discr = discriminator_basic(image_shape_bw, image_shape_rgb, loss_weights)
  else:
    discr = discriminator_convolved(image_shape_bw, image_shape_rgb, scale_f[0])
  return discr

"""*   GAN"""

def gan(generator, discriminator, image_shape_bw, loss_weights_discr, loss_weights_gan):
  discriminator.trainable = False
  discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=loss_weights_discr)

  input_gen = Input(shape=image_shape_bw)
  output_gen = generator(input_gen)
  output_dis = discriminator([input_gen, output_gen])
  gan = Model(input_gen, [output_dis, output_gen])
  gan.compile(loss=['binary_crossentropy', 'mse'], optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=loss_weights_gan)
  gan.summary()
  return gan

"""*   Create GENERATOR models"""

image_shape = (256, 256, 1)
skip_conn_vals_gen = [True, False]
upsample_methods_gen = ['Conv2DTranspose', 'UpSampling2D']
gener_models = []
for sc in skip_conn_vals_gen:
  for upsample_method in upsample_methods_gen:
    g = generator(image_shape, sc, upsample_method)
    gener_models.append(g)
    g.summary()
print(len(gener_models))

"""*   Create DISCRIMINATOR models"""

image_shape_bw = (256, 256, 1)
image_shape_rgb = (256, 256, 3)
scale_factors = [4, 8]
discr_models = []
discr_models.append(discriminator('basic', image_shape_bw, image_shape_rgb))
for sf in scale_factors:
  d = discriminator('convolved', image_shape_bw, image_shape_rgb, sf)
  discr_models.append(d)
  d.summary()
print(len(discr_models))

"""*   Create GAN models"""

image_shape_bw = (256, 256, 1)
image_shape_rgb = (256, 256, 3)
discr_types = ['basic', 'convolved']
skip_conn_vals = [True, False]
upsample_methods = ['Conv2DTranspose', 'UpSampling2D']
loss_weights_discr_vals = [[0.5]]  # [0.2], [0.8] ?
loss_weights_gan_vals = [[20, 80]]  # [5, 95], [35, 65] ?
scale_factors = [4, 8]

gan_models = []
for gener in gener_models:
  for discr in discr_models:
    g = gan(gener, discr, image_shape_bw,
            loss_weights_discr_vals[0], loss_weights_gan_vals[0])
    gan_models.append(g)
    g.summary()
            
print(len(gan_models))

"""*   Train GAN"""

def path_models_results_gan(model_index):
  model_index = int(model_index)
  path = ''
  if model_index == 11:
    path = '11_skipTrue_conv2D_basic_-_0.5_20,80/'
  elif model_index == 12:
    path = '12_skipTrue_conv2D_convo_4_0.5_20,80/'
  elif model_index == 13:
    path = '13_skipTrue_conv2D_convo_8_0.5_20,80/'
  elif model_index == 21:
    path = '21_skipTrue_upsample2D_basic_-_0.5_20,80/'
  elif model_index == 22:
    path = '22_skipTrue_upsample2D_convo_4_0.5_20,80/'
  elif model_index == 23:
    path = '23_skipTrue_upsample2D_convo_8_0.5_20,80/'
  elif model_index == 31:
    path = '31_skipFalse_conv2D_basic_-_0.5_20,80/'
  elif model_index == 32:
    path = '32_skipFalse_conv2D_convo_4_0.5_20,80/'
  elif model_index == 33:
    path = '33_skipFalse_conv2D_convo_8_0.5_20,80/'
  elif model_index == 41:
    path = '41_skipFalse_upsample2D_basic_-_0.5_20,80/'
  elif model_index == 42:
    path = '42_skipFalse_upsample2D_convo_4_0.5_20,80/'
  elif model_index == 43:
    path = '43_skipFalse_upsample2D_convo_8_0.5_20,80/'
  path = path_gan + path
  return path

p = path_models_results_gan(11)
print(p)

from tqdm import tqdm


def type_of_discr(model_index):
  if model_index == 11 or model_index == 21 or model_index == 31 or model_index == 41:
    discr = 'basic'
  if model_index == 12 or model_index == 22 or model_index == 32 or model_index == 42:
    discr = 'conv4'
  if model_index == 13 or model_index == 23 or model_index == 33 or model_index == 43:
    discr = 'conv8'
  return discr

def discr_output(discr_type):
  if discr_type == 'basic':
    return 27,22
  if discr_type == 'conv4':
    return 4,4
  if discr_type == 'conv8':
    return 8,8


def train_gan(train_number, model_index_gan, path_dataset, len_dataset, nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch):
  path_to_save = path_models_results_gan(model_index_gan)
  path_models = path_to_save + 'models' + train_number + '/'
  path_results = path_to_save + 'results' + train_number + '/'
  path_history = path_to_save + 'history' + train_number + '/'
  history=dict()

  model_index_g = int(model_index_gan / 10)
  model_index_d = int(model_index_gan % 10)
  model_index_gan = len(discr_models) * (model_index_g-1) + model_index_d - 1
  gener = gener_models[int(model_index_g)-1]
  discr = discr_models[int(model_index_d)-1]
  gan = gan_models[int(model_index_gan)]

  nr_im_in_zip = int(len_dataset / nr_archives)
  bat_per_epo = int(len_dataset / batch_size)
  nr_iterations = bat_per_epo * nr_epochs

  losses1_discr = []
  losses2_discr = []
  losses3_gan = []
  
  for i in tqdm(range(nr_iterations)):
    # select bw images
    idx = sample(range(1, nr_archives), 1)[0]
    idx_start = sample(range(0, nr_im_in_zip-1-batch_size), 1)[0]
    idx_stop = idx_start+batch_size
    x_real = np.load(path_dataset + 'x_train_{}.npz'.format(idx))['x_train'][idx_start:idx_stop]/255
    y_real = np.load(path_dataset + 'y_train_{}.npz'.format(idx))['y_train'][idx_start:idx_stop]/255
    y_real_patch = np.ones((len(y_real), dim1_patch, dim2_patch, 1))

    # generate rgb images
    y_fake_g = gener.predict(x_real)
    y_fake_patch = np.zeros((len(y_fake_g), dim1_patch, dim2_patch, 1))

    # train discr with real images
    loss1_discr = discr.train_on_batch([x_real, y_real], y_real_patch)
    # train discr with generated images
    loss2_discr = discr.train_on_batch([x_real, y_fake_g], y_fake_patch)
    # train gener
    loss3_gan, _, _ = gan.train_on_batch(x_real, [y_real_patch, y_real])

    losses1_discr.append(loss1_discr)
    losses2_discr.append(loss2_discr)
    losses3_gan.append(loss3_gan)

    if i % (5*bat_per_epo) == 0:
        discr.save(path_models + 'discr_ep{}.h5'.format(i / (5*bat_per_epo)))
        gener.save(path_models + 'gener_ep{}.h5'.format(i / (5*bat_per_epo)))
        gan.save(path_models + 'gan_ep{}.h5'.format(i / (5*bat_per_epo)))
        history['losses1_discr'] = losses1_discr
        history['losses2_discr'] = losses2_discr
        history['losses3_gan'] = losses3_gan
        np.save(path_history + 'history.npy', history)

    
    if i % (bat_per_epo / 4) == 0:
      # print()
      print('{}. loss1_discr: {}, loss2_discr: {}, loss3_gan: {}'.format(i+1, loss1_discr, loss2_discr, loss3_gan))
    
    if i % bat_per_epo == 0:
      print('End of epoch', i // bat_per_epo)

  # history = dict()
  history['losses1_discr'] = losses1_discr
  history['losses2_discr'] = losses2_discr
  history['losses3_gan'] = losses3_gan
  np.save(path_history + 'history.npy', history.history)

# autoencoder setup
nr_archives_train = 80
batch_size = 32
len_dataset = 81920
train_number = '1'
path_landscapes = '/content/drive/My Drive/Licenta/portraits_celeba/'
# path_landscapes = ''
path_archives = path_landscapes + 'dataset/dataset_arrays_archives_to_send/'


# model_index = 1  # skip_conn = True, deconv_upscale_method='Conv2DTranspose'
# model_index = 2  # skip_conn = True, deconv_upscale_method='UpSampling2D'
# model_index = 3  # skip_conn = False, deconv_upscale_method='Conv2DTranspose'
# model_index = 4  # skip_conn = False, deconv_upscale_method='UpSampling2D'

# 1. autoencoder 1
model_index = 1  # skip_conn = True, deconv_upscale_method='Conv2DTranspose'
train_autoenc(train_number, batch_size, nr_archives_train, len_dataset, model_index, path_archives)

# 2. autoencoder 2
model_index = 2  # skip_conn = True, deconv_upscale_method='UpSampling2D'
train_autoenc(train_number, batch_size, nr_archives_train, len_dataset, model_index, path_archives)

# 3. autoencoder 3
model_index = 3  # skip_conn = False, deconv_upscale_method='Conv2DTranspose'
train_autoenc(train_number, batch_size, nr_archives_train, len_dataset, model_index, path_archives)

# 4. autoencoder 4
model_index = 4  # skip_conn = False, deconv_upscale_method='UpSampling2D'
train_autoenc(train_number, batch_size, nr_archives_train, len_dataset, model_index, path_archives)

# gan setup
# model_index_gan = 11  # generator: skip_conn=True, upsample=conv2D, discriminator: basic, scale_factor = -
# model_index_gan = 12  # generator: skip_conn=True, upsample=conv2D, discriminator: convo, scale_factor = 4
# model_index_gan = 13  # generator: skip_conn=True, upsample=conv2D, discriminator: convo, scale_factor = 8
# model_index_gan = 21  # generator: skip_conn=True, upsample=upsample2D, discriminator: basic, scale_factor = -
# model_index_gan = 22  # generator: skip_conn=True, upsample=upsample2D, discriminator: convo, scale_factor = 4
# model_index_gan = 23  # generator: skip_conn=True, upsample=upsample2D, discriminator: convo, scale_factor = 8
# model_index_gan = 31  # generator: skip_conn=False, upsample=conv2D, discriminator: basic, scale_factor = -
# model_index_gan = 32  # generator: skip_conn=False, upsample=conv2D, discriminator: convo, scale_factor = 4
# model_index_gan = 33  # generator: skip_conn=False, upsample=conv2D, discriminator: convo, scale_factor = 8
# model_index_gan = 41  # generator: skip_conn=False, upsample=upsample2D, discriminator: basic, scale_factor = -
# model_index_gan = 42  # generator: skip_conn=False, upsample=upsample2D, discriminator: convo, scale_factor = 4
# model_index_gan = 43  # generator: skip_conn=False, upsample=upsample2D, discriminator: convo, scale_factor = 8

def get_scale_factor(model_index):
  if model_index == 12 or model_index == 22 or model_index == 32 or model_index == 42:
    sf = 4
  if model_index == 13 or model_index == 23 or model_index == 33 or model_index == 43:
    sf = 8
  return sf

train_number = '1'
# path_landscapes = '/content/drive/My Drive/Licenta/portraits_celeba/'
# path_archives = path_landscapes + 'dataset/dataset_arrays_archives_to_send/'
path_landscapes = ''
path_archives = path_landscapes + 'archives/'
len_dataset = 81920
nr_archives = 80
nr_epochs = 100
batch_size = 32
height = 256
width = 256

def type_of_discr(model_index):
  if model_index == 11 or model_index == 21 or model_index == 31 or model_index == 41:
    discr = 'basic'
  if model_index == 12 or model_index == 22 or model_index == 32 or model_index == 42:
    discr = 'conv4'
  if model_index == 13 or model_index == 23 or model_index == 33 or model_index == 43:
    discr = 'conv8'
  return discr

def discr_output(discr_type):
  if discr_type == 'basic':
    return 32,32
  if discr_type == 'conv4':
    return 4,4
  if discr_type == 'conv8':
    return 8,8

# 5. gan 11
model_index_gan = 11  # generator: skip_conn=True, upsample=conv2D, discriminator: basic, scale_factor = -
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

# 6. gan 12
model_index_gan = 12  # generator: skip_conn=True, upsample=conv2D, discriminator: convo, scale_factor = 4
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

# 12. gan 32
model_index_gan = 32  # generator: skip_conn=False, upsample=conv2D, discriminator: convo, scale_factor = 4
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

os.makedirs(path_cleancode + 'gan/13_skipTrue_conv2D_convo_8_0.5_20,80/models1')
os.makedirs(path_cleancode + 'gan/13_skipTrue_conv2D_convo_8_0.5_20,80/history1')
os.makedirs(path_cleancode + 'gan/21_skipTrue_upsample2D_basic_-_0.5_20,80/models1')
os.makedirs(path_cleancode + 'gan/21_skipTrue_upsample2D_basic_-_0.5_20,80/history1')
os.makedirs(path_cleancode + 'gan/22_skipTrue_upsample2D_convo_4_0.5_20,80/models1')
os.makedirs(path_cleancode + 'gan/22_skipTrue_upsample2D_convo_4_0.5_20,80/history1')
os.makedirs(path_cleancode + 'gan/23_skipTrue_upsample2D_convo_8_0.5_20,80/models1')
os.makedirs(path_cleancode + 'gan/23_skipTrue_upsample2D_convo_8_0.5_20,80/history1')
os.makedirs(path_cleancode + 'gan/31_skipFalse_conv2D_basic_-_0.5_20,80/models1')
os.makedirs(path_cleancode + 'gan/31_skipFalse_conv2D_basic_-_0.5_20,80/history1')
os.makedirs(path_cleancode + 'gan/33_skipFalse_conv2D_convo_8_0.5_20,80/models1')
os.makedirs(path_cleancode + 'gan/33_skipFalse_conv2D_convo_8_0.5_20,80/history1')
os.makedirs(path_cleancode + 'gan/41_skipFalse_upsample2D_basic_-_0.5_20,80/models1')
os.makedirs(path_cleancode + 'gan/41_skipFalse_upsample2D_basic_-_0.5_20,80/history1')
os.makedirs(path_cleancode + 'gan/42_skipFalse_upsample2D_convo_4_0.5_20,80/models1')
os.makedirs(path_cleancode + 'gan/42_skipFalse_upsample2D_convo_4_0.5_20,80/history1')
os.makedirs(path_cleancode + 'gan/43_skipFalse_upsample2D_convo_8_0.5_20,80/models1')
os.makedirs(path_cleancode + 'gan/43_skipFalse_upsample2D_convo_8_0.5_20,80/history1')

# 7. gan 13
model_index_gan = 13  # generator: skip_conn=True, upsample=conv2D, discriminator: convo, scale_factor = 8
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

# 8. gan 21
model_index_gan = 21  # generator: skip_conn=True, upsample=upsample2D, discriminator: basic, scale_factor = -
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

# 9. gan 22
model_index_gan = 22  # generator: skip_conn=True, upsample=upsample2D, discriminator: convo, scale_factor = 4
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

# 10. gan 23
model_index_gan = 23  # generator: skip_conn=True, upsample=upsample2D, discriminator: convo, scale_factor = 8
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

# 11. gan 31
model_index_gan = 31  # generator: skip_conn=False, upsample=conv2D, discriminator: basic, scale_factor = -
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

# 13. gan 33
model_index_gan = 33  # generator: skip_conn=False, upsample=conv2D, discriminator: convo, scale_factor = 8
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

# 14. gan 41
model_index_gan = 41  # generator: skip_conn=False, upsample=upsample2D, discriminator: basic, scale_factor = -
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)

# 15. gan 42
model_index_gan = 42  # generator: skip_conn=False, upsample=upsample2D, discriminator: convo, scale_factor = 4
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)



# 16. gan 43
model_index_gan = 43  # generator: skip_conn=False, upsample=upsample2D, discriminator: convo, scale_factor = 8
(dim1_patch, dim2_patch) = discr_output(type_of_discr(model_index_gan))
train_gan(train_number, model_index_gan, path_archives, len_dataset,
          nr_archives, nr_epochs, batch_size, dim1_patch, dim2_patch)