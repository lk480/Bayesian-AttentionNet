import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook


def sorted_paths(image_filenames, mask_filenames, image_dir, mask_dir):
  image_paths = []
  mask_paths = []
  for image_name in image_filenames:
      for mask_name in mask_filenames:
          if image_name[7:12] == mask_name[7:12]:
              image_paths.append(os.path.join(image_dir, image_name))
              mask_paths.append(os.path.join(mask_dir, mask_name))
  assert len(image_paths) == len(mask_paths)
  return image_paths, mask_paths

def load_data(image_path, mask_path, size=(128,128)):
  #Load Image
  image = tf.io.read_file(image_path)
  image = tf.image.decode_png(image, 1)
  image = tf.image.resize(image, size)
  image = tf.cast(image, tf.float32) / 255.0

  #Load Mask
  mask = tf.io.read_file(mask_path)
  mask = tf.image.decode_png(mask, 1)
  mask = tf.image.resize(mask, size)
  mask = tf.cast(mask, tf.float32) / 255.0
  return image, mask

def data_visualiser(image):
  image_np = image.numpy()
  print(f"Array Shape is {np.shape(image_np)}")
  plt.imshow(image_np)
  plt.axis('off')
  plt.show()

def train_test_split(sorted_image_paths, sorted_mask_paths, split_ratio=0.2):
  assert len(sorted_image_paths) == len(sorted_mask_paths)
  sorted_paths = list(zip(sorted_image_paths, sorted_mask_paths))
  split_threshold = int((1 - split_ratio) * len(sorted_paths))
  random.shuffle(sorted_paths)
  train_data_paths = sorted_paths[:split_threshold]
  test_data_paths = sorted_paths[split_threshold+1:]
  train_image_paths, train_mask_paths = zip(*train_data_paths)
  test_image_paths, test_mask_paths = zip(*test_data_paths)
  #Convert to lists
  train_image_paths = list(train_image_paths)
  train_mask_paths = list(train_mask_paths)
  test_image_paths = list(test_image_paths)
  test_mask_paths = list(test_mask_paths)
  return train_image_paths, train_mask_paths, test_image_paths, test_mask_paths
