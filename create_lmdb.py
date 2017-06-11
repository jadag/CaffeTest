'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''# def create_db(output_file):
#     print(">>> Write database...")
#     LMDB_MAP_SIZE = 1 << 40   # MODIFY
#     env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)
# 
#     checksum = 0
#     with env.begin(write=True) as txn:
#         for j in range(0, 128):
#             # MODIFY: add your own data reader / creator
#             label = j % 10
#             width = 64
#             height = 32
# 
#             img_data = np.random.rand(3, width, height)
#             # ...
# 
#             # Create TensorProtos
#             tensor_protos = caffe2_pb2.TensorProtos()
#             img_tensor = tensor_protos.protos.add()
#             img_tensor.dims.extend(img_data.shape)
#             img_tensor.data_type = 1
# 
#             flatten_img = img_data.reshape(np.prod(img_data.shape))
#             img_tensor.float_data.extend(flatten_img)
# 
#             label_tensor = tensor_protos.protos.add()
#             label_tensor.data_type = 2
#             label_tensor.int32_data.append(label)
#             txn.put(
#                 '{}'.format(j).encode('ascii'),
#                 tensor_protos.SerializeToString()
#             )
# 
#             checksum += np.sum(img_data) * label
#             if (j % 16 == 0):
#                 print("Inserted {} rows".format(j))
# 
#     print("Checksum/write: {}".format(int(checksum)))
#     return checksum
## @package lmdb_create_example
# Module caffe2.python.examples.lmdb_create_example
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import cv2
import lmdb
import os
import pandas as pd
import pickle as pc
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, cnn
IMAGE_WIDTH = 28
IMAGE_HEIGHT =28
'''
Simple example to create an lmdb database of random image data and labels.
This can be used a skeleton to write your own data import.
It also runs a dummy-model with Caffe2 that reads the data and
validates the checksum is same.
'''

img_mean_fn = 'image_mean.txt'
img_std_fn = 'image_std.txt'

INDEX_SKIP = 5


def read_db_with_caffe2(db_file, expected_checksum):
    print(">>> Read database...")
    model = cnn.CNNModelHelper(
        order="NCHW", name="lmdbtest")
    batch_size = 32
    data, label = model.TensorProtosDBInput(
        [], ["data", "label"], batch_size=batch_size,
        db=db_file, db_type="lmdb")

    checksum = 0

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    for batch_idx in range(0, 4):
        workspace.RunNet(model.net.Proto().name)

        img_datas = workspace.FetchBlob("data")
        labels = workspace.FetchBlob("label")
        for j in range(batch_size):
            checksum += np.sum(img_datas[j, :]) * labels[j]

    print("Checksum/read: {}".format(int(checksum)))
    assert np.abs(expected_checksum - checksum < 0.1), \
        "Read/write checksums dont match"
# 
# 
# def main():
#     parser = argparse.ArgumentParser(
#         description="Example LMDB creation"
#     )
#     parser.add_argument("--output_file", type=str, default=None,
#                         help="Path to write the database to",
#                         required=True)
# 
#     args = parser.parse_args()
#     checksum = create_db(args.output_file)
# 
#     # For testing reading:
#     read_db_with_caffe2(args.output_file, checksum)


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    
    img[:, :] = cv2.equalizeHist(img[:, :])
#     img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
#     img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img_data, label):
    # Create TensorProtos
    tensor_protos = caffe2_pb2.TensorProtos()
    img_tensor = tensor_protos.protos.add()
    img_tensor.dims.extend(img_data.shape)
    img_tensor.data_type = 1

    flatten_img = img_data.reshape(np.prod(img_data.shape))
    img_tensor.float_data.extend(flatten_img)

    label_tensor = tensor_protos.protos.add()
    label_tensor.data_type = 2
    label_tensor.int32_data.append(label)
    
    
    return tensor_protos
    


def writeToFile(data, labels,output_file):
#     if not os.path.exists(output_file):
#         os.mkdir(output_file)
    
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)
    
    with env.begin(write=True) as txn:
        for i in range(data.shape[0]):
            tensor_protos = make_datum(data[i], labels[i])
            txn.put('{}'.format(i).encode('ascii'),tensor_protos.SerializeToString())
            
    
    return
def Normalize(images, calc_norm):
    
    multiplier = 10000
    
    if calc_norm:        
        var_im = np.var(images,0)
        std_im = np.sqrt(var_im)*multiplier
        mean_im = np.mean(images,0)
        
        filehandler = open(img_mean_fn, 'w') 
        pc.dump(mean_im, filehandler) 
        filehandler = open(img_std_fn, 'w') 
        pc.dump(std_im, filehandler) 
#         np.savetxt(img_mean_fn, mean_im)
#         np.savetxt(img_std_fn, std_im)
#         pd.DataFrame(mean_im[0]).to_csv(img_mean_fn,header=0, parse_dates=False)
#         pd.DataFrame(std_im[0]).to_csv(img_std_fn,header=0, parse_dates=False)
    else:
        filehandler = open(img_mean_fn, 'r') 
        mean_im =pc.load( filehandler) 
        filehandler = open(img_std_fn, 'r') 
        std_im = pc.load( filehandler) 
#         mean_im = np.load(img_mean_fn) #pd.read_csv(img_mean_fn,header=0, parse_dates=False).as_matrix()
#         std_im=np.load(std_im) #pd.read_csv(img_std_fn).values
        
#     np.divide(images.astype(np.int),std_im.astype(np.int))
    
    image_mean = np.subtract(images,mean_im)*multiplier*multiplier
    images_int = image_mean.astype(np.int)
    std_int = std_im.astype(np.int)
    normalized_images = np.divide(images_int, std_int)
    normalized_images = normalized_images.astype(np.float)/(5*multiplier)
    
    return normalized_images
    

def loadImagesLabels(data, labels):
    
    images = []
    sum_img = []
    
    counter = 0
    labels_per_image =[]
    
    for label in labels:
        for in_idx, img_path in enumerate(data[label]):
            if in_idx % INDEX_SKIP != 0:
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_channel = np.array([img])
#             if sum_img == []:
#                 sum_img =img.astype(np.int64)
#             else:
#                 sum_img += img.astype(np.int64)
#             counter += 1 
            images.append(img_channel)
            labels_per_image.append(label)
            
    images_np = np.array(images)
#     avrg = sum_img/counter #np.mean(images, 1)
    var = np.var(images_np,0)
    
    return images_np , labels_per_image   


def write_to_lmdb(data, output_file, labels, calc_norm = True):
    if not os.path.exists(output_file):
        os.mkdir(output_file)
    
    images_np , labels_img  = loadImagesLabels(data, labels)
    normalized_images = Normalize(images_np,calc_norm)
    writeToFile(normalized_images, labels_img,output_file)
    

def write_to_lmdb2(data, output_file, labels):
    if not os.path.exists(output_file):
        os.mkdir(output_file)
    
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)

    counter = 0
    images_np , labels   = loadImagesLabels(data, labels)
    
    with env.begin(write=True) as txn:
        for label in labels:
            for in_idx, img_path in enumerate(data[label]):
                if in_idx % INDEX_SKIP != 0:
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_channel = np.array([img])
#                 img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
#                 if 'cat' in img_path:
#                     label = 0
#                 else:
#                     label = 1
                tensor_protos = make_datum(img_channel, label)
                txn.put('{}'.format(counter).encode('ascii'),tensor_protos.SerializeToString())
                print( '{:08}'.format(counter).encode('ascii') + ':' + img_path)
                counter+=1
            
    env.close()

