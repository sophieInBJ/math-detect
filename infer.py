# coding=utf-8
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

from utils_hs import *

import cv2
import os,sys
import numpy as np
import shutil
import argparse
from tqdm import tqdm 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def get_index(score_list):

    score_array = np.array(score_list)
    ind = np.where(score_array<0.30)
    return ind[0][0]

def dict2list(list_in):
    oral_list = []
    for item in list_in:
        bbox_in = item[0:4]
        label = item[4]
        content = ''
        oral_obj = Gound_truth()
        oral_obj.set(bbox_in,label,content)
        oral_list.append(oral_obj)
    return oral_list, len(oral_list)

def get_pr(boxes, scores, labels, labels_to_names):
    boxes /= scale
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.30:
            break
            # pass
        return labels_to_names[label], box, score
    return 'nothing', [], None 


def draw_pr_image(box, labels_to_names, label, score, img):
    if len(box) == 0:
        return img 
    color = (255, 0, 255)
    b = box.astype(int)
    draw_box(img, b, color=color)
    # print(labels_to_names[label], score, b)
    caption = "{} {:.3f}".format(label, score)
    draw_caption(img, b, caption)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--imgpath')
    parser.add_argument('--minsize', default=800, type=int)
    parser.add_argument('--maxsize', default=1333, type=int)
    arg = parser.parse_args()
    # 初始化模型
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())
    model_path = arg.model
    image_path = arg.imgpath

    labels_to_names = {0:'class1',1:'class2',2:'class3',3:'class4',4:'class5'}
    name_list = [ labels_to_names[k] for k in labels_to_names]

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    print('finish loading model~~~~~~')

    basename = image_path.split('/')[-1].split('.')[0]

    image = read_image_bgr(image_path)
    # preprocess image for network
    image = preprocess_image(image)
    h,w,d = image.shape

    image, scale = resize_image(image, arg.minsize, arg.maxsize)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    pr, box, score  = get_pr(boxes, scores, labels, labels_to_names)

    img_show = cv2.imread(image_path)
    img_show = draw_pr_image(box, labels_to_names, pr, score, img_show)




