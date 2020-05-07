'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import numpy as np
import argparse
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model,Sequential

#imports for dataloader
from data_loader import Echo
import tensorflow as tf
import tqdm

tf.enable_eager_execution()
from i3d_inception import Inception_Inflated3d

NUM_FRAMES = 79 #changing this can help
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 2

def main(args):
    if args.eval_type in ['rgb', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for RGB data
            # and load pretrained weights (trained on kinetics dataset only) 
            rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='i3d_inception_rgb_kinetics_only_no_top',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for RGB data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)



    return rgb_model


if __name__ == '__main__':

    class args:
        eval_type = 'rgb'
        no_imagenet_pretrained = False
        train=True
        # Namespace(eval_type='rgb', no_imagenet_pretrained=False, train='True')

    # mean, std = get_mean_and_std(Echo(root="delipynb/a4c-video-dir/",split="train"))
    kwargs = {"target_type": 'EF',
          "length": NUM_FRAMES,
          "period": 1,
          }
    train_dataset = Echo(root="a4c-video-dir/",split="train", **kwargs)
    train_dataloader = tf.data.Dataset.from_generator(train_dataset, output_types=(tf.float32, tf.float32)).\
        shuffle(buffer_size=32).batch(1)
    actual_vals = []

    model = main(args)
    loss_func = tf.keras.losses.mse
    optim = tf.keras.optimizers.Adam()
    total = 0
    correct = 0
    
    
    model_2 = Sequential(model)
    model_2.add(Flatten())
    model_2.add(Dense(2))
    model_2.compile(optim,loss_func)
    # print(args)
    for X,EF_val in train_dataloader:
        X = tf.transpose(X,perm = [0,2,3,4,1])
        if EF_val.numpy()>75:
            continue
        else:
            total +=1
            if EF_val.numpy() < 50:
                out = np.array([[0.0,1.0]])
            else:
                out = np.array([[1.0,0.0]]) 

        model_2.train_on_batch(X,out)