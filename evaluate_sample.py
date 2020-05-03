'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import numpy as np
import argparse
from tensorflow.keras.layers import Dense
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

NUM_CLASSES = 400 

def main(args,to_predict):
    if args.eval_type in ['rgb', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for RGB data
            # and load pretrained weights (trained on kinetics dataset only) 
            rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_kinetics_only',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for RGB data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)


    # load RGB sample (just one example)
    # rgb_sample = np.load(SAMPLE_DATA_PATH['rgb'])
    # import ipdb;ipdb.set_trace()
    # make prediction

    rgb_logits = rgb_model.predict(to_predict)


    if args.eval_type in ['flow', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for optical flow data
            # and load pretrained weights (trained on kinetics dataset only)
            flow_model = Inception_Inflated3d(
                include_top=True,
                weights='flow_kinetics_only',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for optical flow data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            flow_model = Inception_Inflated3d(
                include_top=True,
                weights='flow_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)


    # load flow sample (just one example)

    # flow_sample = np.load(SAMPLE_DATA_PATH['flow'])
    
    # make prediction
    # flow_logits = flow_model.predict(to_predict)


    return rgb_logits


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-type', 
        help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).', 
        type=str, choices=['rgb', 'flow', 'joint'], default='rgb')

    parser.add_argument('--no-imagenet-pretrained',
        help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
        action='store_true')

    parser.add_argument("--train",default = False)


    args = parser.parse_args()
    # mean, std = get_mean_and_std(Echo(root="delipynb/a4c-video-dir/",split="train"))
    kwargs = {"target_type": 'EF',
          "length": NUM_FRAMES,
          "period": 1,
          }
    train_dataset = Echo(root="a4c-video-dir/",split="train", **kwargs)
    train_dataloader = tf.data.Dataset.from_generator(train_dataset, output_types=(tf.float32, tf.float32)).\
        shuffle(buffer_size=32).batch(1)
    actual_vals = []

    #model for classification only (not the proper way, but need to use pretrained weights directly)
    model_2 = Sequential()
    model_2.add(Dense(100,input_shape=(400,)))
    model_2.add(Dense(2,activation = "sigmoid"))
    loss_func = tf.keras.losses.binary_crossentropy
    optim = tf.keras.optimizers.Adam()
    total = 0
    correct = 0
    for X,EF_val in train_dataloader:
        if EF_val.numpy()>75:
            continue
        else:
            total +=1
            if EF_val.numpy() < 50:
                out = np.array([1.0,0.0])
            else:
                out = np.array([0.0,1.0]) 
            # actual_vals.append(act)
            # print("before feed shape ::", X.numpy().shape)
            feed = X.numpy().transpose(0,2,3,4,1)
            out_1 = main(args,feed)
            out_2 = model_2(out_1)
            if args.train:
                with tf.GradientTape() as tape:
                    # out_2 = model_2(dummy_incomping_from_model)
                    out_2 = model_2(out_1)
                    loss = loss_func(out,out_2)
                    print(loss)
                    grads = tape.gradient(loss,model_2.trainable_variables)
                    optim.apply_gradients(zip(grads,model_2.trainable_variables))
            else:
                out_2 = model_2.predict(out_1)
                final_out = np.argmax(out_2)

                if final_out == np.argmax(out):
                  correct+=1
                print("accuracy:::",correct/total)
