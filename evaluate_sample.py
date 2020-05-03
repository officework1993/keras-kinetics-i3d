'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import numpy as np
import argparse

#imports for dataloader
from data_loader import Echo
import tensorflow as tf
import tqdm

from i3d_inception import Inception_Inflated3d

NUM_FRAMES = 79
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

SAMPLE_DATA_PATH = {
    'rgb' : 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow' : 'data/v_CricketShot_g04_c01_flow.npy'
}

LABEL_MAP_PATH = 'data/label_map.txt'

def get_mean_and_std(dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):
    dataloader = tf.data.Dataset.from_generator(dataset, output_types=(tf.float32, tf.float32)).\
        shuffle(buffer_size=256).batch(batch_size)

    if samples is not None and len(dataset) > samples:
        dataloader = dataloader.take(samples)


    n = 0  
    s1 = 0.  
    s2 = 0. 
    for (x, *_) in tqdm.tqdm(dataloader):
        x = tf.transpose(x, perm=[1,0,2,3,4])
        x = tf.reshape(x, [3,-1])
        # x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += tf.math.reduce_sum(x, axis=1).numpy()
        s2 += tf.math.reduce_sum(x ** 2, axis=1).numpy()
        # s1 += torch.sum(x, dim=1).numpy()
        # s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std

def main(args,to_predict):
    # load the kinetics classes
    kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]


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
        flow_logits = flow_model.predict(to_predict)


    # produce final model logits
    if args.eval_type == 'rgb':
        sample_logits = rgb_logits
    elif args.eval_type == 'flow':
        sample_logits = flow_logits
    else: # joint
        sample_logits = rgb_logits + flow_logits

    import pdb;pdb.set_trace()
    # produce softmax output from model logit for class probabilities
    sample_logits = sample_logits[0] # we are dealing with just one example
    sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))

    sorted_indices = np.argsort(sample_predictions)[::-1]



    return


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-type', 
        help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).', 
        type=str, choices=['rgb', 'flow', 'joint'], default='joint')

    parser.add_argument('--no-imagenet-pretrained',
        help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
        action='store_true')


    args = parser.parse_args()
    mean, std = get_mean_and_std(Echo(root="a4c-video-dir/",split="train"))
    kwargs = {"target_type": 'EF',
          "mean": mean,
          "std": std,
          "length": 79,
          "period": 2,
          }
    train_dataset = Echo(root="a4c-video-dir/",split="train", **kwargs)
    train_dataloader = tf.data.Dataset.from_generator(train_dataset, output_types=(tf.float32, tf.float32)).\
        shuffle(buffer_size=256).batch(1)

    for X,EF_val in train_dataloader:
        if EF_val.numpy()>75:
            continue
        else:
            main(args,X)
