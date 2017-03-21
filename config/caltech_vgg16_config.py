# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for caltech dataset"""

import numpy as np
from easydict import EasyDict as edict

def caltech_vgg16_config():
  """Specify the parameters to tune below."""
  mc = edict()

  # a small value used to prevent numerical instability
  mc.EPSILON = 1e-16
  # threshold for safe exponential operation
  mc.EXP_THRESH=1.0
  
  mc.CLASS_NAMES = ['pedestrian']
  mc.CLASSES = len(mc.CLASS_NAMES)
  mc.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])
  mc.KEEP_PROB = 0.5
  ## model input
  mc.IMAGE_WIDTH           = 800
  mc.IMAGE_HEIGHT          = 600
  mc.BATCH_SIZE            = 5
  ## anchor samples
  mc.POS_OVERLAPS_THRES = 0.5
  ## momentum sgd-solver param
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 10000
  mc.MAX_GRAD_NORM         = 1.0 # grad clipped value
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5
  ## loss coef
  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CLASS       = 1.0
  mc.WEIGHT_DECAY          = 0.0001
  ## nms param
  mc.NMS_THRESH            = 0.4
  mc.PLOT_PROB_THRESH      = 0.4
  mc.TOP_N_DETECTION       = 64
  ## data augmentation
  mc.DATA_AUGMENTATION     = True
  ## anchor box
  mc.ANCHOR_BOX            = set_caltech_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 9

  ## pretrained model
  mc.LOAD_PRETRAINED_MODEL = True
  mc.PRETRAINED_MODEL_PATH = '/home/bryant/rpn-tf-1.0/data/VGG16-pretrained/VGG_ILSVRC_16_layers_weights.pkl'
  return mc
  
def set_caltech_anchors(mc):
  H, W, B = 38, 50, 9
  h = np.linspace(40,515,9)*600.0/480
  w = 0.41*h
  anchor_shapes = np.reshape([np.transpose(np.vstack([w,h]))]*W*H,(H,W,B,2))
  
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(7.5, 16*W,16)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(7.5, 16*H,16)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors
