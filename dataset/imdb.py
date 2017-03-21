# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""The data base wrapper class"""

import os
import random
import shutil

from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from utils.util import iou, batch_iou, batch_overlaps

class imdb(object):
  """Image database."""

  def __init__(self, name, mc):
    self._name = name
    self._classes = []
    self._image_set = []
    self._image_idx = []
    self._data_root_path = []
    self._rois = {}
    self.mc = mc

    # batch reader
    self._perm_idx = None
    self._perm_rois = None
    self._cur_idx = 0

  @property
  def name(self):
    return self._name

  @property
  def classes(self):
    return self._classes

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def image_idx(self):
    return self._image_idx

  @property
  def image_set(self):
    return self._image_set

  @property
  def data_root_path(self):
    return self._data_root_path

  @property
  def year(self):
    return self._year

  def _shuffle_image_idx(self):
    perm = np.random.permutation(np.arange(len(self._image_idx)))
    self._perm_idx = [self._image_idx[i] for i in perm]
    self._perm_rois = [self._rois[i] for i in perm]
    self._cur_idx = 0

  def read_batch(self):
    """Read a batch of image and bounding box annotations.
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      batch_img: images. Shape: batch_size x width x height x [b, g, r]
      batch_cls_label: labels. Shape: batch_size x anchor_size x CLASSES
      batch_cls_label_mask: labels_mask. Shape: batch_size x anchor_size x CLASSES
      batch_bbox_delta: bounding box deltas. Shape: batch_size x anchor_size x 
          [dx ,dy, dw, dh]
      batch_bbox_delta_mask: bounding box deltas mask. Shape: batch_size x anchor_size x 
          [dx ,dy, dw, dh]
    """
    mc = self.mc

    if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
    batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
    batch_rois = self._perm_rois[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
    self._cur_idx += mc.BATCH_SIZE

    batch_img = np.zeros([mc.BATCH_SIZE,mc.IMAGE_HEIGHT,mc.IMAGE_WIDTH,3])
    batch_cls_label = np.zeros([mc.BATCH_SIZE,mc.ANCHORS,mc.CLASSES])
    batch_cls_label_mask = np.zeros([mc.BATCH_SIZE,mc.ANCHORS,mc.CLASSES])
    batch_bbox_delta = np.zeros([mc.BATCH_SIZE,mc.ANCHORS,4])
    batch_bbox_delta_mask = np.zeros([mc.BATCH_SIZE,mc.ANCHORS,4])

    for i in range(mc.BATCH_SIZE):
      # load the image
      im = cv2.imread(self._image_path_at(batch_idx[i])).astype(np.float32, copy=False)
      im -= mc.BGR_MEANS
      orig_h, orig_w, _ = [float(v) for v in im.shape]

      # load annotations
      gt_bbox = np.array([[b[0], b[1], b[2], b[3] ]for b in batch_rois[i][:]])
      gt_bbox_label = np.array([[b[4]] for b in batch_rois[i][:]])
      
      if batch_rois[i][0][5]:
          im = im[:, ::-1, :]

      # scale image
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))

      # scale annotation
      x_scale = mc.IMAGE_WIDTH/orig_w
      y_scale = mc.IMAGE_HEIGHT/orig_h
      gt_bbox[:, 0::2] = gt_bbox[:, 0::2]*x_scale
      gt_bbox[:, 1::2] = gt_bbox[:, 1::2]*y_scale


      ## overlaps
      overlaps = batch_overlaps(mc.ANCHOR_BOX,gt_bbox)
      
      ## get cls_labels adn cls_label_mask
      cls_labels = np.zeros([mc.ANCHORS,mc.CLASSES])
      cls_label_mask = np.ones([mc.ANCHORS,mc.CLASSES])
      for idx in range(mc.ANCHORS):
          max_overlaps = np.max(overlaps[idx,:],axis = 0)
          if max_overlaps>mc.POS_OVERLAPS_THRES:
              gt_box_idx = np.argmax(overlaps[idx,:])
              if gt_bbox_label[gt_box_idx]!=-1:
                 cls_labels[idx,gt_bbox_label[gt_box_idx]] = 1
              else:
                 cls_label_mask[idx,:] = 0
      for idx in range(len(gt_bbox)):
          argmax = np.argmax(overlaps[:,idx])
          if gt_bbox_label[idx]!=-1:
             cls_labels[argmax,gt_bbox_label[idx]] = 1
          else:
             cls_label_mask[argmax,:] = 0
             
      ## get bbox_delta and bbox_delta_mask
      bbox_delta = np.zeros([mc.ANCHORS,4])
      bbox_delta_mask = np.ones([mc.ANCHORS,4])
      for idx in range(mc.ANCHORS):
          if cls_label_mask[idx] and np.max(cls_labels[idx,:])>0:# only account for positive and not masked anchors
              gt_box_idx = np.argmax(overlaps[idx,:])
              box_cx, box_cy, box_w, box_h = gt_bbox[gt_box_idx]   
              bbox_delta[idx,0] = (box_cx - mc.ANCHOR_BOX[idx][0])/box_w
              bbox_delta[idx,1] = (box_cy - mc.ANCHOR_BOX[idx][1])/box_h
              bbox_delta[idx,2] = np.log(box_w/mc.ANCHOR_BOX[idx][2])
              bbox_delta[idx,3] = np.log(box_h/mc.ANCHOR_BOX[idx][3])
          else:
              bbox_delta_mask[idx,:]=0
      
      batch_img[i,:,:] = im
      batch_cls_label[i,:,:] = cls_labels
      batch_cls_label_mask[i,:] = cls_label_mask
      batch_bbox_delta[i,:,:] = bbox_delta
      batch_bbox_delta_mask[i,:,:] = bbox_delta_mask

    return batch_img, batch_cls_label, batch_cls_label_mask, \
        batch_bbox_delta, batch_bbox_delta_mask

  def evaluate_detections(self):
    raise NotImplementedError

  def visualize_detections(
      self, image_dir, image_format, det_error_file, output_image_dir,
      num_det_per_type=10):

    # load detections
    with open(det_error_file) as f:
      lines = f.readlines()
      random.shuffle(lines)
    f.close()

    dets_per_type = {}
    for line in lines:
      obj = line.strip().split(' ')
      error_type = obj[1]
      if error_type not in dets_per_type:
        dets_per_type[error_type] = [{
            'im_idx':obj[0], 
            'bbox':[float(obj[2]), float(obj[3]), float(obj[4]), float(obj[5])],
            'class':obj[6],
            'score': float(obj[7])
        }]
      else:
        dets_per_type[error_type].append({
            'im_idx':obj[0], 
            'bbox':[float(obj[2]), float(obj[3]), float(obj[4]), float(obj[5])],
            'class':obj[6],
            'score': float(obj[7])
        })

    out_ims = []
    # Randomly select some detections and plot them
    COLOR = (200, 200, 0)
    for error_type, dets in dets_per_type.iteritems():
      det_im_dir = os.path.join(output_image_dir, error_type)
      if os.path.exists(det_im_dir):
        shutil.rmtree(det_im_dir)
      os.makedirs(det_im_dir)

      for i in range(min(num_det_per_type, len(dets))):
        det = dets[i]
        im = Image.open(
            os.path.join(image_dir, det['im_idx']+image_format))
        draw = ImageDraw.Draw(im)
        draw.rectangle(det['bbox'], outline=COLOR)
        draw.text((det['bbox'][0], det['bbox'][1]), 
                  '{:s} ({:.2f})'.format(det['class'], det['score']),
                  fill=COLOR)
        out_im_path = os.path.join(det_im_dir, str(i)+image_format)
        im.save(out_im_path)
        im = np.array(im)
        out_ims.append(im[:,:,::-1]) # RGB to BGR
    return out_ims

