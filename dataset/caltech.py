# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Image data base class for caltech"""

import cv2
import os 
import numpy as np
import subprocess

from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou

class caltech(imdb):
    def __init__(self, image_set = 'train', data_path = '/home/bryant/MATLAB-tools/caltech/',mc = None):
        imdb.__init__(self, 'caltech_'+image_set, mc)
        self._image_set = image_set
        self._data_root_path = data_path
        self._image_path = os.path.join(self._data_root_path, 'raw_image/')
        self._label_path = os.path.join(self._data_root_path, 'raw_annotations/')
        if image_set == 'test':
            self._image_set_num = ['set06/','set07/','set08/','set09/','set10/']
            self._image_set_stride = 30# sample 1 frame every 4 frames from _image_set
        elif image_set == 'valdation':
            self._image_set_num = ['set05/']
            self._image_set_stride = 30# sample 1 frame every 4 frames from _image_set  
        elif image_set == 'train':
            self._image_set_num = ['set00/','set01/','set02/','set03/','set04/','set05/']
            self._image_set_stride = 3# sample 1 frame every 4 frames from _image_set   
        self._classes = self.mc.CLASS_NAMES
        self._class_to_idx = dict(zip(self.classes, xrange(self.num_classes)))
        
        # a list of string indices of images in the directory
        self._image_idx = self._load_image_set_idx()
        # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
        # the image width and height
        if image_set=='train':
           self._rois = self.gt_roidb()
           if mc.DATA_AUGMENTATION:
               self._append_flip_rois()
        ## batch reader ##
        self._perm_idx = None
        self._cur_idx = 0
        # TODO(bichen): add a random seed as parameter
        if image_set=='train':
           self._shuffle_image_idx()

        self._eval_tool = None # './src/dataset/kitti-eval/cpp/evaluate_object'

    def _append_flip_rois(self):
        '''
        used to append flip annotations
        update self._img_idx && self._rois
        '''
        for idx in range(len(self._rois)):
            cur_rois = self._rois[idx]
            cur_img_idx = self._image_idx[idx]
            for roi_idx in range(len(cur_rois)):
                cur_rois[roi_idx][5] = 1
                cur_rois[roi_idx][0] = 640 - 1 - cur_rois[roi_idx][0]
            self._rois.append(cur_rois)
            self._image_idx.append(cur_img_idx)

    def _load_image_set_idx(self):
        '''
        get image_index_list, ie. all image path list
        '''
        image_full_path_list = []
        count = 0
        for set_name in self._image_set_num:
            image_sets_path = os.listdir(self._image_path + set_name)
            image_sets_path.sort()
            for video_name in image_sets_path:
                set_video_sets = os.listdir(self._image_path + set_name + video_name)
                set_video_sets.sort()
                count = 0
                for im_name in set_video_sets:   
                    if count % self._image_set_stride == 0 :
                        image_full_path = self._image_path + set_name + video_name + '/' + im_name
                        image_full_path_list.append(image_full_path)
                    count = count + 1
        return image_full_path_list
    
    def _image_path_at(self,idx):
        '''
        return path of i-th image
        '''
        return idx
    
    def gt_roidb(self):
        '''
        '''  
        gt_roidb=[]
        img_new_list = []
        for index in self.image_idx:
            roi1 = self._load_caltech_annotation(index)
            if roi1 is not None:
                gt_roidb.append(roi1)
                img_new_list.append(index)
        self._image_idx = img_new_list
        return gt_roidb
    
    def _filter_ann(self,line_str,filter_cond = None):
        '''
        input:
             line_str: str_line info
             filter_cond: see @ _load_caltech_annotation
        return:
             bboxes: [cent_x,cent_y,w,h]
             gt_class: 0/-1,true/ignore
        '''
        strs = line_str.split(' ')
        str_len = len(strs)
        label = strs[0]
        x = float(strs[1])
        y = float(strs[2])
        w = float(strs[3])
        h = float(strs[4])
        occ_flag = float(strs[5])
        vis_x = float(strs[6])
        vis_y = float(strs[7])
        vis_w = float(strs[8])
        vis_h = float(strs[9])
        
        ignore_flag = 0
        if str_len>9:
            ignore_flag = float(strs[10])
        if str_len>10: #not used now
            orientation_angle = float(strs[11])
        
        # priority: error_anno>ignore>true
        err_flag =  x<0 or y<0 or w<=0 or h<=0
        err_flag = err_flag or (label != filter_cond['lbls'] and label!=filter_cond['ilbls'])
        
        if err_flag:
            return [0,0,0,0,-2,0]
        
        ignore_flag = ignore_flag or (x+w)>640 or (y+h)>480 or label == filter_cond['ilbls']
        ignore_flag = ignore_flag or (not filter_cond['hRng'][1]>w>filter_cond['hRng'][0])
        if occ_flag:
            vis_ratio = 1.0*vis_w*vis_h/w/h
            ignore_flag = ignore_flag or (not filter_cond['vRng'][1]>vis_ratio>filter_cond['vRng'][0])
                    
        x2 = x + w
        y2 = y + h
        if x2>640:
            x2 = 640
        if y2>480:
            y2 = 480
        
        x, y, w, h = bbox_transform_inv([x,y,x2,y2])
        
        if ignore_flag:
            return [x,y,w,h,-1,0]
        
        return [x, y, w, h,0,0]
    
    def _load_caltech_annotation(self,index,filter_cond = None):
        '''
        input:
                index          :   single image_path of image,'./image_path'
                filter_cond    :   [squarify - [] controls optional reshaping of bbs to fixed aspect ratio
                                   lbls     - [] return objs with these labels (or [] to return all)
                                   ilbls    - [] return objs with these labels but set to ignore
                                   hRng     - [] range of acceptable obj heights
                                   wRng     - [] range of acceptable obj widths
                                   aRng     - [] range of acceptable obj areas
                                   arRng    - [] range of acceptable obj aspect ratios
                                   oRng     - [] range of acceptable obj orientations (angles)
                                   xRng     - [] range of x coordinates of bb extent
                                   yRng     - [] range of y coordinates of bb extent
                                   vRng     - [] range of acceptable obj occlusion levels
                                   ]
                                   
        return: 
                roidb list
                roidb[0] [x,y,w,h,cls_label,flip_flag]
                          
        annotation txt format: 'person,x,y,w,h,occ_flag,vis_x,vis_y,visw,vis_h,ignore_flag,orientation_angle'
        '''
        filter_con = {'squarity':0.41,'lbls':'person','ilbls':'people','hRng':[10,1000],'vRng':[0.7,1]}   

        pos1 = len(self._image_path)
        pos2 = index.find('.jpg')
        annotation_path = self._label_path + index[pos1:pos2] + '.txt'
        
        assert os.path.exists(annotation_path), 'Path does not exist: {}'.format(annotation_path)
        
        with open(annotation_path,'r') as f:
            lines = f.readlines()
            count = len(lines) - 1
            
        rois = []
        
        # gt_class is either true:0,or ignore:-1
        for line_num in range(count):
            roi = self._filter_ann(lines[line_num+1],filter_con)
            if roi[4]!=-2:
              rois.append(roi)
    
        if len(rois) == 0:
            return None#[[0,0,0,0,0,0]]
        else:
            return rois
