import torch
import torch.nn as nn
import torchvision
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import math

import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os

from PIL import Image
import cfgs.config as cfg
import scipy.misc 
import cv2
import matplotlib.pyplot as plt
from cStringIO import StringIO


# save the patch as numpy
def save_patch(patch, epoch):
    patch_size = patch.size(2)
    patch_np = patch.data.cpu().numpy()
        
    save_patch_name = os.path.join(cfg.patch_dir, '{}.npy'.format(epoch))
    print("save patch as ", save_patch_name)
    np.save(save_patch_name, patch_np)

    patch_img_np = np.zeros((patch_size, patch_size,3))
    patch_img_np[:,:,0] = patch_np[0][0]*255.0 # B(0)
    patch_img_np[:,:,1] = patch_np[0][1]*255.0 # G(1)
    patch_img_np[:,:,2] = patch_np[0][2]*255.0 # R(2)
    np.transpose(patch_img_np, (2,1,0)) #RGB

    patch_img = Image.fromarray(patch_img_np.astype('uint8'))
    save_patch_img = os.path.join(cfg.patch_dir, '{}.png'.format(epoch))
    print("save patch as img ", save_patch_img)
    patch_img.save(save_patch_img)

def create_patch_mask(in_features, my_patch, patch_size):
    width = in_features.size(1)
    height = in_features.size(2)
    patch_mask = torch.zeros([3, width,height])

    p_w = patch_size + cfg.patch_x
    p_h = patch_size + cfg.patch_y
    patch_mask[:, int(cfg.patch_x):int(p_w), int(cfg.patch_y):int(p_h)]= 1

    return patch_mask

def create_patch_mask_bbox(im_data, bbox, advpatch):
    width = im_data.size(1)
    height = im_data.size(2)
    patch_mask = torch.zeros([3,width,height])

    p_w = bbox[2]-bbox[0]
    p_h = bbox[3]-bbox[1]
    patch_mask[:, 0:p_w,0:p_h]=1
    return patch_mask

def create_img_mask(in_features, patch_mask):
    mask = torch.ones([3,in_features.size(1), in_features.size(2)])
    img_mask = mask - patch_mask

    return img_mask


# add a patch to the original image
def add_patch(in_features, my_patch):
    
    # in_features: [1,3,416,416]
    patch_size = cfg.patch_size
    patch_mask = create_patch_mask(in_features, my_patch, patch_size)

    img_mask = create_img_mask(in_features, patch_mask)

    patch_mask = Variable(patch_mask.cuda(), requires_grad=False)
    img_mask = Variable(img_mask.cuda(), requires_grad=False)


    with_patch = in_features * img_mask + my_patch * patch_mask
    
    return with_patch

def try_save_img(im_data, name):
    h = im_data.size(1)
    w = im_data.size(2)
    im_feature = im_data.data.cpu().numpy()

    print('im_feature', im_feature.shape) # 3 416 416
    patch_img_np = np.zeros([3, h, w])
    patch_img_np[0, :,:] = im_feature[0]*255.0
    patch_img_np[1, :,:] = im_feature[1]*255.
    patch_img_np[2,:,:] = im_feature[2]*255.
    Image.fromarray(patch_img_np.astype('uint8')).save('BGR_perturb.png')


def save_img(im_data, name):
#    print('------------before reshape ---------------')
#    print(im_data)

    h = im_data.size(1)
    w = im_data.size(2)
    im_t = im_data*255.
    torchvision.utils.save_image(im_t, 'test.png')
    im_feature = im_data.data.cpu().numpy()
    
    patch_img_np = np.zeros([h, w,3])
    
    patch_img_np[:,:,0] = im_feature[0]*255.0
    patch_img_np[:,:,1] = im_feature[1]*255.0
    patch_img_np[:,:,2] = im_feature[2]*255.0
    
    np.transpose(patch_img_np, (2,1,0))
    # cv2 to save the image
    cv2.imwrite('with_patch.png', patch_img_np, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    #np_to_save.save('with_patch.png')


    Image.fromarray(patch_img_np.astype('uint8')).save('BGR_perturb.png')
    np.transpose(patch_img_np, (2,1,0)) #RGB
    Image.fromarray(patch_img_np.astype('uint8')).save('RGB_perturb.png')
    print('-------- shape ----------')
    print(patch_img_np.shape)

    #scipy.misc.toimage(patch_img_np, cmin=0.0, cmax=255).save('outfile'+name+'.png')
    patch_img = Image.fromarray(patch_img_np.astype('uint8'))
    #save_patch_img = os.path.join('{}.png'.format(name))
    #print("save patch as img ", save_patch_img)
    #patch_img.save(save_patch_img)

def transform_patch(width, x_shift, y_shift, im_scale, rot_in_degree):
    """
      If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
      then it maps the output point (x, y) to a transformed input point 
      (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
      where k = c0 x + c1 y + 1. 
      The transforms are inverted compared to the transform mapping input points to output points.
     """
    rot = float(rot_in_degree) /90. *(math.pi/2)
     
    # rotation matrix
    rot_matrix = np.array( 
                [[math.cos(rot), -math.sin(rot)],
                 [math.sin(rot), math.cos(rot)]] )

    # scale it
     
