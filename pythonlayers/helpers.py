#helper function for blob loading and stuff

import numpy as np
import cv2

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def im_to_blob(im):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = im.shape
    blob = np.zeros((1, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    blob[0, 0:im.shape[0], 0:im.shape[1], :] = im
    
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


#def _get_image_from_binaryproto(filename):
  #TODO returns image from binaryproto

def _get_image_list_blob( im_list, mean_image):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    for i in xrange(num_images):
        #TODO random cropping and noise addition
        
        im = cv2.imread( im_list[i][0])
        target_size = 256
        im_scaley = float(target_size) / float(im.shape[0])
        im_scalex = float(target_size) / float(im.shape[1])
        im = cv2.resize(im_orig, None, None, fx=im_scalex, fy=im_scaley,
                            interpolation=cv2.INTER_LINEAR)
        im-=mean_image 
        processed_ims.append(im)

    #TODO  Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob

