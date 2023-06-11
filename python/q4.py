import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    image = skimage.restoration.denoise_bilateral(image, multichannel=True) # Denoising

    gray = skimage.color.rgb2gray(image) # Grayscale

    thresh = skimage.filters.threshold_otsu(gray) # Thresholding

    bw = skimage.morphology.closing(gray<thresh, skimage.morphology.square(7)) # Morphology

    label_image = skimage.morphology.label(bw,connectivity=2) # Label

    #Skip small boxes
    props_region = skimage.measure.regionprops(label_image)
    total_area = 0
    for region in props_region:
        total_area += region.area
    mean_area = total_area / len(props_region)

    for region in props_region:
        if region.area > 0.5 * mean_area:
            bboxes.append(region.bbox)

    bw = (~bw).astype(np.float)
    
    return bboxes, bw