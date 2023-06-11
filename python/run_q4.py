import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    print(bw.max())
    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    print('File name:', img)
    # Cordinates of bounding boxes: (x of center, y of center, width, height)
    cordis_letters = [((bbox[3]+bbox[1])//2, (bbox[2]+bbox[0])//2, bbox[3]-bbox[1], bbox[2]-bbox[0]) for bbox in bboxes]

    # sort by centerY
    cordis_letters = sorted(cordis_letters, key=lambda x: x[1])

    heights_letters = [bbox[2] - bbox[0] for bbox in bboxes]
    half_avg_height = np.mean(heights_letters)/2


    cordis_single_row = []
    cordis_rows = []
    y_pre = None
    for cordi in cordis_letters:
        if y_pre == None or cordi[1] - y_pre < half_avg_height: # Check whether the letter is on the same line with previous letter
            cordis_single_row.append(cordi)
        else: 
            cordis_single_row = sorted(cordis_single_row, key=lambda x: x[0]) # Sort according to x coordinates
            cordis_rows.append(cordis_single_row)
            cordis_single_row = [cordi]

        y_pre = cordi[1]

    # Last row
    cordis_single_row = sorted(cordis_single_row, key=lambda x: x[0])
    cordis_rows.append(cordis_single_row)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    letters_target = []
    for cordis_single_row in cordis_rows:
        row_raw_data = []
        for x, y, w, h in cordis_single_row:
            cropped = bw[y-h//2:y+h//2, x-w//2:x+w//2]
            # pad it to square
            if h < w:
                w_pad = w//16
                h_pad = (w-h)//2+w_pad
            else:
                h_pad = h//16
                w_pad = (h-w)//2+h_pad
            cropped = np.pad(cropped, ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=(1, 1))
            # resize to 32*32
            cropped = skimage.transform.resize(cropped, (32, 32))
            cropped = skimage.morphology.erosion(cropped, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
            cropped = cropped.T
            row_raw_data.append(cropped.reshape(-1))
        
        letters_target.append(np.array(row_raw_data))
    
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    for letters_row in letters_target:
        h1 = forward(letters_row, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        output_str = ''
        for i in range(probs.shape[0]):
            predict_idx = np.argmax(probs[i])
            output_str += letters[predict_idx]

        print(output_str)

    plt.show()