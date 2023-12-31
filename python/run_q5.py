import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20

batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################

in_size = train_x.shape[1] # 1024
initialize_weights(in_size, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size,params, 'hidden1')
initialize_weights(hidden_size, hidden_size,params, 'hidden2')
initialize_weights(hidden_size, in_size,params, 'output')

# Make new params (m_Wlayer1, ...) for momentum 
keys = [k for k in params.keys()]
for k in keys:
    params['m_' + k] = np.zeros(params[k].shape)

# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        pass
        h1 = forward(X=xb, params=params, name='layer1', activation=relu)
        h2 = forward(X=h1, params=params, name='hidden1', activation=relu)
        h3 = forward(X=h2, params=params, name='hidden2', activation=relu)
        x_recon = forward(X=h3, params=params, name='output', activation=sigmoid)

        loss = np.sum((x_recon-xb)**2)
        total_loss += loss

        delta1 = 2*(x_recon-xb)
        delta2 = backwards(delta=delta1, params=params, name='output', activation_deriv=sigmoid_deriv)
        delta3 = backwards(delta=delta2, params=params, name='hidden2', activation_deriv=relu_deriv)
        delta4 = backwards(delta=delta3, params=params, name='hidden1', activation_deriv=relu_deriv)
        backwards(delta=delta4, params=params, name='layer1', activation_deriv=relu_deriv)

        # Momentum update
        for k in params.keys():
            if '_' not in k:
                params['m_'+k] = 0.9 * params['m_'+k] - learning_rate * params['grad_'+k]
                params[k] += params['m_'+k] 

    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr, total_loss))
    if itr % lr_rate == lr_rate - 1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
##########################
##### your code here #####
##########################

h1 = forward(visualize_x, params=params, name='layer1', activation=relu)
h2 = forward(X=h1, params=params, name='hidden1', activation=relu)
h3 = forward(X=h2, params=params, name='hidden2', activation=relu)
reconstructed_x = forward(X=h3, params=params, name='output', activation=sigmoid)

# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
        # ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")

    ax.set_axis_off()
plt.show()

# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
num_valid = len(valid_x)
psnr = 0
for x in valid_x:
    h1 = forward(x, params=params, name='layer1', activation=relu)
    h2 = forward(X=h1, params=params, name='hidden1', activation=relu)
    h3 = forward(X=h2, params=params, name='hidden2', activation=relu)
    reconstructed_x = forward(X=h3, params=params, name='output', activation=sigmoid)

    psnr += peak_signal_noise_ratio(x, reconstructed_x)
psnr /= num_valid
print("PSNR: {0}".format(psnr))