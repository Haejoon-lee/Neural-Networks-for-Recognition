from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

import glob

# Refered to: https://pytorch.org/vision/stable/models.html

# Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Initialize the inference transforms
preprocess = weights.transforms()

# Read a image
img = read_image("ILSVRC2012_val_00041865.JPEG") # Test on ImageNet validation image, category: Old English sheepdog

# Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")


# Testing on Old English Sheepdog video on each frame
filst = sorted(glob.glob('frames_english_sheepdog/*.jpeg'))
num_frames = len(filst)

count_cor = 0
for i in range(num_frames):
    
    img = read_image(filst[i])

    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    # print(f"{category_name}: {100 * score:.1f}%")
    if class_id == 229:
        count_cor += 1


print('Accuracy from {0} frames: {1}%'.format(num_frames, 100*count_cor/num_frames))