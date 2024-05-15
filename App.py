import torch
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st


def load_checkpoint(filepath, map_location):
    model = models.densenet121()  # we do not specify pretrained=True, i.e. do not load default weights
    model.classifier = nn.Sequential(nn.Linear(1024, 500),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(500, 6),
                                     nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(filepath, map_location))
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img = Image.open(image_path)
    ratio_aspect = max(img.size) / min(img.size)

    new_size = [0, 0]
    new_size[0] = 256
    new_size[1] = int(new_size[0] * ratio_aspect)

    img = img.resize(new_size)

    width, height = new_size

    # defining left, top, right, bottom margin
    l_margin = (width - 224) / 2
    t_margin = (height - 224) / 2
    r_margin = (width + 224) / 2
    b_margin = (height + 224) / 2
    # cropping
    img = img.crop((l_margin, t_margin, r_margin, b_margin))

    # converting to numpy array
    img = np.array(img)

    # Normalizing
    img = img / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # transpose to get color channel to 1st position
    img = img.transpose((2, 0, 1))

    return img


def imshow(image, ax=None, title=None):
    """Displays the Image to be classified"""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''

    model.cpu()
    model.eval()
    image = process_image(image_path)
    imshow(image)
    image = torch.FloatTensor(image)
    image.unsqueeze_(0)  # add a new dimension in pos 0

    output = model(image)
    # get the top k classes of prob
    ps = torch.exp(output).data[0]
    topk_prob, topk_idx = ps.topk(topk)

    topk_idx = topk_idx.numpy()
    topk_prob = topk_prob.numpy()

    return topk_prob, topk_idx

def main():
    piece_label = {0: "Bishop", 1: "King", 2: "knight", 3: "pawn", 4: "Queen", 5:"Rook"}
    # Loading the model
    if torch.cuda.is_available():
        map_location =lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    model = load_checkpoint('checkpoint.pth', map_location)

    st.title("Chess Piece Image Classifier ♟")

    piece_image = st.file_uploader(label="Upload Image", type=['png', 'jpg', 'jfif', 'webp'])

    if st.button("Classify"):
        topk_prob, topk_idx = predict(piece_image, model, topk=1)
        piece = piece_label[topk_idx[0]].upper()
        if topk_prob[0] > 0.5:
            st.image(piece_image)
            st.success(f'THIS IS A {piece}')
        else:
            st.success("Please upload a clearer image!")



if __name__ == '__main__':
    main()



