import argparse

import numpy
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable

from unet import UNet
from utils import *


def predict_img(net, full_img, gpu=False):
    img = resize_and_crop(full_img)

    left = get_square(img, 0)
    right = get_square(img, 1)

    right = normalize(right)
    left = normalize(left)

    right = np.transpose(right, axes=[2, 0, 1])
    left = np.transpose(left, axes=[2, 0, 1])

    X_l = torch.FloatTensor(left).unsqueeze(0)
    X_r = torch.FloatTensor(right).unsqueeze(0)

    if gpu:
        X_l = Variable(X_l, volatile=True).cuda()
        X_r = Variable(X_r, volatile=True).cuda()
    else:
        X_l = Variable(X_l, volatile=True)
        X_r = Variable(X_r, volatile=True)

    y_l = F.sigmoid(net(X_l))
    y_r = F.sigmoid(net(X_r))
    y_l = F.upsample_bilinear(y_l, scale_factor=2).data[0][0].cpu().numpy()
    y_r = F.upsample_bilinear(y_r, scale_factor=2).data[0][0].cpu().numpy()

    y = merge_masks(y_l, y_r, full_img.size[0])
    yy = dense_crf(np.array(full_img).astype(np.uint8), y)

    return yy > 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--dataset-path', help='path to root of image dataset', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='directory to output the predictions', required=True)
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    args = parser.parse_args()
    print("Using model file : {}".format(args.model))
    net = UNet(3, 1)
    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
    else:
        net.cpu()
        print("Using CPU version of the net, this may be very slow")

    all_jpg_files = []
    for root, dirs, files in os.walk(args.dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                all_jpg_files.append(os.path.join(root, file))
                
    print("number of test files=",len(all_jpg_files))

    print("Loading model ...")
    net.load_state_dict(torch.load(args.model))
    print("Model loaded !")

    for i, fn in enumerate(all_jpg_files):
        #print("\nPredicting image {} ...".format(fn))
        img = Image.open(fn)
        out = predict_img(net, img, not args.cpu)
        
        
        output_path = os.path.join(args.output, *fn.split("/")[-2:])
        
        os.makedirs(os.path.join(*output_path.split("/")[0:-1]))
        
       
        result = Image.fromarray((out * 255).astype(numpy.uint8))
        result.save(output_path)
              
            

