import sys
import time
from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from utils.utils import *
from tool.darknet2pytorch import Darknet
import cv2


def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    checkpoint = torch.load(weightfile)
    model_dict = m.state_dict()
    pretrained_dict = checkpoint
    keys = []
    for k, v in pretrained_dict.items():
        keys.append(k)
    i = 0
    for k, v in model_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            i = i + 1
    m.load_state_dict(model_dict)

    # m.load_state_dict(torch.load(weightfile))

    # m.print_network()
    # m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    namesfile = 'data/mydata.names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    input_img = cv2.imread(imgfile)
    # orig_img = Image.open(imgfile).convert('RGB')

    start = time.time()
    boxes,scale = do_detect(m, input_img, 0.5, 0.4, use_cuda)
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)

    # draw_boxes(input_img,boxes,scale=scale)
    plot_boxes_cv2(input_img, boxes, 'predictions1.jpg',class_names=class_names,scale=scale)

if __name__ == '__main__':
    cfgfile = r'cfg/yolov4.cfg'
    weightfile = r'weight/net1.pth'
    imgfile = r'data/3-369.jpg'
    detect(cfgfile,weightfile,imgfile)
