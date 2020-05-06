import sys
import time
from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from utils.utils import *
from tool.darknet2pytorch import Darknet

def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)
    m.load_state_dict(torch.load(weightfile))

    # m.print_network()
    # m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    num_classes = 20
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    input_img = cv2.imread(imgfile)
    orig_img = Image.open(imgfile)
    for i in range(2):
        start = time.time()
        boxes,scale = do_detect(m, input_img, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes(orig_img, boxes, 'predictions.jpg', class_names,scale=scale)

if __name__ == '__main__':
    cfgfile = r'cfg/yolov4.cfg'
    weightfile = r'weight/net.pth'
    imgfile = r'data/000017.jpg'
    detect(cfgfile,weightfile,imgfile)