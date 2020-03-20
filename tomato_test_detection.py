from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test different models')

parser.add_argument('--epoch', type=str, default="latest", help='dataset version')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print("Visible GPU ID: %s" % args.gpu)

## Image sources
data_soure = '/data/quan/tomato_classification/Tomato_test_smaller'
save_result = '/data/quan/tomato_classification/result_Joe'

images = [os.path.join(data_soure, x) for x in os.listdir(data_soure)]
np.random.shuffle(images)
## Load configuration and trained model
config_name = 'tomato_faster_rcnn_x101_64x4d_fpn_1x'
config_file = 'configs/' + config_name + '.py'

if(args.epoch=="latest"):
	checkpoint_file = '/home/quan/WorkSpace/mmdetection_new/work_dirs/' + config_name + '/latest.pth'
else:
	checkpoint_file = '/home/quan/WorkSpace/mmdetection_new/work_dirs/' + config_name + '/epoch_%s.pth' % args.epoch
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda')

## Test the model
for file_path in tqdm(images[:100]):
	result = inference_detector(model, file_path)
	_, img_name = os.path.split(file_path)

	show_result(file_path, result, model.CLASSES, show=False, out_file=os.path.join(save_result, img_name))