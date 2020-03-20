
# MMDetection stable version

### Requirements
- Python 3.5+
- PyTorch 1.1 or higher
- CUDA 10.0 or higher
- NCCL 2
- GCC(G++) 4.9 or higher
- [mmcv] 0.2.14(https://github.com/open-mmlab/mmcv)

- CUDA: 10.0
- CUDNN: 7.6
- NCCL: 2.4	
- GCC(G++): 7.3

(Optional) If you want to install GCC version 7.3 using `conda`

In your conda environment:
`conda install gxx_linux-64`

### Install mmdetection
1. Create a conda virtual environment and activate it.

```shell
conda create -n py36 python=3.6 -y
conda activate py36
```

2. Install `mmcv` 0.2.14
```shell
pip install mmcv==0.2.14 --no-cache-dir
```

3. Install PyTorch stable for CUDA 10.0

```shell
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

4. Clone the mmdetection repository.

```shell
git clone https://github.com/huuquan1994/mmdetection_plant.git
cd mmdetection_plant
```

5. Install mmdetection (other dependencies will be installed automatically).

```shell
python setup.py develop
# or "pip install -v -e ."
```

### Test a dataset
Use the following commands to test a VOC format dataset.
```shell
# 1. Test and save the result
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]

# 2. Call a specific function for showing the VOC result pkl file
python tools/voc_eval.py ${RESULT_FILE} ${CONFIG_FILE}
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle (.pkl) format.
- `CONFIG_FILE`: Configuration file when used for training
- `CHECKPOINT_FILE`: The trained model you want to test

Examples:

1. Test and save the result into the `result.pkl`. Assume that you have already trained model `latest.pth`.

```shell
python tools/test.py configs/custom_faster_rcnn_r50_fpn_1x_VOC.py work_dirs/custom_faster_rcnn_r50_fpn_1x_VOC/latest.pth --out result.pkl
```
2. Show the result from `result.pkl`

```shell
python tools/voc_eval.py result.pkl configs/custom_faster_rcnn_r50_fpn_1x_VOC.py
```