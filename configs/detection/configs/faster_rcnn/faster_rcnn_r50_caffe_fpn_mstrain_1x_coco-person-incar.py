_base_ = './faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
model = dict(roi_head=dict(bbox_head=dict(num_classes=5)))
classes = ('steering wheel', 'cell phone', 'hat', 'bottle', 'control panel',)
data = dict(
    train=dict(img_prefix='data/incar/train/', classes=classes, ann_file='data/incar/train/annotation_coco.json'),
    val=dict(img_prefix='data/incar/val/', classes=classes, ann_file='data/incar/val/annotation_coco.json'),
    test=dict(img_prefix='data/incar/test/', classes=classes, ann_file='data/incar/test/annotation_coco.json'))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth'  # noqa
