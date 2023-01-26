# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction

from PIL import Image


"""
python demo/demo_skeleton.py demo/demo2.MP4 demo/demo1.MP4 --config configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py --checkpoint work_dirs/slowonly_r50_u48_240e_ntu120_xsub_keypoint/latest.pth --det_config configs/detection/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py --det_checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth --det_score_thr 0.3 --pose_config configs/skeleton/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py --pose_checkpoint checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth --label_map tools/data/skeleton/label_map_aicity.txt
"""

from mmaction.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model`, '
                      '`init_pose_model`, and `vis_pose_result` form '
                      '`mmpose.apis`. These apis are required in this demo! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument(
        '--out_filename',
        default=('demo/outputs/'
                 'filename.avi'),
        help='Output video')
    # parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('work_dirs/slowonly_r50_u48_240e_ntu120_xsub_keypoint/'
                 'latest.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det_config',
        default='configs/detection/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det_checkpoint',
        default=('checkpoints/'
                 'faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose_config',
        default='configs/skeleton/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose_checkpoint',
        default=('checkpoints/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det_score_thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label_map',
        default='tools/data/skeleton/label_map_aicity.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short_side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
                        11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
                        21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
                        31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
                        41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
                        51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
                        61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
                        71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' } 

def main_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    num_frame = len(frame_paths)
    model_det = init_detector(args.det_config, args.det_checkpoint, args.device)
    model_pose = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    assert model_det.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    mmdet_result_persons = []
    mmdet_result_bottles = []
    mmdet_result_phones = []
    human_pose_results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        mmdet_result_infer = inference_detector(model_det, frame_path)

        mmdet_result_person = mmdet_result_infer[0][mmdet_result_infer[0][:, 4] >= 0.92]# args.det_score_thr]
        
        mmdet_result_bottle = mmdet_result_infer[39][mmdet_result_infer[39][:, 4] >= 0.3]
        
        mmdet_result_phone = mmdet_result_infer[67][mmdet_result_infer[67][:, 4] >= 0.3]

        person_bboxes = [dict(bbox=x) for x in list(mmdet_result_person)]
        
        mmdet_result_persons.append(mmdet_result_person)
        mmdet_result_bottles.append(mmdet_result_bottle)
        mmdet_result_phones.append(mmdet_result_phone)
        
        human_pose = inference_top_down_pose_model(model_pose, frame_path, person_bboxes, format='xyxy')[0]

        human_pose_results.append(human_pose)

        prog_bar.update()

    vis_frames = [
        vis_pose_result(model_pose, frame_paths[i], human_pose_results[i])
        for i in range(num_frame)
    ]
    return mmdet_result_persons,human_pose_results,mmdet_result_bottles,mmdet_result_phones,vis_frames


# def pose_inference(args, frame_paths, det_results):
#     model_pose = init_pose_model(args.pose_config, args.pose_checkpoint,
#                             args.device)
#     ret = []
#     print('Performing Human Pose Estimation for each frame')
#     prog_bar = mmcv.ProgressBar(len(frame_paths))
#     for f, d in zip(frame_paths, det_results):
#         # Align input format
#         d = [dict(bbox=x) for x in list(d)]
#         pose = inference_top_down_pose_model(model_pose, f, d, format='xyxy')[0]
#         ret.append(pose)
#         prog_bar.update()
#     return ret


def main():
    args = parse_args()

    frame_paths, original_frames = frame_extraction(args.video,
                                                    args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape
    size = (w, h)
    result_video = cv2.VideoWriter(args.out_filename, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmcv.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    for component in config.data.test.pipeline:
        if component['type'] == 'PoseNormalize':
            component['mean'] = (w // 2, h // 2, .5)
            component['max_value'] = (w, h, 1.)

    model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Get Human poses and object detection results
    det_results,pose_results,mmdet_result_bottles,mmdet_result_phones,vis_frames = main_inference(args, frame_paths)
    torch.cuda.empty_cache()

    # pose_results = pose_inference(args, frame_paths, det_results)
    # torch.cuda.empty_cache()

    # print("\n \n Pose_results", pose_results[0])
    
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)

    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    # print("\n fake_anno['keypoint'] : ",fake_anno['keypoint'])

    # print("\n fake_anno['keypoint_score'] : ",fake_anno['keypoint_score'])

    results = inference_recognizer(model, fake_anno)

    # print("\n results : ",results)
    
    action_label = label_map[results[0][0]]

    # pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
    #                              args.device)
    

    # vis_frames = [
    #     vis_pose_result(pose_model, frame_paths[i], pose_results[i])
    #     for i in range(num_frame)
    # ]

    print("\n len(vis_frames): ",len(vis_frames))
    print("\n vis_frames: ",vis_frames[279])

    print("\n \n mmdet_result_bottles: ",mmdet_result_bottles[279])
    bbox_color=(0, 255, 0)
    text_color=(0, 0, 255)
    for frame, count  in zip(vis_frames, range(len(vis_frames))):
        cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)

        print("\n \n pose_results[count]", pose_results[count])
        for id_keypoint, pose_keypoints in enumerate(pose_results[count]):
            if len(pose_keypoints) == 0:
                continue
            print("id_keypoint: ",id_keypoint)
            print("pose_keypoints: ",pose_keypoints["keypoints"])
            x, y, _ = pose_keypoints["keypoints"][1] #
            print("x, y, _: ",x, y)
        for i in range(len(mmdet_result_bottles[count])):
            left = int(mmdet_result_bottles[count][i, 0])
            top = int(mmdet_result_bottles[count][i, 1])
            right = int(mmdet_result_bottles[count][i, 2])
            bottom = int(mmdet_result_bottles[count][i, 3])
            caption = "bottle"
            cv2.rectangle(frame, (left, top), (right, bottom), color=bbox_color, thickness=2)
            cv2.putText(frame, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.37, text_color, 1)
            print("left,top,right,bottom",left,top,right,bottom)  
        for i in range(len(mmdet_result_phones[count])):
            left = int(mmdet_result_phones[count][i, 0])
            top = int(mmdet_result_phones[count][i, 1])
            right = int(mmdet_result_phones[count][i, 2])
            bottom = int(mmdet_result_phones[count][i, 3])
            caption = "phone"
            cv2.rectangle(frame, (left, top), (right, bottom), color=bbox_color, thickness=2)
            cv2.putText(frame, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.37, text_color, 1)
            print("left,top,right,bottom",left,top,right,bottom)                                               
        result_video.write(frame)   
        print("counts ",count)                                            
    # for frame in vis_frames:
    # for i in len(vis_frames):
    #     cv2.imshow("Name",vis_frames[0])
        # for i in range(len(mmdet_result_bottles)):
        #     left = int(mmdet_result_bottles[i, 0])
        #     top = int(mmdet_result_bottles[i, 1])
        #     right = int(mmdet_result_bottles[i, 2])
        #     bottom = int(mmdet_result_bottles[i, 3])
        #     caption = "bottles"
        # cv2.putText(vis_frames[i], action_label, (10, 30), FONTFACE, FONTSCALE,
        #             FONTCOLOR, THICKNESS, LINETYPE)
        # result_video.write(vis_frames[i])        

    # vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    # vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)

    result_video.release()
if __name__ == '__main__':
    main()
