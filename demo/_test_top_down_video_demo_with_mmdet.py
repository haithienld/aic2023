# Copyright (c) OpenMMLab. All rights reserved.
"""
python demo/_test_top_down_video_demo_with_mmdet.py configs/detection/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth configs/skeleton/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth --video-path demo/demo1.MP4 --out-video-root demo/outputs/DB_24491
"""
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np

from mmpose.apis import (
    collect_multi_frames,
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
    vis_pose_result,
)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
#                         11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
#                         21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
#                         31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
#                         41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
#                         51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
#                         61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
#                         71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' }

def get_labeled_detected_img(results, img_array, score_threshold, is_print=True):
    # ????????? ????????? image_array??? ??????.
    draw_img = img_array.copy()
    bbox_color = (0, 255, 0)
    text_color = (0, 0, 255)
    print(results)

    # model??? image array??? ?????? ????????? inference detection ???????????? ????????? results??? ??????.
    # results??? 80?????? 2?????? array(shape=(??????????????????, 5))??? ????????? list.

    person_results = process_mmdet_results(results, 1)

def get_detected_img(results, img_array, score_threshold, is_print=True):
    # ????????? ????????? image_array??? ??????.
    draw_img = img_array.copy()
    bbox_color = (0, 255, 0)
    text_color = (0, 0, 255)

    # model??? image array??? ?????? ????????? inference detection ???????????? ????????? results??? ??????.
    # results??? 80?????? 2?????? array(shape=(??????????????????, 5))??? ????????? list.

    person_results = process_mmdet_results(results, 1)

    # print("person_results", person_results)
    for result_ind, result in enumerate(person_results):
        if len(result) == 0:
            continue
        result = result["bbox"]
        result_filtered = result[
            np.where((result[:] > 0) & (result[4] > score_threshold))
        ]
        # print("result_ind", result_ind)
        # print(
        #     "result", result_filtered
        # )  # result[np.where((result[:] > 0.9)&(result[4] > 0.6))])
        # print("len(result_filtered)", result_filtered)
        if len(result_filtered) != 0:
            left = int(result_filtered[0])
            top = int(result_filtered[1])
            right = int(result_filtered[2])
            bottom = int(result_filtered[3])
            caption = "human"

            # print("left,top,right,bottom,caption", left, top, right, bottom, caption)
            cv2.rectangle(
                draw_img, (left, top), (right, bottom), color=bbox_color, thickness=2
            )

            cv2.putText(
                draw_img,
                caption,
                (int(left), int(top - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.37,
                text_color,
                1,
            )
        if is_print:
            print(caption)
    return draw_img



def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    Args:
        mmdet_results (list|tuple): mmdet results.
        person_id: 0
        cell phone: 
        cat_id (int): category id (default: 1 for human)

    Returns:
        person_results (list): a list of detected bounding boxes
    """
    # print("mmdet_results",mmdet_results)
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person["bbox"] = bbox
        person_results.append(person)

    return person_results


def main():

    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """

    parser = ArgumentParser()
    parser.add_argument("det_config", help="Config file for detection")
    parser.add_argument("det_checkpoint", help="Checkpoint file for detection")
    # *****Tommy*****#
    # ***************#
    parser.add_argument("pose_config", help="Config file for pose")
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument("--video-path", type=str, help="Video path")
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="whether to show visualizations.",
    )
    parser.add_argument(
        "--out-video-root",
        default="",
        help="Root of the output video file. "
        "Default not saving the visualization video.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--det-cat-id",
        type=int,
        default=1,
        help="Category id for bounding box detection model",
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Keypoint score threshold"
    )
    parser.add_argument(
        "--radius", type=int, default=4, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )

    parser.add_argument(
        "--use-multi-frames",
        action="store_true",
        default=False,
        help="whether to use multi frames for inference in the pose"
        "estimation stage. Default: False.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        default=False,
        help="inference mode. If set to True, can not use future frame"
        "information when using multi frames for inference in the pose"
        "estimation stage. Default: False.",
    )

    assert has_mmdet, "Please install mmdet to run the demo."

    args = parser.parse_args()

    assert args.show or (args.out_video_root != "")
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    print("Initializing model...")
    # build the detection model from a config file and a checkpoint file

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower()
    )

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower()
    )

    # *****Tommy*****#
    # det_hand_model = init_detector(
    #     args.det_hand_config, args.det_hand_checkpoint, device=args.device.lower()
    # )
    # ***************#

    dataset = pose_model.cfg.data["test"]["type"]

    # get datasetinfo
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
            DeprecationWarning,
        )
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video

    video = mmcv.VideoReader(args.video_path)

    assert video.opened, f"Faild to load video file {args.video_path}"

    if args.out_video_root == "":
        save_out_video = False

    else:

        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoWriter = cv2.VideoWriter(
            os.path.join(
                args.out_video_root, f"vis_{os.path.basename(args.video_path)}"
            ),
            fourcc,
            fps,
            size,
        )

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert "frame_indices_test" in pose_model.cfg.data.test.data_cfg
        indices = pose_model.cfg.data.test.data_cfg["frame_indices_test"]

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print("Running inference...")
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)

        print("\n frame_id: \n", frame_id)

        mmdet_results = inference_detector(det_model, cur_frame)

        # *****Tommy*****#
        # mmdet_hand_results = inference_detector(det_hand_model , cur_frame)
        # cur_frame = det_hand_model.show_result(cur_frame, mmdet_hand_results, score_thr=0.4)
        # ***************#

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices, args.online)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            frames if args.use_multi_frames else cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names,
        )

        for id_keypoint, pose_keypoints in enumerate(pose_results):
            if len(pose_keypoints) == 0:
                continue
            x, y, _ = pose_keypoints["keypoints"][1]
            # pose_keypoints = pose_keypoints["keypoints"]
            # pose_bbox = pose_bbox["bbox"]
            # print("id_keypoint", id_keypoint)
            # print("pose_keypoints", x, y)

        # print("returned_outputs",returned_outputs)
        # show the results

        vis_frame = vis_pose_result(
            pose_model,
            cur_frame,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False,
        )

        vis_frame = get_detected_img(
            mmdet_results, vis_frame, score_threshold=0.8, is_print=False
        )

        if args.show:
            cv2.imshow("Frame", vis_frame)

        if save_out_video:
            videoWriter.write(vis_frame)

        if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if save_out_video:
        videoWriter.release()

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
