#pip install imghdr
#pip install python-magic

import argparse
import glob
import os
import time
from loguru import logger

import cv2
import torch
import io
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess
import json
import imghdr
import magic
from paddleocr import PaddleOCR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
import contextlib
import itertools
import numpy as np
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to load model into memory


def check_status(file_path):
    if imghdr.what(file_path):
        status = "image"
    else:
        mime_type = magic.from_file(file_path, mime=True)
        if mime_type.startswith('video'):
            status = "video"
        else:
            status = None
    return status


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")

    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )

    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )

    return parser


def load_ground_truths(annotations_file):
    with open(annotations_file) as f:
        data = json.load(f)
    #ground_truths = {}
    hashmap = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        for image in data['images']:
            if image["id"] == image_id:
                ann["file_name"] = image["file_name"]
                hashmap[image['file_name']] = image_id
    return data, hashmap


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            args,
            cls_names=COCO_CLASSES,
            device="cpu",
            fp16=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = 0.5
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.args = args
        self.preproc = ValTransform()
        print("Device==", self.device)

    def inference(self, file_path, vis_folder):
        current_time = time.localtime()
        ground_truths, hashmap = load_ground_truths(
            "/app/datasets/COCO/annotations/instances_val2017.json")
        if os.path.isdir(file_path):
            files = glob.glob(os.path.join(file_path, '*'))
        else:
            files = [file_path]

        pred_list = []
        #file_list = []
        index = 0
        for file_name in files:
            status = check_status(file_name)

            if status == "image":
                img = cv2.imread(file_name)
                image_name = os.path.basename(file_name)
                outputs, img_info = self.prediction(img)

                if outputs[0] == None:
                    # file_list.append(file_name)
                    # index += 1
                    pass
                else:
                    result_image, bboxes_values, score, cls_name = self.visual(outputs[0], img_info, self.confthre)

                    bboxs = self.xyxy2xywh(bboxes_values)
                    predict_info = {
                        "image_id": hashmap[image_name],
                        "category_id": cls_name.cpu().numpy().astype(int).tolist()[0] + 1,
                        "bbox": bboxs.cpu().numpy().astype(int).tolist()[0],
                        "score": score.cpu().numpy().tolist()[0]
                    }
                    pred_list.append(predict_info)

                    if self.args.save_result:
                        save_folder = os.path.join(
                            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                        )
                        os.makedirs(save_folder, exist_ok=True)
                        save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                        logger.info("Saving detection result in {}".format(save_file_name))
                        cv2.imwrite(save_file_name, result_image)
            else:
                if status == "video":
                    cap = cv2.VideoCapture(file_name)
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

                    if self.args.save_result:
                        save_folder = os.path.join(
                            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                        )
                        os.makedirs(save_folder, exist_ok=True)
                        #if self.args.demo == "video":
                        save_path = os.path.join(save_folder, os.path.basename(file_name))
                        # else:
                        #     save_path = os.path.join(save_folder, "camera.mp4")
                        logger.info(f"video save_path is {save_path}")
                        vid_writer = cv2.VideoWriter(
                            save_path, fourcc, fps, (int(width), int(height)))
                    frame_count = 0
                    skipped_frames = 0

                    while True:
                        ret_val, frame = cap.read()
                        if not ret_val:
                            break
                        frame_count += 1
                        outputs, img_info = self.prediction(frame)
                        result_frame = self.visual(outputs[0], img_info, self.confthre, 1, 15, 2)
                        #print(result_frame)
                        if self.args.save_result:
                            vid_writer.write(result_frame[0])
                        else:
                            cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                            cv2.imshow("yolox", result_frame[0])
                        # ch = cv2.waitKey(int(1000 / fps))
                        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
                        #     break


                    cap.release()
                    if self.args.save_result:
                        vid_writer.release()

                    print(f"Total frames processed: {frame_count}")
                    print(f"Total frames skipped: {skipped_frames}")

        if pred_list != []:
            # Convert ground truth dictionary to COCO object
            coco_gt = COCO()
            coco_gt.dataset = ground_truths
            coco_gt.createIndex()

            # Convert predictions list to COCO object
            coco_dt = coco_gt.loadRes(pred_list)
            # Initialize COCOeval object
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            info = ""
            info += redirect_string.getvalue()
            cat_names = ["license"]

            AP_table = self.per_class_AP_table(coco_eval, class_names=cat_names)
            info += "per class AP:\n" + AP_table + "\n"

            AR_table = self.per_class_AR_table(coco_eval, class_names=cat_names)
            info += "per class AR:\n" + AR_table + "\n"
            print(info)

            # with open('inference_validation.json', 'w') as f:
            #     json.dump(pred_list, f)

    def prediction(self, img):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            print("output",outputs)
            outputs = postprocess(outputs, self.num_classes, self.confthre,
                                  self.nmsthre, class_agnostic=True)
            print(outputs)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35, text_scale=0.5, padding=5, thickeness=1):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return  img, [], [], []
        # if output is None:
        #     return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        for box_val, score in zip(bboxes, scores):
            if score >= self.confthre:
                x0, y0, x1, y1 = map(int, box_val)
                roi = img[y0:y1, x0:x1]

                x0 = int(box_val[0])
                y0 = int(box_val[1])
                x1 = int(box_val[2])
                y1 = int(box_val[3])

                text = ''
                result = ocr.ocr(roi, det=True)
                for idx in range(len(result)):
                    res = result[idx]
                    if res != None:
                        for line in res:
                            text += line[1][0] + " "
                # print("Paddle Ocr===", text)

                if text.strip() != "":
                    bg_color = (51, 159, 255)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    txt_size = cv2.getTextSize(text, font, text_scale, thickeness)[0]
                    cv2.rectangle(img, (x0, y0), (x1, y1), bg_color, 2)

                    cv2.rectangle(
                        img,
                        (x0, y0 - txt_size[1] - padding - 5),
                        (x0 + txt_size[0] + 2, y0 - 5),
                        bg_color,
                        -1
                    )
                    cv2.putText(img, text, (x0, y0 - int(0.5 * txt_size[1])), font, text_scale, (255, 255, 255),
                                thickness=thickeness, lineType=cv2.LINE_AA)
                else:
                    print("Empty Text>>>>>>>>>>")
        return img, bboxes, scores, cls

    def xyxy2xywh(self, bboxes_values):
        bboxes = bboxes_values.clone()
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return bboxes

    def per_class_AP_table(self, coco_eval, class_names=[], headers=["class", "AP"], colums=6):
        per_class_AP = {}
        precisions = coco_eval.eval["precision"]
        # dimension of precisions: [TxRxKxAxM]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            per_class_AP[name] = float(ap * 100)

        num_cols = min(colums, len(per_class_AP) * len(headers))
        result_pair = [x for pair in per_class_AP.items() for x in pair]
        row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
        )
        return table

    def per_class_AR_table(self, coco_eval, class_names=[], headers=["class", "AR"], colums=6):
        per_class_AR = {}
        recalls = coco_eval.eval["recall"]
        # dimension of recalls: [TxKxAxM]
        # recall has dims (iou, cls, area range, max dets)
        assert len(class_names) == recalls.shape[1]

        for idx, name in enumerate(class_names):
            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            ar = np.mean(recall) if recall.size else float("nan")
            per_class_AR[name] = float(ar * 100)

        num_cols = min(colums, len(per_class_AR) * len(headers))
        result_pair = [x for pair in per_class_AR.items() for x in pair]
        row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
        )
        return table


def main(exp, args):
    file_name = os.path.join(exp.output_dir, exp.exp_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    # if args.conf is not None:
    #     exp.test_conf = args.conf
    # if args.nms is not None:
    #     exp.nmsthre = args.nms
    # if args.tsize is not None:
    #     exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    predictor = Predictor(
        model, exp, args, COCO_CLASSES, args.device)
    predictor.inference(args.path, vis_folder)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(None, args.name)
    main(exp, args)
