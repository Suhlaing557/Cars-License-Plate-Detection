import imghdr
import magic
import cv2
import numpy as np
import argparse
import glob
import os
import time
from loguru import logger
import torch
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import get_model_info, postprocess
from paddleocr import PaddleOCR


class InputMedia:

    def __init__(self, filename, test_size):
        self.filename = filename
        self.preproc = ValTransform()
        self.status = self.statusFunc()
        self.test_size = test_size

        self.filename = filename

    def __str__(self):
        return f"FileName: {self.filename}, Status: {self.status}"

    def statusFunc(self):
        if imghdr.what(self.filename):
            status = "image"
        else:
            mime_type = magic.from_file(self.filename, mime=True)
            if mime_type.startswith('video'):
                status = "video"
            else:
                status = "unknown"
        return status

    def imageInfo(self, img):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["ratio"] = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        return img_info

    def readImage(self):
        img = cv2.imread(self.filename)
        img_info  = self.imageInfo(img)
        img = self.preprcess_frame(img)
        return img, img_info

    def preprcess_frame(self, frame):
        frame, _ = self.preproc(frame,None, self.test_size)
        frame = torch.from_numpy(frame).unsqueeze(0).float()
        return frame

   
class OcrRecognition:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to load model into memory

    def ocrText(self, image):
        text = ''
        result = self.ocr.ocr(image, det=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                text += line[1][0] + " "
        return text

class InferenceModel:
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
        self.ocr = OcrRecognition()

    def model_prediction(self, frame):
        with torch.no_grad():
            if self.device == "gpu":
                frame = frame.cuda()
                if self.fp16:
                    frame = frame.half()  # to FP16

            outputs = self.model(frame)
            outputs = postprocess(outputs, self.num_classes, self.confthre,
                                  self.nmsthre, class_agnostic=True)
        return outputs

    def predict(self, media, save_file_name):
        if media.status == 'image':
            imgs,image_info = media.readImage()
            outputs = self.model_prediction(imgs)
            self.write_image(outputs, image_info, save_file_name)
        elif media.status == 'video':
            self.video_process(media, save_file_name)
        else:
            raise ValueError("Unsupported media type")

    def annotate_frame(self, frame, outputs, ratio,text_scale=0.5, padding=5, thickness=1):
        if not outputs[0] is None or outputs != []:
            output = outputs[0].cpu().numpy()
            bboxes = output[:, 0:4]
            bboxes /= ratio
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            if not output is None:
                for box_val, score in zip(bboxes, scores):
                    if score >= self.confthre:
                        x0, y0, x1, y1 = map(int, box_val)
                        roi = frame[y0:y1, x0:x1]

                        x0, yo = int(box_val[0]), int(box_val[1])
                        x1, y1 = int(box_val[2]), int(box_val[3])

                        text = self.ocr.ocrText(roi)
                        if text.strip() != "":
                            bg_color = (51, 159, 255)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            txt_size = cv2.getTextSize(text, font, text_scale, thickness)[0]
                            cv2.rectangle(frame, (x0, y0), (x1, y1), bg_color, 2)

                            cv2.rectangle(
                                frame,
                                (x0, y0 - txt_size[1] - padding - 5),
                                (x0 + txt_size[0] + 2, y0 - 5),
                                bg_color,
                                -1
                            )
                            cv2.putText(frame, text, (x0, y0 - int(0.5 * txt_size[1])), font, text_scale, (255, 255, 255),
                                        thickness=thickness, lineType=cv2.LINE_AA)


    def write_image(self, outputs, img_info, output_path):
        frame = img_info["raw_img"]
        ratio = img_info["ratio"]
        if outputs[0] !=None:
            self.annotate_frame(frame, outputs, ratio)
        cv2.imwrite(output_path, frame)

    def video_process(self, media, save_path):
        cap = cv2.VideoCapture(media.filename)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vid_writer = cv2.VideoWriter(
            save_path, fourcc, fps, (int(width), int(height)))

        while True:
            ret_val, frame = cap.read()
            if not ret_val:
                break
            img_info = media.imageInfo(frame)
            img = media.preprcess_frame(frame)
            outputs = self.model_prediction(img)
            if outputs[0] != None:
                self.annotate_frame(frame, outputs, img_info["ratio"],1,15,2)
            vid_writer.write(frame)
            # ch = cv2.waitKey(int(1000 / fps))
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break
        cap.release()
        vid_writer.release()


class MediaProcessor:
    def __init__(self, args, exp):
        self.exp = exp
        self.args = args
        self.input_file = args.path
        self.test_size = exp.test_size
        self.filename = os.path.join(exp.output_dir, exp.exp_name)
        self.output_folder = self.make_dirs()
        self.model = self.load_model()
        self.inference = InferenceModel(self.model, self.exp, self.args, COCO_CLASSES, self.args.device)

    def make_dirs(self):
        current_time = time.localtime()
        output_folder = None
        if self.args.save_result:
            output_folder = os.path.join(self.filename, "vis_res")
            os.makedirs(output_folder, exist_ok=True)
        save_folder = os.path.join(output_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        os.makedirs(save_folder, exist_ok=True)
        return save_folder

    def load_model(self):
        model = self.exp.get_model()
        if self.args.device == "gpu":
            model.cuda()
            if self.args.fp16:
                model.half()  # to FP16
        model.eval()
        if args.ckpt is None:
            ckpt_file = os.path.join(self.filename, "best_ckpt.pth")
            ckpt = torch.load(ckpt_file, map_location="cpu")
        else:
            ckpt_file = self.args.ckpt
            logger.info("loading checkpoint")

            ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return model

    def make_prediction(self):
        files = glob.glob(os.path.join(self.input_file, '*')) if os.path.isdir(self.input_file) else [self.input_file]
        for input_file in files:
           media = InputMedia(input_file, self.test_size)
           save_file_name = os.path.join(self.output_folder, os.path.basename(input_file))
           self.inference.predict(media, save_file_name)


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


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp  = get_exp(None, args.name)
    mediaProcessor = MediaProcessor(args, exp)
    mediaProcessor.make_prediction()