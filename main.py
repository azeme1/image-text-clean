import argparse
import os
import re
import ast
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import onnxruntime as rt
import easyocr

import craft_utils


def get_bounding_box(points):
    x, y, w, h = cv2.boundingRect(np.array(points).astype(np.float32))
    return x, y, x + w - 1, y + h - 1


def points_from_box(box_item):
    x1, y1, x2, y2 = box_item
    p = np.array(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)
    return p


def detect_text(frame):
    if not hasattr(detect_text, "model_craft_onnx"):
        model_craft_path = "./data/models/craft_mlt_25k.onnx"
        detect_text.model_craft_onnx = rt.InferenceSession(model_craft_path, providers=['CPUExecutionProvider'])
    #    providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
    input_name = detect_text.model_craft_onnx.get_inputs()[0].name
    y = detect_text.model_craft_onnx.run(None, {input_name: frame[None, ...]})[0]
    return y


def process_image(frame_data, inference_size=1280):
    low_text = 0.4
    poly = False
    text_threshold = 0.7
    link_threshold = 0.4

    frame_shape = frame_data.shape[:2]
    p_frame_src = points_from_box((0, 0) + tuple(reversed(frame_data.shape[:2])))

    normalized_shape = np.array(frame_shape)/np.array(frame_shape).max()
    shape_document_dst = (inference_size * normalized_shape).astype(np.int32)
    p_document_dst = points_from_box((0, 0) + tuple(reversed(shape_document_dst)))
    M_document_craft, _ = cv2.findHomography(p_frame_src, p_document_dst)
    shape_document_craft_dst = (inference_size * normalized_shape).astype(np.int32)

    frame_data_inference = cv2.warpPerspective(frame_data, M_document_craft, 
                                               tuple(reversed(shape_document_craft_dst)))

    y = detect_text(frame_data_inference)
    score_text = y[0, 0]
    score_link = y[0, 1]

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    ratio_w = ratio_h = max(np.array(frame_data_inference.shape[:2]))/(max(np.array(score_text.shape[:2]))*2)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    frame_src = (255 * frame_data_inference.astype(np.float32) / 270).astype(np.uint8)
    mask = np.zeros(frame_data_inference.shape[:2], dtype=np.uint8)

    for _, box in enumerate(boxes):
        p_src = np.array(box).astype(np.float32)

        frame_src = cv2.fillPoly(frame_src, [p_src.astype(np.int32)], True, 0)
        mask = cv2.fillPoly(mask, [p_src.astype(np.int32)], True, 255)

    mask = cv2.dilate(mask,  np.ones((7, 7), np.uint8))
    frame_dst = cv2.inpaint(frame_src, mask, 3, cv2.INPAINT_TELEA)
    return frame_dst, mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Remove Text From Image")
    parser.add_argument("--src", default="./data/data_src/", type=str,
                        help="Image folder source")
    parser.add_argument("--dst", default="./data/data_dst/", type=str,
                        help="Image folder destination")
    parser.add_argument("--ki", default=(5, 3), type=tuple,
                        help="Enlarge mask rule (K, I): K for I iterations")
    parser.add_argument("--exclude_patterns_path", default="./data/exclude_pattern.txt",
                        help="Path to text patterns for exclude")

    args = parser.parse_args()
    folder_src = args.src
    folder_dst = args.dst
    ksize, iterations = args.ki
    exclude_patterns_path = args.exclude_patterns_path
    model_q_path_onnx = "./data/models/places2.pth_Q.onnx"

    if os.path.isfile(exclude_patterns_path):
        with open(exclude_patterns_path, 'r') as f:
            exclude_patterns = ast.literal_eval(''.join(f.readlines()))
    else:
        exclude_patterns = []

    extension_list = ['png', 'jpg', 'jpeg']
    file_list_src = glob(os.path.join(folder_src, '', '**', '*.*')) + glob(os.path.join(folder_src, '', '*.*'))
    file_list_src = [item for item in file_list_src if item.split('.')[-1].lower() in extension_list]
    file_list_dst = [item.replace('\\', '/').replace(folder_src, folder_dst) for item in file_list_src]

    reader = easyocr.Reader(['en'])
    model_inpaint = rt.InferenceSession(model_q_path_onnx)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    for path_src, path_dst in tqdm(zip(file_list_src, file_list_dst), total=len(file_list_dst)):
        assert os.path.abspath(path_src) != os.path.abspath(path_dst)
        frame_src = cv2.imread(path_src, cv2.IMREAD_UNCHANGED)

        if len(frame_src.shape) == 2:
            frame_src = cv2.cvtColor(frame_src, cv2.COLOR_GRAY2BGR)

        frame_src = np.ascontiguousarray(frame_src[..., :3])

        ocr_result = reader.readtext(path_src)

        mask_src = np.full_like(frame_src, 255, dtype=np.uint8)
        for shape, text, confidence in ocr_result:
            if any([re.match(pattern, text) for pattern in exclude_patterns]):
                continue
            cv2.fillPoly(mask_src, [np.array(shape, dtype=np.int32)], 0)

        frame_in = cv2.resize(frame_src, (512, 512))
        mask_in = cv2.resize(mask_src, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_in = cv2.erode(mask_in, kernel, iterations=iterations)
        frame_out = model_inpaint.run(None, {'frame': frame_in, 'mask': mask_in})[0][0]

        frame_out = cv2.resize(frame_out, tuple(reversed(frame_src.shape[:2])))
        alpha = cv2.resize(mask_in, tuple(reversed(frame_src.shape[:2]))) / 255.

        frame_result = (frame_src * alpha + (1 - alpha) * frame_out).astype(np.uint8)

        # frame_show = np.hstack([frame_src, frame_result])

        os.makedirs(os.path.dirname(path_dst), exist_ok=True)
        cv2.imwrite(path_dst, frame_result)
