import argparse
import os

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import onnxruntime as rt

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

    frame_data_inference = cv2.warpPerspective(frame_data, M_document_craft, tuple(reversed(shape_document_craft_dst)))

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
    parser.add_argument("--src", default="./data/data_src/", type=str, help="Image folder source")
    parser.add_argument("--dst", default="./data/data_dst/", type=str, help="Image folder destination")

    args = parser.parse_args()
    folder_src = args.src
    folder_dst = args.dst

    extension_list = ['png', 'jpg', 'jpeg']
    file_list_src = glob(os.path.join(folder_src, '', '**', '*.*')) + glob(os.path.join(folder_src, '', '*.*'))
    file_list_src = [item for item in file_list_src if item.split('.')[-1].lower() in extension_list]
    file_list_dst = [item.replace('\\', '/').replace(folder_src, folder_dst) for item in file_list_src]

    for path_src, path_dst in tqdm(zip(file_list_src, file_list_dst), total=len(file_list_dst)):
        assert os.path.abspath(path_src) != os.path.abspath(path_dst)
        frame_src = cv2.imread(path_src, cv2.IMREAD_UNCHANGED)

        if len(frame_src.shape) == 2:
            frame_src = cv2.cvtColor(frame_src, cv2.COLOR_GRAY2BGR)

        frame_src = np.ascontiguousarray(frame_src[..., :3])

        frame_dst, _ = process_image(frame_src)

        os.makedirs(os.path.dirname(path_dst), exist_ok=True)

        cv2.imwrite(path_dst, frame_dst)
