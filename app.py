import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pandas as pd

# ==========================================
#              設定與模型載入
# ==========================================
st.set_page_config(page_title="AI 手寫數字辨識 (V42 Fix 7)", page_icon="🔢", layout="wide")

MODEL_FILE = "cnn_model_robust.h5"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return tf.keras.models.load_model(MODEL_FILE)
    return None

if not os.path.exists(MODEL_FILE):
    st.error(f"找不到模型檔案: {MODEL_FILE}")
    st.stop()

cnn_model = load_model()

# ==========================================
#              核心演算法
# ==========================================
def center_by_moments_cnn(src):
    img = src.copy()
    m = cv2.moments(img, True)
    if m['m00'] < 0.1: return cv2.resize(img, (28, 28))
    cX, cY = m['m10'] / m['m00'], m['m01'] / m['m00']
    tX, tY = 14.0 - cX, 14.0 - cY
    M = np.float32([[1, 0, tX], [0, 1, tY]])
    return cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def split_touching_digits(roi_binary):
    h, w = roi_binary.shape
    if w / h < 1.2: return [(0, roi_binary)]
    projection = np.sum(roi_binary, axis=0)
    mid_start, mid_end = int(w * 0.25), int(w * 0.75)
    if mid_end <= mid_start: return [(0, roi_binary)]
    split_x = mid_start + np.argmin(projection[mid_start:mid_end])
    if projection[split_x] > (h * 255 * 0.5): return [(0, roi_binary)]
    part1 = roi_binary[:, :split_x]
    part2 = roi_binary[:, split_x:]
    if part1.shape[1] < 5 or part2.shape[1] < 5: return [(0, roi_binary)]
    return [(0, part1), (split_x, part2)]

def analyze_hole_geometry(binary_roi):
    roi_copy = binary_roi.copy()
    contours, hierarchy = cv2.findContours(roi_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None: return 0, None
    valid_holes = []
    h_img, w_img = roi_copy.shape
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1: 
            area = cv2.contourArea(contours[i])
            if area > 15: 
                M = cv2.moments(contours[i])
                if M['m00'] != 0:
                    cy = int(M['m01'] / M['m00'])
                    norm_y = cy / float(h_img)
                    valid_holes.append((area, norm_y))
    if not valid_holes: return 0, None
    valid_holes.sort(key=lambda x: x[0], reverse=True)
    largest_hole_y = valid_holes[0][1]
    return len(valid_holes), largest_hole_y

def process_and_predict(image_bgr, min_area, min_density, min_confidence, proc_mode="adaptive", manual_thresh=127, use_smart_logic=True, show_debug=False):
    result_img = image_bgr.copy()
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if proc_mode == "adaptive":
        binary_proc = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 10)
    elif proc_mode == "manual":
        _, thresh = cv2.threshold(blur, manual_thresh, 255, cv2.THRESH_BINARY_INV)
        binary_proc = thresh
    else: # "otsu"
        if np.mean(gray) > 127:
            flag = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        else:
            flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        _, thresh = cv2.threshold(blur, 0, 255, flag)
        binary_proc = thresh

    binary_proc = cv2.dilate(binary_proc, None, iterations=1)
    
    if show_debug:
        st.image(binary_proc, caption=f"【Debug】二值化影像", width=300)
    
    nb, output, stats_cc, _ = cv2.connectedComponentsWithStats(binary_proc, connectivity=8)
    raw_boxes = sorted([stats_cc[i, :4] for i in range(1, nb)], key=lambda b: b[0])

    rois_to_pred = []
    coords_to_draw = []
    detected_info = []

    for box in raw_boxes:
        x, y, w, h = box
        if x < 5 or y < 5 or (x + w) > binary_proc.shape[1] - 5 or (y + h) > binary_proc.shape[0] - 5: continue
        if h < 20: continue 

        split_results = split_touching_digits(binary_proc[y:y+h, x:x+w])
        
        for offset_x, sub_roi in split_results:
            sh, sw = sub_roi.shape
            if sw == 0 or sh == 0: continue
            
            n_white_pix = cv2.countNonZero(sub_roi)
            box_area = sw * sh
            density = n_white_pix / float(box_area)

            if n_white_pix < min_area:
                if show_debug: cv2.rectangle(result_img, (x+offset_x, y), (x+offset_x+sw, y+sh), (255, 0, 255), 1)
                continue
            if density < min_density:
                if show_debug: cv2.rectangle(result_img, (x+offset_x, y), (x+offset_x+sw, y+sh), (255, 0, 0), 1)
                continue
            
            side = max(sw, sh)
            container = np.zeros((side+40, side+40), dtype=np.uint8)
            offset_y, offset_x_c = 20 + (side-sh)//2, 20 + (side-sw)//2
            container[offset_y:offset_y+sh, offset_x_c:offset_x_c+sw] = sub_roi
            
            final_roi = center_by_moments_cnn(cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA))
            final_roi_norm = np.expand_dims(final_roi.astype('float32') / 255.0, axis=-1)
            
            rois_to_pred.append(final_roi_norm)
            coords_to_draw.append((x + offset_x, y, sw, sh, sub_roi))

    if len(rois_to_pred) > 0:
        predictions = cnn_model.predict(np.array(rois_to_pred), verbose=0)
        
        for i, pred_probs in enumerate(predictions):
            res_id = np.argmax(pred_probs)
            confidence = np.max(pred_probs)
            rx, ry, w, h, roi_original = coords_to_draw[i]
            
            if confidence < min_confidence:
                if show_debug:
                    cv2.rectangle(result_img, (rx, ry), (rx+w, ry+h), (0, 0, 255), 1)
                continue

            display_text = str(res_id)
            color = (0, 255, 0)
            
            # [V42 重點修改] 
            is_corrected = False
            if use_smart_logic:
                num_holes, hole_y = analyze_hole_geometry(roi_original)
                aspect_ratio = w / float(h)
                pixel_count = cv2.countNonZero
