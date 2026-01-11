import streamlit as st
import cv2
import numpy as np
import os
import time
import av
import joblib
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from streamlit_image_coordinates import streamlit_image_coordinates

# --------------------------------------------------------------------------------
# 環境設定與依賴庫配置
# --------------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------------------------------------------------------
# 全域參數設定 (V84 修改處)
# --------------------------------------------------------------------------------
MIN_HEIGHT = 50           
MIN_AREA = 500            
SHRINK_PX = 4             
# [修改 1] 延長穩定時間：從 1.2秒 改為 2.5秒，讓你有時間對準
STABILITY_DURATION = 2.5  
# [修改 2] 放寬移動門檻：從 80 改為 100，容許手部微抖動
MOVEMENT_THRESHOLD = 100   
CONFIDENCE_THRESHOLD = 0.85 
KNN_VERIFY_RANGE = (0.85, 0.95) 
ROI_MARGIN_X = 60         
ROI_MARGIN_Y = 60         
TEXT_Y_OFFSET = 15        

# --------------------------------------------------------------------------------
# 1. 模型載入與初始化模組
# --------------------------------------------------------------------------------
@st.cache_resource
def load_ai_models():
    cnn = None
    if os.path.exists("mnist_cnn.h5"):
        try:
            cnn = load_model("mnist_cnn.h5")
            print("✅ CNN 模型載入成功")
        except:
            print("❌ CNN 模型載入失敗")
    
    knn = None
    knn_path = "knn_model.pkl"
    if os.path.exists(knn_path):
        try:
            knn = joblib.load(knn_path)
            print("✅ KNN 模型載入成功")
        except:
            print("⚠️ KNN 模型損壞，重新訓練...")
    
    if knn is None:
        try:
            (x_train, y_train), _ = mnist.load_data()
            x_flat = x_train.reshape(-1, 784) / 255.0
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(x_flat[:10000], y_train[:10000])
            joblib.dump(knn, knn_path)
            print("✅ KNN 模型訓練完成")
        except Exception as e:
            knn = None
    return cnn, knn

model, knn_model = load_ai_models()

# --------------------------------------------------------------------------------
# 2. 核心影像演算法
# --------------------------------------------------------------------------------
def center_by_moments_cnn(src):
    img = src.copy()
    m = cv2.moments(img, True)
    if m['m00'] < 0.1: return cv2.resize(img, (28, 28))
    cX, cY = m['m10'] / m['m00'], m['m01'] / m['m00']
    tX, tY = 14.0 - cX, 14.0 - cY
    M = np.float32([[1, 0, tX], [0, 1, tY]])
    return cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2: return img
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def is_valid_content(img_bgr):
    if img_bgr is None or img_bgr.size == 0: return False
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_s = np.mean(hsv[:,:,1])
    if mean_s > 60: return False
    if 30 < mean_s <= 60:
        mean_h = np.mean(hsv[:,:,0])
        if (mean_h < 25 or mean_h > 155): return False
    return True

# --------------------------------------------------------------------------------
# 3. 圖片上傳模式專用函式庫
# --------------------------------------------------------------------------------
def detect_image_source(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    extreme_pixels = np.sum((gray < 10) | (gray > 245))
    ratio = extreme_pixels / gray.size
    return "digital" if ratio > 0.5 else "photo"

def merge_overlapping_boxes(boxes):
    if len(boxes) < 2: return boxes
    merged = []
    while len(boxes) > 0:
        curr = boxes.pop(0)
        x1, y1, w1, h1 = curr
        rx1, ry1 = x1 + w1, y1 + h1
        has_overlap = False
        i = 0
        while i < len(boxes):
            next_box = boxes[i]
            x2, y2, w2, h2 = next_box
            rx2, ry2 = x2 + w2, y2 + h2
            pad = 15
            overlap = not ((rx1 + pad) < x2 or (x1 - pad) > rx2 or (ry1 + pad) < y2 or (y1 - pad) > ry2)
            if overlap:
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(rx1, rx2) - new_x
                new_h = max(ry1, ry2) - new_y
                curr = (new_x, new_y, new_w, new_h)
                x1, y1, w1, h1 = curr
                rx1, ry1 = new_x + new_w, new_y + new_h
                boxes.pop(i)
                has_overlap = True
            else:
                i += 1
        if has_overlap: boxes.insert(0, curr)
        else: merged.append(curr)
    return merged

def filter_small_boxes(boxes, img_height, img_width, source_type):
    if not boxes: return []
    total_area = img_width * img_height
    if source_type == "digital":
        kept = [box for box in boxes if (box[2] * box[3]) < (total_area * 0.6) and box[3] > 5]
        return kept
    
    abs_min_h = int(img_height * 0.02)
    valid_h = [b[3] for b in boxes if b[3] > abs_min_h]
    median_h = np.median(valid_h) if valid_h else 0
    kept_boxes = []
    
    for box in boxes:
        w, h = box[2], box[3]
        if (w * h) > (total_area * 0.6) or h < abs_min_h: continue
        aspect = w / float(h)
        if aspect < 0.35 and median_h > 0 and h > (median_h * 0.35): kept_boxes.append(box); continue
        if median_h > 0 and h < (median_h * 0.5): continue
        if source_type == "photo" and h < 65 and 0.7 < aspect < 1.3: continue
        kept_boxes.append(box)
    return kept_boxes

def filter_low_contrast_boxes(boxes, gray_img):
    if not boxes: return []
    flat = np.sort(gray_img.ravel())
    ink_black = np.mean(flat[:int(len(flat)*0.02)])
    paper_bg = np.median(flat)
    threshold = paper_bg - ((paper_bg - ink_black) * 0.6)
    kept_boxes = []
    for box in boxes:
        x, y, w, h = box
        roi = gray_img[y:y+h, x:x+w]
        if roi.size == 0: continue
        roi_flat = np.sort(roi.ravel())
        if np.mean(roi_flat[:max(1, int(len(roi_flat)*0.1))]) <= threshold: kept_boxes.append(box)
    return kept_boxes

def preprocess_for_mnist(roi_binary):
    h, w = roi_binary.shape
    canvas = np.zeros((28, 28), dtype=np.uint8)
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    roi_resized = cv2.resize(roi_binary, (nw, nh), interpolation=cv2.INTER_AREA)
    y_off, x_off = (28 - nh) // 2, (28 - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = roi_resized
    _, canvas = cv2.threshold(canvas, 10, 255, cv2.THRESH_BINARY)
    M = cv2.moments(canvas)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, 14-cx], [0, 1, 14-cy]]), (28, 28))
    return cv2.dilate(canvas, None, iterations=1)

def try_add_manual_box(click_x, click_y, binary_img, model):
    h, w = binary_img.shape
    if not (0 <= click_x < w and 0 <= click_y < h): return None, "❌ 點擊位置超出範圍"
    cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_contour = None
    for c in cnts:
        if cv2.pointPolygonTest(c, (click_x, click_y), False) >= 0:
            target_contour = c; break
    if target_contour is None: return None, "⚠️ 沒點到東西 (請點擊文字筆跡的黑色區域)"
    
    bx, by, bw, bh = cv2.boundingRect(target_contour)
    if bw < 5 or bh < 10: return None, "⚠️ 區域太小，視為雜訊"
    
    roi = binary_img[by:by+bh, bx:bx+bw]
    roi_processed = preprocess_for_mnist(roi)
    input_data = roi_processed.reshape(1, 28, 28, 1).astype('float32') / 255.0
    pred = model.predict(input_data, verbose=0)[0]
    res_id = np.argmax(pred)
    conf = float(pred[res_id])
    return {"rect": (bx, by, bw, bh), "label": int(res_id), "conf": conf}, f"✅ 手動加入成功：數字 {res_id}"

# --------------------------------------------------------------------------------
# 4. 手寫板模式專用邏輯
# --------------------------------------------------------------------------------
def get_edge_distance(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    rx1, ry1 = x1 + w1, y1 + h1
    rx2, ry2 = x2 + w2, y2 + h2
    dx = max(0, max(x1 - rx2, x2 - rx1))
    dy = max(0, max(y1 - ry2, y2 - ry1))
    return np.sqrt(dx*dx + dy*dy)

def merge_boxes_logic(contours, merge_dist_limit, time_limit):
    if 'box_cache' not in st.session_state: st.session_state['box_cache'] = []
    current_time = time.time()
    raw_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    current_boxes_with_time = []
    for r_new in raw_boxes:
        assigned_time = current_time
        best_overlap = 0
        for old_item in st.session_state['box_cache']:
            ox, oy, ow, oh = old_item['rect']
            ix = max(r_new[0], ox)
            iy = max(r_new[1], oy)
            iw = min(r_new[0]+r_new[2], ox+ow) - ix
            ih = min(r_new[1]+r_new[3], oy+oh) - iy
            if iw > 0 and ih > 0:
                overlap = iw * ih
                if overlap > best_overlap: best_overlap = overlap; assigned_time = old_item['time']
        current_boxes_with_time.append({'rect': r_new, 'time': assigned_time})

    has_merged = True
    while has_merged:
        has_merged = False
        new_list = []
        skip_indices = set()
        for i in range(len(current_boxes_with_time)):
            if i in skip_indices: continue
            merged_rect = current_boxes_with_time[i]['rect']
            merged_time = current_boxes_with_time[i]['time']
            for j in range(i + 1, len(current_boxes_with_time)):
                if j in skip_indices: continue
                b1, b2 = current_boxes_with_time[i], current_boxes_with_time[j]
                dist = get_edge_distance(merged_rect, b2['rect'])
                time_diff = abs(merged_time - b2['time'])
                if dist < merge_dist_limit and time_diff < time_limit:
                    x1, y1 = merged_rect[0], merged_rect[1]
                    x2, y2 = merged_rect[0] + merged_rect[2], merged_rect[1] + merged_rect[3]
                    bx1, by1 = b2['rect'][0], b2['rect'][1]
                    bx2, by2 = b2['rect'][0] + b2['rect'][2], b2['rect'][1] + b2['rect'][3]
                    merged_rect = (min(x1, bx1), min(y1, by1), max(x2, bx2)-min(x1, bx1), max(y2, by2)-min(y1, by1))
                    merged_time = max(merged_time, b2['time'])
                    skip_indices.add(j)
                    has_merged = True
            new_list.append({'rect': merged_rect, 'time': merged_time})
        if has_merged: current_boxes_with_time = new_list
        else: break

    st.session_state['box_cache'] = current_boxes_with_time
    final_output = []
    for item in current_boxes_with_time:
        x, y, w, h = item['rect']
        final_output.append({'rect': (x,y,w,h), 'center': (x + w//2, y + h//2)})
    return final_output

def update_tracker_from_boxes(box_items):
    current_items = box_items
    used_current_indices = set()
    new_tracker_state = {}
    if 'tracker_state' in st.session_state:
        for old_id, old_center in st.session_state['tracker_state'].items():
            min_dist = 9999
            match_idx = -1
            for i, item in enumerate(current_items):
                if i in used_current_indices: continue
                dist = np.hypot(item['center'][0]-old_center[0], item['center'][1]-old_center[1])
                if dist < 50 and dist < min_dist: min_dist = dist; match_idx = i
            if match_idx != -1:
                current_items[match_idx]['id'] = old_id
                used_current_indices.add(match_idx)
                new_tracker_state[old_id] = current_items[match_idx]['center']
    for i, item in enumerate(current_items):
        if 'id' not in item:
            item['id'] = st.session_state['next_id']
            st.session_state['next_id'] += 1
            new_tracker_state[item['id']] = item['center']
    st.session_state['tracker_state'] = new_tracker_state
    current_items.sort(key=lambda x: x['id'])
    return current_items

# --------------------------------------------------------------------------------
# 5. UI 介面輔助工具
# --------------------------------------------------------------------------------
def get_responsive_layout(ratios):
    if st.session_state.get('last_device_mode') and "手機" in st.session_state['last_device_mode']:
        return [st.container() for _ in ratios]
    else: return st.columns(ratios)

def get_bar_html(confidence, is_uncertain=False):
    percent = min(int(confidence * 100), 100)
    color = "#ff9f43" if is_uncertain else "#2ecc71" if confidence > 0.95 else "#f1c40f"
    return f"""<div style="display:flex;align-items:center;margin-top:4px;"><div style="width:50%;height:8px;background:#444;border-radius:4px;overflow:hidden;"><div style="width:{percent}%;height:100%;background:{color};"></div></div><span style="margin-left:8px;font-size:0.8em;color:{color};">{percent}%</span></div>"""

# --------------------------------------------------------------------------------
# 6. WebRTC 影像處理核心 (鏡頭模式用) - 慢速穩定版
# --------------------------------------------------------------------------------
class HandwriteProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.knn = knn_model
        self.last_boxes = []
        self.stability_start_time = None
        self.frozen = False
        self.frozen_frame = None
        self.detected_count = 0
        self.ui_results = []
        self.frame_counter = 0
        
        # [修改 3] 提高跳幀數：從 4 改為 12，每秒約只偵測 2-3 次，大幅降低閃爍感
        self.skip_rate = 12          
        
        self.cached_rois = []
        self.session_start_time = time.time()
        self.warmup_duration = 1.5

    def resume(self):
        self.frozen = False
        self.stability_start_time = None
        self.last_boxes = []
        self.ui_results = [] 
        self.frame_counter = 0
        self.session_start_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not hasattr(self, 'session_start_time') or self.session_start_time is None:
            self.session_start_time = time.time()
            self.warmup_duration = 1.5
        is_warming_up = (time.time() - self.session_start_time) < self.warmup_duration

        if self.frozen and self.frozen_frame is not None:
            return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")
        
        display_img = img.copy()
        h_f, w_f = img.shape[:2]
        roi_rect = [ROI_MARGIN_X, ROI_MARGIN_Y, w_f - 2*ROI_MARGIN_X, h_f - 2*ROI_MARGIN_Y]
        roi_color = (0, 0, 255) if is_warming_up else (255, 0, 0)
        cv2.rectangle(display_img, (roi_rect[0], roi_rect[1]), (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), roi_color, 2)

        self.frame_counter += 1
        if not (self.frame_counter % self.skip_rate == 0):
            if len(self.cached_rois) > 0:
                for (dx, dy, dw, dh, txt, box_color, box_thick) in self.cached_rois:
                    cv2.rectangle(display_img, (dx, dy), (dx+dw, dy+dh), box_color, box_thick)
                    cv2.putText(display_img, txt, (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if is_warming_up: cv2.putText(display_img, "Initializing...", (20, h_f - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(display_img, format="bgr24")
        
        roi_img = img[roi_rect[1]:roi_rect[1]+roi_rect[3], roi_rect[0]:roi_rect[0]+roi_rect[2]]
        if roi_img.size == 0: return av.VideoFrame.from_ndarray(display_img, format="bgr24")

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 18)
        binary_proc = cv2.dilate(thresh, None, iterations=2)
        
        contours, hierarchy = cv2.findContours(binary_proc, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = []
        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] == -1:
                    area = cv2.contourArea(cnt)
                    if area > MIN_AREA:
                        x, y, w, h = cv2.boundingRect(cnt)
                        has_hole = hierarchy[0][i][2] != -1
                        valid_boxes.append({"box": (x, y, w, h), "has_hole": has_hole, "aspect_ratio": w / float(h)})
        
        valid_boxes = sorted(valid_boxes, key=lambda b: b["box"][0])
        batch_rois, batch_info, raw_boxes_for_stability = [], [], []
        self.cached_rois = []

        for item in valid_boxes:
            x, y, w, h = item["box"]
            rx, ry = x + roi_rect[0], y + roi_rect[1]
            if x < 5 or y < 5 or (x+w) > binary_proc.shape[1]-5 or (y+h) > binary_proc.shape[0]-5: continue
            if h < MIN_HEIGHT: continue
            roi_color_check = display_img[ry:ry+h, rx:rx+w]
            if not is_valid_content(roi_color_check): continue
            
            raw_boxes_for_stability.append(item)
            roi_single = deskew(binary_proc[y:y+h, x:x+w])
            side = max(w, h); padding = int(side * 0.2); container_size = side + padding * 2
            container = np.zeros((container_size, container_size), dtype=np.uint8)
            offset_y = (container_size - h) // 2; offset_x = (container_size - w) // 2
            roi_single = cv2.resize(roi_single, (w, h)) 
            container[offset_y:offset_y+h, offset_x:offset_x+w] = roi_single
            roi_resized = cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA)
            roi_norm = roi_resized.astype('float32') / 255.0
            batch_rois.append(roi_norm.reshape(28, 28, 1))
            batch_info.append({"coords": (rx, ry, w, h), "has_hole": item["has_hole"], "aspect": item["aspect_ratio"], "flat_input": roi_norm.reshape(1, 784)})
            
        detected_count = 0
        detected_something = False
        current_frame_text_results = []
        valid_ui_counter = 1

        if len(batch_rois) > 0 and self.model is not None:
            detected_something = True
            try:
                batch_input = np.stack(batch_rois)
                predictions = self.model.predict(batch_input, verbose=0)
                for i, pred in enumerate(predictions):
                    top_indices = pred.argsort()[-3:][::-1]
                    res_id = top_indices[0]; confidence = pred[res_id]
                    if confidence < CONFIDENCE_THRESHOLD: continue 

                    info = batch_info[i]
                    rx, ry, w, h = info["coords"]
                    aspect = info["aspect"]
                    has_hole = info["has_hole"]
                    
                    if res_id == 1 and aspect > 0.6: res_id = 7
                    elif res_id == 7 and aspect < 0.25: res_id = 1
                    if res_id == 7 and has_hole: res_id = 9
                    if res_id == 9 and not has_hole and confidence < 0.95: res_id = 7
                    if res_id == 0 and aspect < 0.5: res_id = 1
                    
                    final_label_str = str(res_id)
                    verify_msg = ""
                    is_knned = False
                    if self.knn is not None and KNN_VERIFY_RANGE[0] <= confidence <= KNN_VERIFY_RANGE[1]:
                        try:
                            knn_pred = self.knn.predict(info["flat_input"])[0]
                            if knn_pred != res_id: final_label_str = str(res_id); verify_msg = f" ⚠️ KNN: {knn_pred}"; is_knned = True
                        except: pass
                    
                    if is_warming_up: box_color = (0, 0, 255); box_thickness = 1
                    elif is_knned: box_color = (0, 165, 255); box_thickness = 2
                    else: box_color = (0, 255, 0); box_thickness = 2

                    draw_x = rx + SHRINK_PX; draw_y = ry + SHRINK_PX
                    draw_w = max(1, w - (SHRINK_PX * 2)); draw_h = max(1, h - (SHRINK_PX * 2))
                    cv2.rectangle(display_img, (draw_x, draw_y), (draw_x+draw_w, draw_y+draw_h), box_color, box_thickness)
                    text_label = f"#{valid_ui_counter}"
                    cv2.putText(display_img, text_label, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    self.cached_rois.append((draw_x, draw_y, draw_w, draw_h, text_label, box_color, box_thickness))
                    
                    info_text = f"**#{valid_ui_counter}**: 數字 `{res_id}` (信心: {int(confidence*100)}%){verify_msg}"
                    current_frame_text_results.append(info_text)
                    detected_count += 1
                    valid_ui_counter += 1
            except: pass

        self.detected_count = detected_count
        if detected_something: self.ui_results = current_frame_text_results

        if len(raw_boxes_for_stability) == 0: self.stability_start_time = None
        elif len(self.last_boxes) == 0: self.last_boxes = raw_boxes_for_stability; self.stability_start_time = time.time()
        else:
            total_movement = 0
            for curr_box in raw_boxes_for_stability:
                c_x, c_y, _, _ = curr_box["box"]
                min_dist = 99999
                for last_box in self.last_boxes:
                    l_x, l_y, _, _ = last_box["box"]
                    dist = abs(c_x - l_x) + abs(c_y - l_y)
                    if dist < min_dist: min_dist = dist
                if min_dist < 30: total_movement += min_dist
                else: total_movement += 20 
            
            count_diff = abs(len(raw_boxes_for_stability) - len(self.last_boxes))
            total_movement += count_diff * 30 
            self.last_boxes = raw_boxes_for_stability

            if total_movement < MOVEMENT_THRESHOLD and not is_warming_up:
                if self.stability_start_time is None: self.stability_start_time = time.time()
                elapsed = time.time() - self.stability_start_time
                progress = min(elapsed / STABILITY_DURATION, 1.0)
                
                bar_y = h_f - 20 
                bar_w = int(600 * progress)
                color = (0, 255, 255) if progress < 1.0 else (0, 255, 0)
                cv2.rectangle(display_img, (20, bar_y - 15), (20 + bar_w, bar_y), color, -1)
                cv2.rectangle(display_img, (20, bar_y - 15), (w_f - 20, bar_y), (255, 255, 255), 2)
                
                if elapsed >= STABILITY_DURATION and detected_something:
                    self.frozen = True
                    self.frozen_frame = display_img.copy()
            else:
                self.stability_start_time = time.time()
                if is_warming_up: cv2.putText(display_img, "Initializing...", (20, h_f - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

# --------------------------------------------------------------------------------
# 7. Streamlit 介面與入口閘門
# --------------------------------------------------------------------------------
st.set_page_config(page_title="手寫辨識", page_icon="📝", layout="wide")

if 'stats' not in st.session_state: st.session_state['stats'] = {'camera': {'total': 0, 'correct': 0}, 'handwriting': {'total': 0, 'correct': 0}, 'upload': {'total': 0, 'correct': 0}}
if 'history' not in st.session_state: st.session_state['history'] = {'camera': [], 'handwriting': [], 'upload': []} 
if 'tracker_state' not in st.session_state: st.session_state['tracker_state'] = {}
if 'next_id' not in st.session_state: st.session_state['next_id'] = 1
if 'hw_display_list' not in st.session_state: st.session_state['hw_display_list'] = []
if 'hw_result_img' not in st.session_state: st.session_state['hw_result_img'] = None
if 'hw_result_count' not in st.session_state: st.session_state['hw_result_count'] = 0
if 'box_cache' not in st.session_state: st.session_state['box_cache'] = [] 
if 'upload_display_list' not in st.session_state: st.session_state['upload_display_list'] = []
if 'upload_result_img' not in st.session_state: st.session_state['upload_result_img'] = None
if 'upload_result_count' not in st.session_state: st.session_state['upload_result_count'] = 0
if 'last_uploaded_file_id' not in st.session_state: st.session_state['last_uploaded_file_id'] = None
if 'ignored_boxes' not in st.session_state: st.session_state['ignored_boxes'] = set()
if 'manual_boxes' not in st.session_state: st.session_state['manual_boxes'] = []
if 'input_key' not in st.session_state: st.session_state['input_key'] = 0

DEVICE_PC = "🖥️ 電腦版 (並排佈局)"
DEVICE_MOBILE = "📱 手機版 (垂直佈局)"

if 'last_device_mode' not in st.session_state:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>👋 歡迎使用手寫數字辨識系統</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>請選擇您的操作裝置以最佳化介面</h3>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        c_pc, c_mo = st.columns(2)
        with c_pc:
            if st.button("🖥️ 電腦 / 平板", use_container_width=True, type="primary"):
                st.session_state['last_device_mode'] = DEVICE_PC
                st.rerun()
        with c_mo:
            if st.button("📱 手機", use_container_width=True, type="primary"):
                st.session_state['last_device_mode'] = DEVICE_MOBILE
                st.rerun()
    st.stop()

device_mode = st.session_state['last_device_mode']
is_mobile = "手機" in device_mode

with st.sidebar:
    st.title("🎛️ 控制台")
    st.markdown("### 📱 顯示設定")
    st.info(f"目前模式：{device_mode}")
    if st.button("🔄 重新選擇裝置"):
        del st.session_state['last_device_mode']
        st.session_state['hw_result_img'] = None
        st.session_state['hw_display_list'] = []
        st.session_state['hw_result_count'] = 0
        st.session_state['tracker_state'] = {}
        st.session_state['box_cache'] = []
        st.session_state['canvas_key'] = f"canvas_{time.time()}"
        st.rerun()
    st.divider()
    app_mode = st.radio("模式選擇", ["📷 鏡頭模式 (Live)", "🎨 手寫板模式", "📁 圖片上傳模式"], index=1)
    st.divider()
    
    st.markdown("### 📷 鏡頭成績")
    c_total = st.session_state['stats']['camera']['total']
    c_correct = st.session_state['stats']['camera']['correct']
    c_acc = (c_correct / c_total * 100) if c_total > 0 else 0.0
    col_c1, col_c2 = st.columns(2)
    with col_c1: st.metric("總數", c_total)
    with col_c2: st.metric("正確", c_correct)
    st.metric("鏡頭準確率", f"{c_acc:.1f}%")
    
    col_undo_c, col_reset_c = st.columns(2)
    with col_undo_c:
        if st.button("↩️ 復原", key="undo_cam"):
            if st.session_state['history']['camera']:
                last_entry = st.session_state['history']['camera'].pop()
                st.session_state['stats']['camera']['total'] -= last_entry['total']
                st.session_state['stats']['camera']['correct'] -= last_entry['correct']
                st.rerun()
    with col_reset_c:
        if st.button("🗑️ 重置", key="reset_cam"):
            st.session_state['stats']['camera'] = {'total': 0, 'correct': 0}
            st.session_state['history']['camera'] = []
            st.rerun()

    st.divider()
    st.markdown("### 🎨 手寫成績")
    h_total = st.session_state['stats']['handwriting']['total']
    h_correct = st.session_state['stats']['handwriting']['correct']
    h_acc = (h_correct / h_total * 100) if h_total > 0 else 0.0
    col_h1, col_h2 = st.columns(2)
    with col_h1: st.metric("總數", h_total)
    with col_h2: st.metric("正確", h_correct)
    st.metric("手寫準確率", f"{h_acc:.1f}%")

    col_undo_h, col_reset_h = st.columns(2)
    with col_undo_h:
        if st.button("↩️ 復原", key="undo_hw"):
            if st.session_state['history']['handwriting']:
                last_entry = st.session_state['history']['handwriting'].pop()
                st.session_state['stats']['handwriting']['total'] -= last_entry['total']
                st.session_state['stats']['handwriting']['correct'] -= last_entry['correct']
                st.rerun()
    with col_reset_h:
        if st.button("🗑️ 重置", key="reset_hw"):
            st.session_state['stats']['handwriting'] = {'total': 0, 'correct': 0}
            st.session_state['history']['handwriting'] = []
            st.session_state['tracker_state'] = {}
            st.session_state['next_id'] = 1
            st.rerun()

    st.divider()
    st.markdown("### 📁 上傳成績")
    u_total = st.session_state['stats']['upload']['total']
    u_correct = st.session_state['stats']['upload']['correct']
    u_acc = (u_correct / u_total * 100) if u_total > 0 else 0.0
    col_u1, col_u2 = st.columns(2)
    with col_u1: st.metric("總數", u_total)
    with col_u2: st.metric("正確", u_correct)
    st.metric("上傳準確率", f"{u_acc:.1f}%")

    col_undo_u, col_reset_u = st.columns(2)
    with col_undo_u:
        if st.button("↩️ 復原", key="undo_up"):
            if st.session_state['history']['upload']:
                last_entry = st.session_state['history']['upload'].pop()
                st.session_state['stats']['upload']['total'] -= last_entry['total']
                st.session_state['stats']['upload']['correct'] -= last_entry['correct']
                st.rerun()
    with col_reset_u:
        if st.button("🗑️ 重置", key="reset_up"):
            st.session_state['stats']['upload'] = {'total': 0, 'correct': 0}
            st.session_state['history']['upload'] = []
            st.session_state['upload_display_list'] = []
            st.session_state['upload_result_img'] = None
            st.session_state['upload_result_count'] = 0
            st.rerun()

st.title("📝 手寫數字辨識系統")

if model is None: st.error("❌ 找不到模型！"); st.stop()

if app_mode == "📷 鏡頭模式 (Live)":
    with st.expander("📖 鏡頭模式指南(請點開)", expanded=False):
        st.markdown("""
        1. **對準鏡頭**：請將寫有數字的紙張平穩置於鏡頭前。
        2. **保持穩定**：當畫面偵測到數字且畫面穩定時，下方 **藍條** 會開始集氣。
        3. **自動抓拍**：集氣滿後畫面會自動 **凍結 (Captured)** 並顯示辨識結果。
        4. **確認成績**：確認無誤後，於右側輸入正確數量並上傳成績。
        5. 如果沒偵測到可能是光線問題或筆跡太細
        6. 畫面上顯示的是序號，想知道判斷結果請按📋 顯示詳情
        """)
    
    layout_containers = get_responsive_layout([2, 1])
    col_cam = layout_containers[0]
    col_data = layout_containers[1]

    with col_cam:
        ctx = webrtc_streamer(
            key="handwrite-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=HandwriteProcessor,
            media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

    with col_data:
        st.markdown("### 📊 詳細數據")
        st.caption("請等待畫面出現 Captured 後，按下方按鈕更新數據")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📋 顯示詳情", type="secondary", use_container_width=True):
                if ctx.video_processor and ctx.video_processor.frozen:
                    results = ctx.video_processor.ui_results
                    if results:
                        st.success(f"共偵測到 {len(results)} 個數字")
                        st.session_state['last_cam_detected'] = len(results)
                        for line in results: st.markdown(line, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ 畫面凍結了，但沒有偵測到數字。")
                        st.session_state['last_cam_detected'] = 0
                else:
                    st.info("⏳ 請先等待鏡頭畫面抓拍凍結 (Captured)...")

        with col_btn2:
            if st.button("🔄 重新攝影", type="primary", use_container_width=True):
                if ctx.video_processor: ctx.video_processor.resume()
                st.session_state['last_cam_detected'] = 0
                st.rerun()

        st.write("---")
        manual_score = st.number_input("✍️ 輸入正確數量", min_value=0, value=0, key=f"score_input_{st.session_state['input_key']}")
        st.write("##") 
        if st.button("💾 上傳成績並繼續", type="primary", use_container_width=True):
            total_add = st.session_state.get('last_cam_detected', 0)
            if total_add > 0 and manual_score > total_add:
                st.error(f"❌ 錯誤：輸入數值 ({manual_score}) 超過偵測總數 ({total_add})")
            else:
                if ctx.video_processor: ctx.video_processor.resume()
                if total_add == 0: total_add = manual_score
                if manual_score > 0:
                    st.session_state['stats']['camera']['total'] += total_add
                    st.session_state['stats']['camera']['correct'] += manual_score
                    st.session_state['history']['camera'].append({'total': total_add, 'correct': manual_score})
                    st.toast(f"✅ 鏡頭模式：已記錄 (總數{total_add}/正確{manual_score})")
                    time.sleep(0.5)
                    st.session_state['input_key'] += 1
                st.rerun()

elif app_mode == "🎨 手寫板模式":
    with st.expander("📖 手寫模式指南(請點開)", expanded=False):
        st.markdown("""
        * **書寫**：在黑色畫布區直接用滑鼠或手指書寫數字。
        * **工具**：左側可切換 **✏️ 畫筆** 或 **🧽 橡皮擦**。
        * **清除**：按「🗑️ 清除」可重置畫布與計數。
        * 信心度低於85不會記錄
        """)
    
    if is_mobile: c_canvas = st.container(); c_res = st.container()
    else: c_canvas, c_res = st.columns([3, 2])

    with c_res:
        st.markdown("### 👁️ 結果")
        res_ph = st.empty()
        if st.session_state['hw_result_img'] is not None: res_ph.image(st.session_state['hw_result_img'], channels="BGR", use_container_width=True)
        else: res_ph.info("請在畫布書寫")

    with c_canvas:
        c_tool, c_clear = st.columns([2, 1])
        with c_tool: tool_mode = st.radio("🖊️ 工具", ["✏️ 畫筆", "🧽 橡皮擦"], horizontal=True, label_visibility="collapsed")
        with c_clear:
            if st.button("🗑️ 清除", use_container_width=True):
                st.session_state['canvas_key'] = f"canvas_{time.time()}"
                st.session_state['tracker_state'] = {}
                st.session_state['box_cache'] = [] 
                st.session_state['next_id'] = 1
                st.session_state['hw_display_list'] = []
                st.session_state['hw_result_img'] = None
                st.session_state['hw_result_count'] = 0
                st.rerun()

        merge_dist = 60       
        erosion_iter = 0      
        dilation_iter = 2     
        hw_min_area = 50

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=15 if tool_mode == "✏️ 畫筆" else 40,
            stroke_color="#FFFFFF" if tool_mode == "✏️ 畫筆" else "#000000",
            background_color="#000000",
            height=400 if not is_mobile else 230,
            width=850 if not is_mobile else 340,
            drawing_mode="freedraw",
            key=st.session_state.get('canvas_key', 'canvas_0'),
            display_toolbar=False,
            update_streamlit=True, 
        )

        if canvas_result.image_data is not None:
            img_data = canvas_result.image_data.astype(np.uint8)
            if np.max(img_data) > 0:
                if img_data.shape[2] == 4: img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                else: img_bgr = img_data.copy()
                
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_proc = thresh

                if erosion_iter > 0:
                    kernel = np.ones((3,3), np.uint8)
                    binary_proc = cv2.erode(binary_proc, kernel, iterations=erosion_iter)
                if dilation_iter > 0:
                    binary_proc = cv2.dilate(binary_proc, None, iterations=dilation_iter)

                contours, _ = cv2.findContours(binary_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                merged_items = merge_boxes_logic(contours, merge_dist_limit=merge_dist, time_limit=1.0)
                tracked_items = update_tracker_from_boxes(merged_items)
                
                draw_img = img_bgr.copy()
                batch_rois = []
                final_results_list = []
                valid_items = []
                
                for item in tracked_items:
                    x, y, w, h = item['rect']
                    if w * h < hw_min_area: continue
                    roi = binary_proc[y:y+h, x:x+w]
                    side = max(w, h)
                    pad = 40
                    container = np.zeros((side+pad, side+pad), dtype=np.uint8)
                    oy, ox = (side+pad-h)//2, (side+pad-w)//2
                    container[oy:oy+h, ox:ox+w] = roi
                    roi_ready = cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA)
                    final_roi = center_by_moments_cnn(roi_ready)
                    batch_rois.append(final_roi.astype('float32') / 255.0)
                    valid_items.append(item)

                detected_count = 0
                if len(batch_rois) > 0:
                    inputs = np.array(batch_rois).reshape(-1, 28, 28, 1)
                    preds = model.predict(inputs, verbose=0)
                    ui_idx = 1
                    for i, pred in enumerate(preds):
                        item = valid_items[i]
                        x, y, w, h = item['rect']
                        top_idx = pred.argsort()[-1]
                        conf = pred[top_idx]
                        
                        if conf < CONFIDENCE_THRESHOLD: continue
                        
                        dx, dy = x + SHRINK_PX, y + SHRINK_PX
                        dw, dh = max(1, w - 2*SHRINK_PX), max(1, h - 2*SHRINK_PX)
                        cv2.rectangle(draw_img, (dx, dy), (dx+dw, dy+dh), (0, 255, 0), 2)
                        text_y = y - 10 if y > 25 else y + 30
                        cv2.putText(draw_img, str(top_idx), (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        
                        final_results_list.append(f"<div><strong>#{ui_idx}</strong>: 數字 {top_idx} {get_bar_html(conf)}</div>")
                        detected_count += 1
                        ui_idx += 1

                res_ph.image(draw_img, channels="BGR", use_container_width=True)
                if final_results_list:
                    c_canvas.write("---")
                    cols = c_canvas.columns(2)
                    for idx, line in enumerate(final_results_list):
                        cols[idx%2].markdown(line, unsafe_allow_html=True)
                
                st.session_state['hw_result_img'] = draw_img
                st.session_state['hw_result_count'] = detected_count

elif app_mode == "📁 圖片上傳模式":
    with st.expander("📖 圖片上傳功能指南 (請點開)", expanded=True):
        st.markdown("""
        **1. 基本操作**
        * 點擊 **Browse files** 上傳圖片，或選擇範例圖片。
        * 系統會自動框選偵測到的數字 (綠框或橘框)。
        
        **2. 編輯模式 (修正錯誤用)**
        * 開啟圖片下方的 **「🗑️ 啟用編輯模式」** 開關。
        * **刪除誤判**：直接點擊畫面上的 **綠框** 或 **紫框** 即可刪除。
        * **手動補點**：若有數字沒被抓到，請點擊該數字的 **黑色筆跡處**，系統會強制加入辨識 (紫框)。
        * 若點了沒反應可考慮將圖片縮放後再點一次
        """)
    
    if 'ignored_boxes' not in st.session_state: st.session_state['ignored_boxes'] = set()
    if 'manual_boxes' not in st.session_state: st.session_state['manual_boxes'] = []

    layout_containers = get_responsive_layout([3, 1])
    col_up_left = layout_containers[0]
    col_up_right = layout_containers[1]
    
    with col_up_left:
        c_u1, c_u2 = st.columns([0.6, 0.4])
        with c_u1: uploaded_file = st.file_uploader("請上傳圖片 (JPG, PNG)", type=['png', 'jpg', 'jpeg'])
        with c_u2:
            st.write("##")
            example_choice = st.selectbox("或使用範例圖片", ["請選擇...", "範例 1 (手寫)", "範例 2 (手寫)", "範例 3 (小畫家)", "範例 4 (非數字類)"])
            if st.button("🔄 重置所有忽略/手動框", use_container_width=True):
                st.session_state['ignored_boxes'] = set()
                st.session_state['manual_boxes'] = [] 
                st.rerun()

        img, source_id = None, None
        if example_choice != "請選擇...":
            ex_map = {"範例 1 (手寫)": "examples/ex1.jpg", "範例 2 (手寫)": "examples/ex2.jpg", "範例 3 (小畫家)": "examples/ex3.png", "範例 4 (非數字類)": "examples/ex4.jpg"}
            path = ex_map.get(example_choice)
            if os.path.exists(path): img, source_id = cv2.imread(path), path
            else: st.error(f"找不到檔案: {path}")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img, source_id = cv2.imdecode(file_bytes, 1), uploaded_file.file_id
        
        if source_id != st.session_state.get('last_uploaded_file_id'):
            st.session_state['ignored_boxes'] = set()
            st.session_state['manual_boxes'] = [] 

        if img is not None:
            st.session_state['last_uploaded_file_id'] = source_id
            source_type = detect_image_source(img)
            display_img, gray = img.copy(), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if source_type == "photo":
                thresh = cv2.adaptiveThreshold(cv2.bilateralFilter(gray, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 12)
                binary_proc = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                min_area_limit = 10 
            else:
                _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
                binary_proc = cv2.dilate(thresh, None, iterations=2)
                binary_proc = cv2.morphologyEx(binary_proc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3)))
                min_area_limit = 5

            cnts, _ = cv2.findContours(binary_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area_limit]
            sized_boxes = filter_small_boxes(merge_overlapping_boxes(raw_boxes), img.shape[0], img.shape[1], source_type)
            final_boxes = filter_low_contrast_boxes(sized_boxes, gray) if source_type == "photo" else sized_boxes

            batch_rois, batch_info = [], []
            for (x, y, w, h) in final_boxes:
                roi = binary_proc[y:y+h, x:x+w]
                if source_type == "photo" and h < 150: 
                    try: roi = deskew(roi)
                    except: pass 
                f_norm = preprocess_for_mnist(roi)
                
                has_hole = False
                c_sub, h_sub = cv2.findContours(f_norm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if h_sub is not None:
                    for idx, cc in enumerate(c_sub):
                        if h_sub[0][idx][3] != -1 and cv2.contourArea(cc) > 5: has_hole = True; break
                
                batch_rois.append(f_norm.reshape(28, 28, 1).astype('float32') / 255.0)
                batch_info.append({"rect": (x, y, w, h), "has_hole": has_hole, "aspect": w/float(h), "flat": f_norm.reshape(1, 784).astype('float32') / 255.0})

            results_text, v_count = [], 1
            all_boxes_data = [] 
            scale = max(1.0, img.shape[1] / 800.0)

            if batch_rois:
                preds = model.predict(np.stack(batch_rois), verbose=0)
                comb = sorted(list(zip(preds, batch_info)), key=lambda x: x[1]["rect"][0])
                
                for pred, info in comb:
                    bx, by, bw, bh = info["rect"]
                    box_id = f"{bx}_{by}_{bw}_{bh}"
                    res_id = np.argmax(pred); conf = pred[res_id]
                    d_thr = 0.3 if source_type == "digital" else CONFIDENCE_THRESHOLD
                    if info["rect"][3] > 150: d_thr = 0.5

                    all_boxes_data.append({"rect": (bx, by, bw, bh), "id": box_id, "conf": conf, "thr": d_thr})

                    is_ignored = box_id in st.session_state['ignored_boxes']
                    if is_ignored:
                        cv2.rectangle(display_img, (bx, by), (bx+bw, by+bh), (128, 128, 128), 2)
                        cv2.line(display_img, (bx, by), (bx+bw, by+bh), (128, 128, 128), 2)
                        cv2.line(display_img, (bx+bw, by), (bx, by+bh), (128, 128, 128), 2)
                        continue

                    if conf < d_thr: continue
                    
                    if res_id == 7 and info["aspect"] < 0.25: res_id = 1
                    if res_id == 1 and info["has_hole"]: res_id = 0
                    if source_type == "digital" and info["aspect"] < 0.2: res_id = 1
                    
                    color, extra_msg, is_uncertain = (0, 255, 0), "", False
                    if knn_model is not None and KNN_VERIFY_RANGE[0] <= conf <= 0.99:
                        try:
                            k_res = knn_model.predict(info["flat"])[0]
                            if k_res != res_id: extra_msg = f" (KNN: {k_res})"; is_uncertain = True; color = (0, 165, 255)
                        except: pass
                    
                    cv2.rectangle(display_img, (bx, by), (bx+bw, by+bh), color, max(2, int(3*scale)))
                    cv2.putText(display_img, str(res_id), (bx, by - TEXT_Y_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 1.0*scale, (0, 0, 255), max(2, int(3*scale)))
                    results_text.append(f"<div><strong>#{v_count}</strong>: {res_id} {extra_msg} {get_bar_html(conf, is_uncertain)}</div>")
                    v_count += 1
            
            if 'manual_boxes' in st.session_state:
                for mbox in st.session_state['manual_boxes']:
                    bx, by, bw, bh = mbox['rect']
                    lbl = mbox.get('label', mbox.get('digit', '?'))
                    conf = mbox['conf']
                    cv2.rectangle(display_img, (bx, by), (bx+bw, by+bh), (255, 0, 255), max(2, int(3*scale)))
                    cv2.putText(display_img, str(lbl), (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (255, 0, 255), max(2, int(3*scale)))
                    bar_html = get_bar_html(conf, is_uncertain=True)
                    results_text.append(f"<div><strong>#{v_count} (手動)</strong>: {lbl} {bar_html}</div>")
                    v_count += 1

            st.session_state['upload_result_img'] = display_img
            st.session_state['upload_display_list'] = results_text
            st.session_state['upload_result_count'] = v_count - 1

        if st.session_state['upload_result_img'] is not None:
            st.write("---") 
            display_width = st.slider("🔍 圖片顯示大小 (手機若跑版請調小)，只有編輯模式能調", min_value=300, max_value=1000, value=700)

            orig_h, orig_w = st.session_state['upload_result_img'].shape[:2]
            scale_ratio = display_width / float(orig_w)
            new_height = int(orig_h * scale_ratio)
            resized_display_img = cv2.resize(st.session_state['upload_result_img'], (display_width, new_height))
            resized_display_img_rgb = cv2.cvtColor(resized_display_img, cv2.COLOR_BGR2RGB)

            c_mode, c_info = st.columns([1, 2])
            with c_mode:
                delete_mode = st.toggle("🗑️ 啟用編輯模式", value=False, help="開啟後，點擊綠框/紫框可刪除；點擊黑色筆跡可手動補框")
            with c_info:
                if delete_mode:
                    st.warning("⚠️ 點擊綠框/紫框=刪除 | 點擊黑字=手動新增")
                else:
                    st.info("這讓數字被判定成陰影或污漬時還原數字，因此有些非數字類也容易被誤判")

            if delete_mode:
                value = streamlit_image_coordinates(resized_display_img_rgb, key="click_img", width=display_width)
                if 'last_clicked_value' not in st.session_state: st.session_state['last_clicked_value'] = None

                if value is not None and value != st.session_state['last_clicked_value']:
                    st.session_state['last_clicked_value'] = value
                    click_x = value['x']; click_y = value['y']
                    real_x = int(click_x / scale_ratio); real_y = int(click_y / scale_ratio)
                    clicked_existing = False
                    
                    if 'manual_boxes' in st.session_state:
                        for i, mbox in enumerate(st.session_state['manual_boxes']):
                            bx, by, bw, bh = mbox['rect']
                            if bx <= real_x <= bx + bw and by <= real_y <= by + bh:
                                st.session_state['manual_boxes'].pop(i)
                                st.toast("🗑️ 已刪除手動框"); clicked_existing = True; time.sleep(0.1); st.rerun(); break
                    
                    if not clicked_existing:
                        for box_data in all_boxes_data:
                            bx, by, bw, bh = box_data["rect"]
                            if bx <= real_x <= bx + bw and by <= real_y <= by + bh:
                                box_id = box_data["id"]
                                if box_id not in st.session_state['ignored_boxes'] and box_data["conf"] < box_data["thr"]: continue 
                                if box_id in st.session_state['ignored_boxes']:
                                    st.session_state['ignored_boxes'].remove(box_id)
                                    st.toast(f"✅ 已恢復自動框")
                                else:
                                    st.session_state['ignored_boxes'].add(box_id)
                                    st.toast(f"🗑️ 已刪除自動框")
                                clicked_existing = True; st.rerun(); break
                    
                    if not clicked_existing:
                        new_box_data, msg = try_add_manual_box(real_x, real_y, binary_proc, model)
                        if new_box_data:
                            st.session_state['manual_boxes'].append(new_box_data)
                            st.toast(msg); time.sleep(0.5); st.rerun()
                        else: st.toast(msg)
            else:
                st.image(resized_display_img_rgb, use_container_width=True)
            
            if st.session_state['upload_display_list']:
                st.divider(); st.markdown("#### 📊 辨識清單"); cols = st.columns(3)
                for i, h in enumerate(st.session_state['upload_display_list']): cols[i % 3].markdown(h, unsafe_allow_html=True)

    with col_up_right:
        st.markdown("### 📝 確認")
        f_cnt = st.session_state.get('upload_result_count', 0)
        is_disabled = False
        if (uploaded_file is not None or example_choice != "請選擇..."):
            if f_cnt > 0: st.success(f"偵測到 {f_cnt} 個")
            else: st.error("⚠️ 無法偵測"); is_disabled = True 
        else: is_disabled = True

        real_val = st.number_input("正確數量", min_value=0, max_value=f_cnt, value=f_cnt, key="up_input_val", disabled=is_disabled)
        
        if st.button("💾 上傳成績", type="primary", use_container_width=True, disabled=is_disabled):
            try:
                if 'stats' not in st.session_state: st.session_state['stats'] = {}
                if 'upload' not in st.session_state['stats']: st.session_state['stats']['upload'] = {'total': 0, 'correct': 0}
                if 'history' not in st.session_state: st.session_state['history'] = {}
                if 'upload' not in st.session_state['history']: st.session_state['history']['upload'] = []

                st.session_state['stats']['upload']['total'] += f_cnt
                st.session_state['stats']['upload']['correct'] += real_val
                st.session_state['history']['upload'].append({'total': f_cnt, 'correct': real_val})
                st.toast(f"✅ 已儲存！(偵測: {f_cnt} / 正確: {real_val})")
                
                st.session_state['upload_result_img'] = None
                st.session_state['last_uploaded_file_id'] = None
                st.session_state['ignored_boxes'] = set()
                st.session_state['manual_boxes'] = []
                st.session_state['upload_display_list'] = []
                st.session_state['upload_result_count'] = 0
                time.sleep(0.5); st.rerun()
            except Exception as e: st.error(f"❌ 錯誤: {str(e)}")
