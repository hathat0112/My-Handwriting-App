import streamlit as st

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="Handwriting AI (V124)", 
    page_icon="✒️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

import cv2
import numpy as np
import os
import time
import av
import joblib
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from streamlit_image_coordinates import streamlit_image_coordinates
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 參數設定
STABILITY_DURATION = 3.0    
MOVEMENT_THRESHOLD = 70     
SHRINK_PX = 4

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# CSS 修飾
st.markdown("""
<style>
    header[data-testid="stHeader"] {background-color: transparent; z-index: 999;}
    section[data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, 0.2);}
    .stButton>button {
        background-color: #4a4a4a !important; color: white !important; border: none; transition: all 0.3s ease;
    }
    .stButton>button:hover {background-color: #FF4B4B !important; transform: scale(1.02);}
    iframe[title="streamlit_drawable_canvas.st_canvas"] {border: none !important; box-shadow: none !important; background-color: transparent !important;}
    div[data-testid="stVerticalBlock"] > div {background-color: transparent;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    .welcome-container {text-align: center; padding: 50px; border-radius: 15px; background: rgba(128, 128, 128, 0.1); margin-top: 50px;}
    .welcome-title {font-size: 3rem; font-weight: 700; margin-bottom: 1rem; color: #333;}
    .welcome-desc {font-size: 1.2rem; color: #666; margin-bottom: 2rem;}
    
    @media (prefers-color-scheme: dark) {
        .welcome-title {color: #ddd;}
        .welcome-desc {color: #aaa;}
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 共用核心與模型載入
# ==========================================
@st.cache_resource
def load_models():
    cnn = None
    model_files = ["cnn_model_robust.h5", "mnist_cnn.h5", "cnn_model.h5"]
    for f in model_files:
        if os.path.exists(f):
            try:
                cnn = load_model(f)
                print(f"✅ CNN Loaded: {f}")
                break
            except: pass
    
    x_flat = None
    y_train = None
    try:
        (x_raw, y_raw), _ = mnist.load_data()
        x_flat = x_raw.reshape(-1, 784)[:10000] / 255.0
        y_train = y_raw[:10000]
    except: pass

    knn = None
    knn_path = "knn_model.pkl"
    if os.path.exists(knn_path):
        try: knn = joblib.load(knn_path)
        except: pass
    
    if knn is None and x_flat is not None:
        try:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(x_flat, y_train)
            joblib.dump(knn, knn_path)
        except: pass

    svm = None
    svm_path = "svm_model.pkl"
    if os.path.exists(svm_path):
        try: svm = joblib.load(svm_path)
        except: pass
    
    if svm is None and x_flat is not None:
        try:
            svm = SVC(kernel='rbf', probability=True)
            svm.fit(x_flat, y_train)
            joblib.dump(svm, svm_path)
        except: pass
        
    return cnn, knn, svm

try:
    cnn_model, knn_model, svm_model = load_models()
except Exception as e:
    st.error(f"❌ 模型載入失敗: {e}")
    st.stop()

def get_contour_mask(binary_img, erosion):
    res = binary_img.copy()
    if erosion > 0:
        kernel = np.ones((3,3), np.uint8)
        res = cv2.erode(res, kernel, iterations=erosion)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel_rect, iterations=1)
    return res

def get_prediction_img(binary_img, dilation):
    res = binary_img.copy()
    if dilation > 0:
        kernel_dil = np.ones((3,3), np.uint8)
        res = cv2.dilate(res, kernel_dil, iterations=dilation)
    return res

def check_complexity(roi):
    cnts, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) <= 1: return True
    internal_shapes = 0
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:
                hole_area = cv2.contourArea(cnts[i])
                if hole_area > 5:
                    internal_shapes += 1
    if internal_shapes > 2:
        return False 
    return True

def merge_nearby_boxes(boxes, distance_threshold=20):
    if not boxes: return []
    rects = []
    for (x, y, w, h) in boxes:
        rects.append([x, y, x+w, y+h])
    rects = np.array(rects)
    while True:
        merged = False
        new_rects = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]: continue
            x1, y1, x2, y2 = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]: continue
                ox1, oy1, ox2, oy2 = rects[j]
                dist_x = max(0, x1 - ox2) + max(0, ox1 - x2)
                dist_y = max(0, y1 - oy2) + max(0, oy1 - y2)
                if dist_x < distance_threshold and dist_y < distance_threshold:
                    x1 = min(x1, ox1)
                    y1 = min(y1, oy1)
                    x2 = max(x2, ox2)
                    y2 = max(y2, oy2)
                    used[j] = True
                    merged = True
            new_rects.append([x1, y1, x2, y2])
        if not merged: break
        rects = np.array(new_rects)
    final_boxes = []
    for (x1, y1, x2, y2) in rects:
        final_boxes.append((x1, y1, x2-x1, y2-y1))
    return final_boxes

def preprocess_input(roi):
    h, w = roi.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off, x_off = (28 - nh) // 2, (28 - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    m = cv2.moments(canvas, True)
    if m['m00'] > 0.1:
        cX, cY = m['m10'] / m['m00'], m['m01'] / m['m00']
        tX, tY = 14.0 - cX, 14.0 - cY
        M = np.float32([[1, 0, tX], [0, 1, tY]])
        canvas = cv2.warpAffine(canvas, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cnn_in = canvas.reshape(1, 28, 28, 1).astype('float32') / 255.0
    flat_in = canvas.reshape(1, 784).astype('float32') / 255.0
    return cnn_in, flat_in

def draw_label(img, text, x, y, color=(0, 255, 255), is_dashed=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (lw, lh), _ = cv2.getTextSize(text, font, scale, thickness)
    if is_dashed:
        cv2.rectangle(img, (x, y), (x + lw + 10, y + 20), color, 1)
    else:
        cv2.rectangle(img, (x, y - lh - 10), (x + lw, y), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y - 5), font, scale, color, thickness)

def ensemble_predict(roi, min_conf, strict_mode=False):
    cnn_in, flat_in = preprocess_input(roi)
    pred_cnn = cnn_model.predict(cnn_in, verbose=0)[0]
    lbl_cnn = np.argmax(pred_cnn)
    conf_cnn = np.max(pred_cnn)
    
    lbl_knn = -1
    if knn_model: lbl_knn = knn_model.predict(flat_in)[0]
    lbl_svm = -1
    if svm_model: lbl_svm = svm_model.predict(flat_in)[0]
    
    final_lbl = lbl_cnn
    final_conf = conf_cnn
    details = ""
    
    agree_count = 0
    if knn_model and lbl_knn == lbl_cnn: agree_count += 1
    if svm_model and lbl_svm == lbl_cnn: agree_count += 1
    
    if strict_mode:
        if (knn_model and lbl_knn != lbl_cnn) or (svm_model and lbl_svm != lbl_cnn):
            if final_conf < 0.85:
                return -1, 0.0, " (Disagree)"
        if final_conf < 0.8:
            return -1, 0.0, " (Low Conf)"

    if agree_count == 2:
        final_conf = min(0.99, final_conf + 0.05)
    else:
        if conf_cnn > 0.85:
            final_conf = conf_cnn
        else:
            final_conf = max(0.0, final_conf - 0.15)
            
        disagreements = []
        if knn_model and lbl_knn != lbl_cnn: disagreements.append(f"K:{lbl_knn}")
        if svm_model and lbl_svm != lbl_cnn: disagreements.append(f"S:{lbl_svm}")
        if disagreements: details = f" ({'/'.join(disagreements)})"
        
    return final_lbl, final_conf, details

# ==========================================
# 2. 鏡頭模式
# ==========================================
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = cnn_model
        self.erosion = 0
        self.dilation = 0 
        self.min_conf = 0.50 
        self.strict_mode = True 
        
        self.last_boxes = []
        self.last_centers = [] 
        self.stability_start_time = None
        self.frozen = False
        self.frozen_frame = None
        self.cached_rois = [] 
        self.last_process_time = 0 
        self.process_interval = 0.25 
        self.session_start_time = time.time()
        self.warmup_duration = 2.0 

    def update_params(self, ero, dil, conf, strict):
        self.erosion = ero
        self.dilation = dil
        self.min_conf = conf
        self.strict_mode = strict

    def resume(self):
        self.frozen = False
        self.stability_start_time = None
        self.last_boxes = []
        self.last_centers = []
        self.cached_rois = []
        self.session_start_time = time.time()

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            current_time = time.time()
            if not hasattr(self, 'session_start_time') or self.session_start_time is None:
                self.session_start_time = current_time
            is_warming_up = (current_time - self.session_start_time) < self.warmup_duration

            if self.frozen and self.frozen_frame is not None:
                return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")
            
            display_img = img.copy()
            h_f, w_f = img.shape[:2]
            
            roi_w = int(w_f * 0.7)
            roi_h = int(h_f * 0.7)
            roi_x = (w_f - roi_w) // 2
            roi_y = (h_f - roi_h) // 2
            roi_rect = [roi_x, roi_y, roi_w, roi_h]
            
            roi_color = (0, 0, 255) if is_warming_up else (255, 0, 0)
            cv2.rectangle(display_img, (roi_rect[0], roi_rect[1]), (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), roi_color, 3)

            if (current_time - self.last_process_time) < self.process_interval:
                if len(self.cached_rois) > 0:
                    for (dx, dy, dw, dh, txt, box_color, dashed) in self.cached_rois:
                        cv2.rectangle(display_img, (dx, dy), (dx+dw, dy+dh), box_color, 2)
                        draw_label(display_img, txt, dx, dy, box_color, dashed)
                if self.stability_start_time is not None:
                    elapsed = current_time - self.stability_start_time
                    progress = min(elapsed / STABILITY_DURATION, 1.0)
                    self._draw_progress_bar(display_img, w_f, h_f, progress)
                if is_warming_up:
                    cv2.putText(display_img, "Initializing...", (20, h_f - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(display_img, format="bgr24")

            self.last_process_time = current_time
            roi_img = img[roi_rect[1]:roi_rect[1]+roi_rect[3], roi_rect[0]:roi_rect[0]+roi_rect[2]]
            if roi_img.size == 0: return av.VideoFrame.from_ndarray(display_img, format="bgr24")

            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0) 
            binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
            
            mask_img = get_contour_mask(binary, self.erosion)
            pred_img = get_prediction_img(binary, self.dilation)
            
            cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            raw_boxes = []
            min_area = 50 if self.erosion > 3 else 150
            
            for c in cnts:
                if cv2.contourArea(c) < min_area: continue 
                x, y, w, h = cv2.boundingRect(c)
                if x<5 or y<5: continue
                aspect_ratio = w / float(h)
                if aspect_ratio > 1.5: continue 
                if h < 15: continue
                if x < 10: continue 
                raw_boxes.append((x,y,w,h))
            
            merged_boxes = merge_nearby_boxes(raw_boxes, distance_threshold=20)
            merged_boxes.sort(key=lambda b: b[0])
            self.cached_rois = []
            
            roi_hd = img[roi_rect[1]:roi_rect[1]+roi_rect[3], roi_rect[0]:roi_rect[0]+roi_rect[2]]
            gray_hd = cv2.cvtColor(roi_hd, cv2.COLOR_BGR2GRAY)
            blur_hd = cv2.GaussianBlur(gray_hd, (5, 5), 0)
            binary_hd = cv2.adaptiveThreshold(blur_hd, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
            pred_img_hd = get_prediction_img(binary_hd, self.dilation)

            current_centers = []
            for (x, y, w, h) in merged_boxes:
                current_centers.append((x + w//2, y + h//2))
            
            total_movement = 0
            if len(current_centers) != len(self.last_centers):
                total_movement = 9999 
            else:
                for i in range(len(current_centers)):
                    dx = abs(current_centers[i][0] - self.last_centers[i][0])
                    dy = abs(current_centers[i][1] - self.last_centers[i][1])
                    total_movement += (dx + dy)
            self.last_centers = current_centers

            detected_something = False
            for (x, y, w, h) in merged_boxes:
                pad = self.erosion * 2
                roi_final = pred_img_hd[max(0, y-pad):min(pred_img_hd.shape[0], y+h+pad), 
                                        max(0, x-pad):min(pred_img_hd.shape[1], x+w+pad)]
                
                if roi_final.size == 0: continue
                if not check_complexity(roi_final): continue

                final_lbl, final_conf, _ = ensemble_predict(roi_final, self.min_conf, self.strict_mode)
                
                rx, ry = x + roi_rect[0], y + roi_rect[1]
                
                if final_lbl != -1 and final_conf > self.min_conf:
                    detected_something = True
                    box_color = (0, 255, 0)
                    self.cached_rois.append((rx, ry, w, h, str(final_lbl), box_color, False))
                    cv2.rectangle(display_img, (rx, ry), (rx+w, ry+h), box_color, 2)
                    draw_label(display_img, str(final_lbl), rx, ry, box_color, False)
                elif self.strict_mode:
                    box_color = (0, 255, 255)
                    self.cached_rois.append((rx, ry, w, h, "?", box_color, True))
                    cv2.rectangle(display_img, (rx, ry), (rx+w, ry+h), box_color, 1)

            is_stable = (detected_something and total_movement < MOVEMENT_THRESHOLD)
            if is_stable:
                if self.stability_start_time is None: self.stability_start_time = current_time
            else:
                self.stability_start_time = None
                
            if self.stability_start_time is not None and not is_warming_up:
                elapsed = current_time - self.stability_start_time
                progress = min(elapsed / STABILITY_DURATION, 1.0)
                self._draw_progress_bar(display_img, w_f, h_f, progress)
                if progress >= 1.0 and len(self.cached_rois) > 0:
                    self.frozen = True
                    self.frozen_frame = display_img.copy()

            return av.VideoFrame.from_ndarray(display_img, format="bgr24")
        except Exception as e:
            return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

    def _draw_progress_bar(self, img, w, h, progress):
        bar_h = 20
        bar_y = h - bar_h 
        bar_x = 0
        bar_w = w
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
        fill_w = int(bar_w * progress)
        bar_color = (0, 255, 255)
        if progress >= 1.0: bar_color = (0, 255, 0)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        status_text = "Scanning..." if progress < 1.0 else "Captured!"
        cv2.putText(img, status_text, (10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

def run_camera_mode(erosion, dilation, min_conf, strict_mode):
    st.caption("請將數字置於鏡頭中央，穩定後自動抓拍")
    col1, col2 = st.columns([3, 1])
    with col1:
        ctx = webrtc_streamer(
            key="v124-cam", 
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=LiveProcessor,
            async_processing=True,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1280},
                    "height": {"min": 480, "ideal": 720, "max": 720},
                    "frameRate": {"max": 30},
                }
            }
        )
    with col2:
        if ctx.video_processor:
            ctx.video_processor.update_params(erosion, dilation, min_conf, strict_mode)
            if st.button("🔄 重新掃描", use_container_width=True):
                ctx.video_processor.resume()
            if ctx.video_processor.frozen:
                st.success("✅ 畫面已鎖定")
            else:
                st.info("⏳ 偵測中...")

# ==========================================
# 3. 手寫板模式
# ==========================================
def run_canvas_mode(erosion, dilation, min_conf, strict_mode):
    if 'canvas_json' not in st.session_state: st.session_state['canvas_json'] = None
    if 'initial_drawing' not in st.session_state: st.session_state['initial_drawing'] = None

    c1, c2 = st.columns([1.8, 1.2], gap="large")
    
    with c1:
        st.subheader("Canvas")
        t1, t2, t3 = st.columns([2, 1, 1])
        with t1:
            tool_mode = st.radio("工具", ["✏️ 畫筆", "🧽 橡皮擦"], horizontal=True, label_visibility="collapsed")
        with t2:
            if st.button("↩️ 復原", use_container_width=True):
                if st.session_state['canvas_json']:
                    data = st.session_state['canvas_json']
                    if "objects" in data and len(data["objects"]) > 0:
                        data["objects"].pop()
                        st.session_state['initial_drawing'] = data
                        st.session_state['canvas_key'] = f"canvas_{time.time()}"
                        st.rerun()
        with t3:
            if st.button("🗑️ 清空", use_container_width=True):
                st.session_state['canvas_key'] = f"canvas_{time.time()}"
                st.session_state['initial_drawing'] = None
                st.rerun()

        canvas_res = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=15 if tool_mode == "✏️ 畫筆" else 40,
            stroke_color="#FFFFFF" if tool_mode == "✏️ 畫筆" else "#000000",
            background_color="#000000",
            height=400, width=600, drawing_mode="freedraw",
            initial_drawing=st.session_state['initial_drawing'],
            key=st.session_state.get('canvas_key', 'canvas_0'),
            display_toolbar=False 
        )
        if canvas_res.json_data is not None: st.session_state['canvas_json'] = canvas_res.json_data
    
    with c2:
        st.subheader("Analysis")
        if canvas_res.image_data is not None and np.max(canvas_res.image_data) > 0:
            raw = canvas_res.image_data.astype(np.uint8)
            img_bgr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR) if raw.shape[2] == 4 else raw
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            mask_img = get_contour_mask(binary, erosion)
            pred_img = get_prediction_img(binary, dilation)
            
            cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            raw_boxes = []
            min_area = 50 if erosion > 3 else 400 
            
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area: continue 
                x, y, w, h = cv2.boundingRect(c)
                if h < 20 or w < 10: continue 
                raw_boxes.append((x,y,w,h))
            
            merged_boxes = merge_nearby_boxes(raw_boxes, distance_threshold=30)
            merged_boxes.sort(key=lambda b: b[0])
            
            draw_img = img_bgr.copy()
            results_list = []
            valid_count = 1
            
            for i, (x, y, w, h) in enumerate(merged_boxes):
                pad = erosion * 2
                roi = pred_img[max(0, y-pad):min(pred_img.shape[0], y+h+pad), 
                               max(0, x-pad):min(pred_img.shape[1], x+w+pad)]
                
                if roi.size == 0: continue
                
                if not check_complexity(roi): continue

                final_lbl, final_conf, details = ensemble_predict(roi, min_conf, strict_mode=True)
                
                if final_lbl != -1 and final_conf > min_conf:
                    cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    draw_label(draw_img, f"#{valid_count}", x, y, (0, 255, 0), False)
                    status_text = f"{int(final_conf*100)}%{details}"
                    results_list.append({"ID": f"#{valid_count}", "數字": str(final_lbl), "信心度": status_text})
                    valid_count += 1
            
            if results_list:
                st.dataframe(results_list, hide_index=True, use_container_width=True)
            else:
                st.info("Waiting for input...")
        else:
            st.markdown("*Ready to analyze...*")

# ==========================================
# 4. 上傳模式
# ==========================================
def run_upload_mode(erosion, dilation, min_conf, strict_mode):
    file = st.file_uploader("Drop an image here", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if not file:
        st.markdown("""
        <div style="text-align: center; color: #888; padding: 3rem; border: 2px dashed #ddd; border-radius: 10px;">
            <h3>📤 Upload Image</h3>
            <p>Drag and drop or click to browse</p>
        </div>
        """, unsafe_allow_html=True)
    
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_origin = cv2.imdecode(file_bytes, 1)
        img_h, img_w = img_origin.shape[:2]
        
        if img_w > 1000:
            scale = 1000 / img_w
            img_origin = cv2.resize(img_origin, (1000, int(img_h * scale)))
            img_h, img_w = img_origin.shape[:2] 
            
        gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        kernel_hat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_hat)
        blackhat_enhanced = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(blackhat_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        mask_img = get_contour_mask(binary, erosion)
        pred_img = get_prediction_img(binary, dilation)
        
        cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        min_area = 20 if erosion > 3 else 80
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area: continue 
            x, y, w, h = cv2.boundingRect(c)
            if w < 5 and h < 5: continue
            if w * h > (img_h * img_w * 0.9): continue
            if y + h > img_h - 10: continue 
            raw_boxes.append((x,y,w,h))
            
        merged_boxes = merge_nearby_boxes(raw_boxes, distance_threshold=25)
        merged_boxes.sort(key=lambda item: (item[1]//50, item[0]))
        
        valid_boxes_data = []
        for (x, y, w, h) in merged_boxes:
            pad = erosion * 2
            roi = pred_img[max(0, y-pad):min(pred_img.shape[0], y+h+pad), 
                           max(0, x-pad):min(pred_img.shape[1], x+w+pad)]
            
            if roi.size == 0: continue
            
            if not check_complexity(roi): continue

            final_lbl, final_conf, details = ensemble_predict(roi, min_conf, strict_mode=False)
            
            if final_lbl != -1 and final_conf > min_conf:
                valid_boxes_data.append({'rect': (x,y,w,h), 'lbl': final_lbl, 'conf': final_conf, 'details': details})

        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1:
            display_img = img_origin.copy()
            valid_count = 1
            results_list = []
            
            for item in valid_boxes_data:
                x, y, w, h = item['rect']
                cv2.rectangle(display_img, (x,y), (x+w,y+h), (0,255,0), 2)
                draw_label(display_img, f"#{valid_count}", x, y, (0, 255, 0), False)
                results_list.append({"ID": f"#{valid_count}", "數字": str(item['lbl']), "信心度": f"{int(item['conf']*100)}%{item['details']}"})
                valid_count += 1
            
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Recognition Result")

        with c2:
            st.subheader("Result")
            if results_list:
                st.dataframe(results_list, hide_index=True, use_container_width=True)
            else:
                st.warning("No digits found.")

# ==========================================
# 5. 主程式分流 (含歡迎頁面)
# ==========================================
def main():
    try:
        if 'page' not in st.session_state:
            st.session_state['page'] = 'welcome'

        if st.session_state['page'] == 'welcome':
            st.markdown("<br><br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown("""
                <div class="welcome-container">
                    <div class="welcome-title">✒️ Handwriting AI</div>
                    <div class="welcome-desc">
                        智慧手寫數字辨識系統<br>
                        支援即時鏡頭、手寫板、圖片上傳
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 開始使用 / START", use_container_width=True, type="primary"):
                    st.session_state['page'] = 'app'
                    st.rerun()

        elif st.session_state['page'] == 'app':
            st.title("HANDWRITING AI")
            
            st.sidebar.header("Settings")
            mode = st.sidebar.selectbox("Mode", ["📷 鏡頭 (Live)", "✍️ 手寫板 (Canvas)", "📂 上傳 (Upload)"], index=1)
            st.sidebar.divider()
            
            with st.sidebar.expander("🔧 Advanced Config", expanded=False):
                st.markdown("""
                <div class="guide-text">
                <b>💡 調整指南</b><br>
                • <b>Erosion</b>: 數字黏在一起時調大。<br>
                • <b>Dilation</b>: 筆畫太淡或斷掉時調大。<br>
                </div>
                """, unsafe_allow_html=True)
                
                strict_mode = True 
                erosion_iter = st.slider("Erosion (切割沾黏)", 0, 5, 0)
                dilation_iter = 0 
                min_conf = st.slider("Confidence (信心門檻)", 0.0, 1.0, 0.50)
            
            if st.sidebar.button("🏠 回到首頁"):
                st.session_state['page'] = 'welcome'
                st.rerun()

            if cnn_model is None:
                st.error("Model not found! 請確保 mnist_cnn.h5 存在")
                st.stop()

            if mode == "📷 鏡頭 (Live)":
                run_camera_mode(erosion_iter, dilation_iter, min_conf, strict_mode)
            elif mode == "✍️ 手寫板 (Canvas)":
                run_canvas_mode(erosion_iter, dilation_iter, min_conf, strict_mode)
            elif mode == "📂 上傳 (Upload)":
                run_upload_mode(erosion_iter, dilation_iter, min_conf, strict_mode)
            
    except Exception as e:
        st.error(f"程式執行發生錯誤: {e}")

if __name__ == "__main__":
    main()
