import streamlit as st
import cv2
import numpy as np
import os
import time
import av
import joblib
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_image_coordinates import streamlit_image_coordinates
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==========================================
# 0. 頁面設定 & 極簡 CSS 注入
# ==========================================
st.set_page_config(page_title="Handwriting AI", page_icon="✒️", layout="wide")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# [V87 舒適對焦參數]
STABILITY_DURATION = 1.5  # 1.5秒：不快也不慢，剛好夠對準
MOVEMENT_THRESHOLD = 120  # 容許手部自然晃動
CONFIDENCE_THRESHOLD = 0.60 # 降低門檻，讓數字更容易被「吸住」
ROI_MARGIN_X = 60
ROI_MARGIN_Y = 60
SHRINK_PX = 4

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stButton>button {
        background-color: #2b2b2b;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4a4a4a;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        text-align: center;
        font-weight: 300 !important;
        letter-spacing: 2px;
        margin-bottom: 2rem !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #eaeaea;
    }
    div[data-testid="stVerticalBlock"] > div {
        border-radius: 10px;
    }
    .guide-text {
        font-size: 0.85rem;
        color: #666;
        line-height: 1.5;
        background-color: #f1f3f5;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 共用核心 (保持 V79/V83 最佳邏輯)
# ==========================================
@st.cache_resource
def load_models():
    cnn = None
    model_files = ["cnn_model_robust.h5", "mnist_cnn.h5", "cnn_model.h5"]
    for f in model_files:
        if os.path.exists(f):
            try:
                cnn = load_model(f)
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

cnn_model, knn_model, svm_model = load_models()

def v65_morphology(binary_img, erosion, dilation):
    res = binary_img.copy()
    if erosion > 0:
        kernel = np.ones((3,3), np.uint8)
        res = cv2.erode(res, kernel, iterations=erosion)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel_rect, iterations=2)
    if dilation > 0:
        res = cv2.dilate(res, None, iterations=dilation)
    return res

def center_by_moments(img):
    m = cv2.moments(img, True)
    if m['m00'] < 0.1: return cv2.resize(img, (28, 28))
    cX, cY = m['m10'] / m['m00'], m['m01'] / m['m00']
    tX, tY = 14.0 - cX, 14.0 - cY
    M = np.float32([[1, 0, tX], [0, 1, tY]])
    return cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def preprocess_input(roi):
    h, w = roi.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off, x_off = (28 - nh) // 2, (28 - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    final = center_by_moments(canvas)
    cnn_in = final.reshape(1, 28, 28, 1).astype('float32') / 255.0
    flat_in = final.reshape(1, 784).astype('float32') / 255.0
    return cnn_in, flat_in

def draw_label(img, text, x, y, color=(0, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (lw, lh), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - lh - 10), (x + lw, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - 5), font, scale, color, thickness)

def ensemble_predict(roi, min_conf):
    cnn_in, flat_in = preprocess_input(roi)
    pred_cnn = cnn_model.predict(cnn_in, verbose=0)[0]
    lbl_cnn = np.argmax(pred_cnn)
    conf_cnn = np.max(pred_cnn)
    
    lbl_knn = -1
    if knn_model: lbl_knn = knn_model.predict(flat_in)[0]
    lbl_svm = -1
    if svm_model: lbl_svm = svm_model.predict(flat_in)[0]
    
    votes = [lbl_cnn]
    if knn_model: votes.append(lbl_knn)
    if svm_model: votes.append(lbl_svm)
    
    final_lbl = max(set(votes), key=votes.count)
    vote_count = votes.count(final_lbl)
    
    final_conf = conf_cnn
    details = ""
    
    if vote_count == len(votes):
        final_conf = min(0.99, final_conf + 0.1)
    elif vote_count >= 2:
        if lbl_cnn != final_lbl:
            final_conf -= 0.15
            details = f" (CNN:{lbl_cnn})"
    else:
        final_conf -= 0.3
        details = f" (分歧: C{lbl_cnn}/K{lbl_knn}/S{lbl_svm})"
        
    return final_lbl, final_conf, details

# ==========================================
# 2. 鏡頭模式 (V87 舒適對焦)
# ==========================================
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = cnn_model
        self.erosion = 0
        self.dilation = 2
        self.min_conf = 0.5
        
        self.last_boxes = []
        self.stability_start_time = None
        self.frozen = False
        self.frozen_frame = None
        self.frame_counter = 0
        
        # [V87 設定]
        # 跳幀率 6 (約 5 FPS)：畫面不閃爍，但也跟得上移動
        self.skip_rate = 6  
        self.cached_rois = []
        self.session_start_time = time.time()
        self.warmup_duration = 2.0 # 給使用者 2 秒鐘準備

    def update_params(self, ero, dil, conf):
        self.erosion = ero
        self.dilation = dil
        self.min_conf = conf

    def resume(self):
        self.frozen = False
        self.stability_start_time = None
        self.last_boxes = []
        self.frame_counter = 0
        self.session_start_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if not hasattr(self, 'session_start_time') or self.session_start_time is None:
            self.session_start_time = time.time()
        is_warming_up = (time.time() - self.session_start_time) < self.warmup_duration

        if self.frozen and self.frozen_frame is not None:
            return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")
        
        display_img = img.copy()
        h_f, w_f = img.shape[:2]
        
        roi_rect = [ROI_MARGIN_X, ROI_MARGIN_Y, w_f - 2*ROI_MARGIN_X, h_f - 2*ROI_MARGIN_Y]
        roi_color = (0, 0, 255) if is_warming_up else (255, 0, 0)
        cv2.rectangle(display_img, (roi_rect[0], roi_rect[1]), (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), roi_color, 2)

        self.frame_counter += 1
        
        # 跳幀邏輯
        if not (self.frame_counter % self.skip_rate == 0):
            if len(self.cached_rois) > 0:
                for (dx, dy, dw, dh, txt, box_color) in self.cached_rois:
                    cv2.rectangle(display_img, (dx, dy), (dx+dw, dy+dh), box_color, 2)
                    cv2.putText(display_img, txt, (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if is_warming_up:
                cv2.putText(display_img, "Initializing...", (20, h_f - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(display_img, format="bgr24")
        
        # 影像處理
        roi_img = img[roi_rect[1]:roi_rect[1]+roi_rect[3], roi_rect[0]:roi_rect[0]+roi_rect[2]]
        if roi_img.size == 0: return av.VideoFrame.from_ndarray(display_img, format="bgr24")

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 10)
        binary_proc = v65_morphology(binary, self.erosion, self.dilation)
        
        cnts, _ = cv2.findContours(binary_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = []
        raw_boxes_for_stability = []
        
        for c in cnts:
            if cv2.contourArea(c) < 300: continue
            x, y, w, h = cv2.boundingRect(c)
            if x<5 or y<5: continue
            valid_boxes.append((x,y,w,h))
            raw_boxes_for_stability.append({'box': (x+roi_rect[0], y+roi_rect[1], w, h)})
        
        valid_boxes.sort(key=lambda b: b[0])
        self.cached_rois = []
        
        detected_something = False
        count_id = 1
        
        for (x, y, w, h) in valid_boxes:
            roi = binary_proc[y:y+h, x:x+w]
            
            # 使用 CONFIDENCE_THRESHOLD (0.60) 進行寬鬆判定
            final_lbl, final_conf, _ = ensemble_predict(roi, CONFIDENCE_THRESHOLD)
            
            # 只要超過寬鬆門檻就視為偵測到
            if final_conf > CONFIDENCE_THRESHOLD:
                detected_something = True
                rx, ry = x + roi_rect[0], y + roi_rect[1]
                box_color = (0, 0, 255) if is_warming_up else (0, 255, 0)
                
                cv2.rectangle(display_img, (rx, ry), (rx+w, ry+h), box_color, 2)
                txt = f"#{count_id}"
                draw_label(display_img, txt, rx, ry)
                self.cached_rois.append((rx, ry, w, h, txt, box_color))
                count_id += 1

        # 穩定度與抓拍邏輯
        if len(raw_boxes_for_stability) == 0:
            self.stability_start_time = None
        elif len(self.last_boxes) == 0:
            self.last_boxes = raw_boxes_for_stability
            self.stability_start_time = time.time()
        else:
            total_movement = 0
            for curr_box in raw_boxes_for_stability:
                c_x, c_y, _, _ = curr_box["box"]
                min_dist = 99999
                for last_box in self.last_boxes:
                    l_x, l_y, _, _ = last_box["box"]
                    dist = abs(c_x - l_x) + abs(c_y - l_y)
                    if dist < min_dist: min_dist = dist
                if min_dist < 50: total_movement += min_dist
                else: total_movement += 30 
            
            count_diff = abs(len(raw_boxes_for_stability) - len(self.last_boxes))
            total_movement += count_diff * 50 
            self.last_boxes = raw_boxes_for_stability

            # V87: 寬鬆的移動判定，確保不會一直斷掉
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
                if is_warming_up: 
                    cv2.putText(display_img, "Initializing...", (20, h_f - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

def run_camera_mode(erosion, dilation, min_conf):
    st.caption("請將數字置於鏡頭中央，穩定後自動抓拍")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ctx = webrtc_streamer(
            key="v65-cam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=LiveProcessor,
            async_processing=True,
        )
    
    with col2:
        if ctx.video_processor:
            ctx.video_processor.update_params(erosion, dilation, min_conf)
            if st.button("🔄 重新掃描", use_container_width=True):
                ctx.video_processor.resume()
                
            if ctx.video_processor.frozen:
                st.success("✅ 畫面已鎖定")
            else:
                st.info("⏳ 偵測中...")

# ==========================================
# 3. 手寫板模式
# ==========================================
def run_canvas_mode(erosion, dilation, min_conf):
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
            processed = v65_morphology(binary, erosion, dilation)
            
            merge_kernel = np.ones((4, 4), np.uint8) 
            merged_mask = cv2.dilate(processed, merge_kernel, iterations=2)
            cnts, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_boxes = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 150: continue 
                x, y, w, h = cv2.boundingRect(c)
                if h < 15 or w < 5: continue 
                valid_boxes.append((x,y,w,h))
            
            boxes = sorted(valid_boxes, key=lambda b: b[0])
            draw_img = img_bgr.copy()
            results_list = []
            valid_count = 1
            
            for i, (x, y, w, h) in enumerate(boxes):
                roi = processed[y:y+h, x:x+w]
                final_lbl, final_conf, details = ensemble_predict(roi, min_conf)
                
                if final_conf > min_conf:
                    cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    draw_label(draw_img, f"#{valid_count}", x, y)
                    status_text = f"{int(final_conf*100)}%{details}"
                    results_list.append({"ID": f"#{valid_count}", "數字": str(final_lbl), "信心度": status_text})
                    valid_count += 1
            
            if results_list:
                st.dataframe(results_list, hide_index=True, use_container_width=True)
            else:
                st.info("Waiting for input...")
                
            with st.expander("查看 AI 視覺 (Debug)"):
                st.image(draw_img, caption="Detection", channels="BGR", use_container_width=True)
        else:
            st.markdown("*Ready to analyze...*")

# ==========================================
# 4. 上傳模式 (V83 邏輯 - 變數修復)
# ==========================================
def run_upload_mode(erosion, dilation, min_conf):
    
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
        
        # [變數命名修正] 避免衝突
        img_h, img_w = img_origin.shape[:2]
        
        if img_w > 1000:
            scale = 1000 / img_w
            img_origin = cv2.resize(img_origin, (1000, int(img_h * scale)))
            img_h, img_w = img_origin.shape[:2] 
            
        gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        # BlackHat 核心
        kernel_hat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_hat)
        blackhat_enhanced = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(blackhat_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel_link = np.ones((3,3), np.uint8)
        processed = cv2.dilate(binary, kernel_link, iterations=1)
        if dilation > 0: processed = cv2.dilate(processed, None, iterations=dilation)
        
        cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_boxes_data = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 80: continue # 寬鬆門檻
            
            x, y, w, h = cv2.boundingRect(c)
            if w < 10 and h < 10: continue
            
            # [邏輯修正]
            if w * h > (img_h * img_w * 0.9): continue
            
            roi = processed[y:y+h, x:x+w]
            final_lbl, final_conf, details = ensemble_predict(roi, min_conf)
            if final_conf > min_conf:
                valid_boxes_data.append({'rect': (x,y,w,h), 'lbl': final_lbl, 'conf': final_conf, 'details': details})

        valid_boxes_data.sort(key=lambda item: (item['rect'][1]//50, item['rect'][0]))
        
        c1, c2 = st.columns([1.5, 1], gap="large")
        
        with c1:
            display_img = img_origin.copy()
            valid_count = 1
            results_list = []
            
            for item in valid_boxes_data:
                x, y, w, h = item['rect']
                cv2.rectangle(display_img, (x,y), (x+w,y+h), (0,255,0), 2)
                draw_label(display_img, f"#{valid_count}", x, y)
                results_list.append({"ID": f"#{valid_count}", "數字": str(item['lbl']), "信心度": f"{int(item['conf']*100)}%{item['details']}"})
                valid_count += 1
            
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Recognition Result")

        with c2:
            st.subheader("Result")
            if results_list:
                st.dataframe(results_list, hide_index=True, use_container_width=True)
            else:
                st.warning("No digits found.")
                
            st.divider()
            with st.expander("查看 AI 黑帽運算 (Debug)"):
                st.image(processed, use_container_width=True, caption="BlackHat Vision")

# ==========================================
# 5. 主程式分流
# ==========================================
def main():
    st.title("HANDWRITING AI")
    
    st.sidebar.header("Settings")
    # 預設手寫板 (index=1)
    mode = st.sidebar.selectbox("Mode", ["📷 鏡頭 (Live)", "✍️ 手寫板 (Canvas)", "📂 上傳 (Upload)"], index=1)
    
    st.sidebar.divider()
    
    with st.sidebar.expander("🔧 Advanced Config", expanded=False):
        st.markdown("""
        <div class="guide-text">
        <b>💡 調整指南</b><br>
        • <b>Erosion (瘦身)</b>: 數字黏在一起時調大。<br>
        • <b>Dilation (增肥)</b>: 筆畫太淡或斷掉時調大。<br>
        • <b>Confidence</b>: 雜訊太多時調高。
        </div>
        """, unsafe_allow_html=True)
        
        erosion_iter = st.slider("Erosion (切割沾黏)", 0, 5, 0, help="把線條變細，用來分開黏在一起的字")
        dilation_iter = st.slider("Dilation (筆畫加粗)", 0, 3, 2, help="把線條變粗，用來連接斷掉的筆畫")
        min_conf = st.slider("Confidence (信心門檻)", 0.0, 1.0, 0.50, help="AI 的最低信心標準，太低會顯示雜訊，太高會漏字")

    if cnn_model is None:
        st.error("Model not found!")
        st.stop()

    if mode == "📷 鏡頭 (Live)":
        run_camera_mode(erosion_iter, dilation_iter, min_conf)
    elif mode == "✍️ 手寫板 (Canvas)":
        run_canvas_mode(erosion_iter, dilation_iter, min_conf)
    elif mode == "📂 上傳 (Upload)":
        run_upload_mode(erosion_iter, dilation_iter, min_conf)

if __name__ == "__main__":
    main()
