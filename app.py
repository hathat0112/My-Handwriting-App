import streamlit as st

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="Handwriting AI", 
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
    
    /* 說明書樣式優化 */
    .manual-box {
        background-color: rgba(128, 128, 128, 0.08);
        border-left: 5px solid #FF4B4B;
        padding: 20px;
        margin-bottom: 25px;
        border-radius: 8px;
    }
    .manual-section {margin-bottom: 15px;}
    .manual-title {font-weight: 800; font-size: 1.15em; color: #FF4B4B; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;}
    .manual-text {font-size: 0.95em; line-height: 1.7; color: inherit; opacity: 0.85;}
    
    @media (prefers-color-scheme: dark) {
        .welcome-title {color: #ddd;}
        .welcome-desc {color: #aaa;}
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 模型與影像核心 (維持現有邏輯)
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
                if hole_area > 5: internal_shapes += 1
    return internal_shapes <= 2

def merge_nearby_boxes(boxes, distance_threshold=20):
    if not boxes: return []
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
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
                if max(0, x1 - ox2) + max(0, ox1 - x2) < distance_threshold and \
                   max(0, y1 - oy2) + max(0, oy1 - y2) < distance_threshold:
                    x1, y1, x2, y2 = min(x1, ox1), min(y1, oy1), max(x2, ox2), max(y2, oy2)
                    used[j] = True
                    merged = True
            new_rects.append([x1, y1, x2, y2])
        if not merged: break
        rects = np.array(new_rects)
    return [(r[0], r[1], r[2]-r[0], r[3]-r[1]) for r in rects]

def preprocess_input(roi):
    h, w = roi.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[(28-nh)//2:(28-nh)//2+nh, (28-nw)//2:(28-nw)//2+nw] = resized
    m = cv2.moments(canvas, True)
    if m['m00'] > 0.1:
        M = np.float32([[1, 0, 14.0 - m['m10']/m['m00']], [0, 1, 14.0 - m['m01']/m['m00']]])
        canvas = cv2.warpAffine(canvas, M, (28, 28), flags=cv2.INTER_CUBIC)
    return canvas.reshape(1, 28, 28, 1).astype('float32')/255.0, canvas.reshape(1, 784).astype('float32')/255.0

def draw_label(img, text, x, y, color=(0, 255, 255), is_dashed=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (lw, lh), _ = cv2.getTextSize(text, font, 1.0, 2)
    if is_dashed: cv2.rectangle(img, (x, y), (x + lw + 10, y + 20), color, 1)
    else:
        cv2.rectangle(img, (x, y - lh - 10), (x + lw, y), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y - 5), font, 1.0, color, 2)

def ensemble_predict(roi, min_conf, strict_mode=False):
    cnn_in, flat_in = preprocess_input(roi)
    pred_cnn = cnn_model.predict(cnn_in, verbose=0)[0]
    lbl_cnn, conf_cnn = np.argmax(pred_cnn), np.max(pred_cnn)
    
    if strict_mode:
        lbl_knn = knn_model.predict(flat_in)[0] if knn_model else -1
        lbl_svm = svm_model.predict(flat_in)[0] if svm_model else -1
        if (lbl_knn != lbl_cnn or lbl_svm != lbl_cnn) and conf_cnn < 0.85: return -1, 0.0, " (Disagree)"
        if conf_cnn < 0.8: return -1, 0.0, " (Low Conf)"

    details = ""
    if not strict_mode:
        lbl_knn = knn_model.predict(flat_in)[0] if knn_model else -1
        lbl_svm = svm_model.predict(flat_in)[0] if svm_model else -1
        dis = []
        if lbl_knn != lbl_cnn: dis.append(f"K:{lbl_knn}")
        if lbl_svm != lbl_cnn: dis.append(f"S:{lbl_svm}")
        if dis: details = f" ({'/'.join(dis)})"
        
    return lbl_cnn, conf_cnn, details

# ==========================================
# 2. 模式組件
# ==========================================

class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.erosion, self.dilation, self.min_conf, self.strict_mode = 0, 0, 0.5, True
        self.last_centers, self.stability_start_time, self.frozen, self.frozen_frame = [], None, False, None
        self.cached_rois, self.last_process_time, self.process_interval = [], 0, 0.25
        self.session_start_time = time.time()

    def update_params(self, ero, dil, conf, strict):
        self.erosion, self.dilation, self.min_conf, self.strict_mode = ero, dil, conf, strict

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        curr = time.time()
        if self.frozen: return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")
        
        h_f, w_f = img.shape[:2]
        rw, rh = int(w_f * 0.7), int(h_f * 0.7)
        rx, ry = (w_f - rw)//2, (h_f - rh)//2
        roi_rect = [rx, ry, rw, rh]
        
        display_img = img.copy()
        cv2.rectangle(display_img, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 3)

        if curr - self.last_process_time < self.process_interval:
            for (dx, dy, dw, dh, txt, clr, dsh) in self.cached_rois:
                cv2.rectangle(display_img, (dx, dy), (dx+dw, dy+dh), clr, 2)
                draw_label(display_img, txt, dx, dy, clr, dsh)
            if self.stability_start_time:
                prog = min((curr - self.stability_start_time)/STABILITY_DURATION, 1.0)
                self._draw_bar(display_img, w_f, h_f, prog)
            return av.VideoFrame.from_ndarray(display_img, format="bgr24")

        self.last_process_time = curr
        roi_hd = img[ry:ry+rh, rx:rx+rw]
        gray = cv2.cvtColor(roi_hd, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
        mask, pred = get_contour_mask(binary, self.erosion), get_prediction_img(binary, self.dilation)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 50]
        merged = merge_nearby_boxes(raw_boxes, 20)
        merged.sort(key=lambda b: b[0])
        
        curr_centers = [(x+w//2, y+h//2) for (x, y, w, h) in merged]
        mov = 9999 if len(curr_centers) != len(self.last_centers) else sum(abs(curr_centers[i][0]-self.last_centers[i][0])+abs(curr_centers[i][1]-self.last_centers[i][1]) for i in range(len(curr_centers)))
        self.last_centers, self.cached_rois = curr_centers, []
        
        detected = False
        for (x, y, w, h) in merged:
            p = self.erosion * 2
            roi_final = pred[max(0,y-p):min(rh,y+h+p), max(0,x-p):min(rw,x+w+p)]
            if roi_final.size > 0 and check_complexity(roi_final):
                lbl, conf, _ = ensemble_predict(roi_final, self.min_conf, self.strict_mode)
                ax, ay = x + rx, y + ry
                if lbl != -1 and conf > self.min_conf:
                    detected = True
                    self.cached_rois.append((ax, ay, w, h, str(lbl), (0, 255, 0), False))
                    cv2.rectangle(display_img, (ax, ay), (ax+w, ay+h), (0,255,0), 2)
                    draw_label(display_img, str(lbl), ax, ay, (0,255,0))
                elif self.strict_mode:
                    self.cached_rois.append((ax, ay, w, h, "?", (0, 255, 255), True))

        if detected and mov < MOVEMENT_THRESHOLD:
            if self.stability_start_time is None: self.stability_start_time = curr
        else: self.stability_start_time = None
        
        if self.stability_start_time:
            prog = min((curr - self.stability_start_time)/STABILITY_DURATION, 1.0)
            self._draw_bar(display_img, w_f, h_f, prog)
            if prog >= 1.0 and self.cached_rois:
                self.frozen, self.frozen_frame = True, display_img.copy()

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

    def _draw_bar(self, img, w, h, p):
        cv2.rectangle(img, (0, h-20), (w, h), (30, 30, 30), -1)
        clr = (0, 255, 255) if p < 1.0 else (0, 255, 0)
        cv2.rectangle(img, (0, h-20), (int(w*p), h), clr, -1)
        cv2.putText(img, "Scanning..." if p < 1.0 else "Captured!", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)

# ==========================================
# 3. 模式分流與指南
# ==========================================

def run_camera_mode(erosion, dilation, min_conf):
    with st.expander("📖 鏡頭辨識指南"):
        st.markdown("""
        <div class="manual-box">
            <div class="manual-section">
                <div class="manual-title">🔍 距離與對焦</div>
                <div class="manual-text">請將紙張拿近鏡頭，確保數字佔據藍色框框 <b>1/4 以上</b> 的高度。若太遠，AI 將因解析度不足而無法識別細節。</div>
            </div>
            <div class="manual-section">
                <div class="manual-title">⏳ 穩定倒數</div>
                <div class="manual-text">看到黃色條出現時，請<b>完全定住手機</b>。倒數 3 秒變綠後會自動完成拍攝並鎖定結果。</div>
            </div>
            <div class="manual-section">
                <div class="manual-title">🔄 重啟掃描</div>
                <div class="manual-text">點擊右側「🔄 重新掃描」可解除凍結，開始下一次辨識。</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    webrtc_streamer(key="cam", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=LiveProcessor, async_processing=True, media_stream_constraints={"video": {"width": 1280, "height": 720, "frameRate": 30}})

def run_canvas_mode(erosion, dilation, min_conf):
    with st.expander("📖 手寫板指南"):
        st.markdown("""
        <div class="manual-box">
            <div class="manual-section">
                <div class="manual-title">✏️ 書寫技巧</div>
                <div class="manual-text">請在中央黑布上書寫，字體不宜過小。系統會即時在右側 Analysis 表格中回報結果。</div>
            </div>
            <div class="manual-section">
                <div class="manual-title">🛡️ 智慧過濾</div>
                <div class="manual-text">手寫板預設開啟<b>高標準過濾</b>，會自動排除像「笑臉」或「無意義塗鴉」的形狀，僅保留高信心的數字。</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    c1, c2 = st.columns([1.8, 1.2], gap="large")
    with c1:
        tool = st.radio("工具", ["✏️ 畫筆", "🧽 橡皮擦"], horizontal=True, label_visibility="collapsed")
        canvas_res = st_canvas(stroke_width=15 if tool=="✏️ 畫筆" else 40, stroke_color="#FFFFFF" if tool=="✏️ 畫筆" else "#000000", background_color="#000000", height=400, width=600, drawing_mode="freedraw", key="canvas")
    with c2:
        if canvas_res.image_data is not None and np.max(canvas_res.image_data) > 0:
            gray = cv2.cvtColor(canvas_res.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask, pred = get_contour_mask(binary, erosion), get_prediction_img(binary, dilation)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            merged = merge_nearby_boxes([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 400], 30)
            merged.sort(key=lambda b: b[0])
            res_list = []
            for i, (x, y, w, h) in enumerate(merged):
                roi = pred[y:y+h, x:x+w]
                if check_complexity(roi):
                    lbl, conf, _ = ensemble_predict(roi, min_conf, strict_mode=True)
                    if lbl != -1: res_list.append({"ID": f"#{i+1}", "數字": str(lbl), "信心度": f"{int(conf*100)}%"})
            st.dataframe(res_list, hide_index=True, use_container_width=True) if res_list else st.info("Waiting for digits...")

def run_upload_mode(erosion, dilation, min_conf):
    with st.expander("📖 上傳辨識指南"):
        st.markdown("""
        <div class="manual-box">
            <div class="manual-section">
                <div class="manual-title">📸 拍照建議</div>
                <div class="manual-text">
                    • <b>距離控制</b>：請將鏡頭靠近紙張拍攝，讓數字充滿畫面，避免過小的字跡影響辨識。<br>
                    • <b>光線環境</b>：請在明亮環境下拍攝，減少強烈陰影對筆畫的干擾。
                </div>
            </div>
            <div class="manual-section">
                <div class="manual-title">🔓 寬容模式</div>
                <div class="manual-text">上傳模式會<b>解鎖嚴格過濾</b>，能偵測光線不足、模糊或寫得較隨意的數字，讓隱藏的數據顯現。</div>
            </div>
            <div class="manual-section">
                <div class="manual-title">🛠️ 微調工具</div>
                <div class="manual-text">若數字相連無法分開，請嘗試調高左側選單的 <b>Erosion (切割沾黏)</b>。</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        h, w = img.shape[:2]
        if w > 1000: img = cv2.resize(img, (1000, int(h * 1000/w)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.normalize(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))), None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask, pred = get_contour_mask(binary, erosion), get_prediction_img(binary, dilation)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 20]
        merged = merge_nearby_boxes(raw_boxes, 25)
        merged.sort(key=lambda b: (b[1]//50, b[0]))
        
        c1, c2 = st.columns([1.5, 1], gap="large")
        res_list, disp = [], img.copy()
        for i, (x, y, w, h) in enumerate(merged):
            roi = pred[y:y+h, x:x+w]
            lbl, conf, det = ensemble_predict(roi, min_conf, strict_mode=False)
            if lbl != -1 and conf > min_conf:
                cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,0), 2)
                draw_label(disp, f"#{i+1}", x, y, (0, 255, 0))
                res_list.append({"ID": f"#{i+1}", "數字": str(lbl), "信心度": f"{int(conf*100)}%{det}"})
        with c1: st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2: st.dataframe(res_list, hide_index=True, use_container_width=True) if res_list else st.warning("No digits found.")

# ==========================================
# 4. 主程式分流
# ==========================================
def main():
    if 'page' not in st.session_state: st.session_state['page'] = 'welcome'
    if st.session_state['page'] == 'welcome':
        st.markdown("<br><br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown('<div class="welcome-container"><div class="welcome-title">✒️ Handwriting AI</div><div class="welcome-desc">智慧手寫數字辨識系統<br>支援即時鏡位、手寫板、圖片上傳</div></div>', unsafe_allow_html=True)
            if st.button("🚀 開始使用 / START", use_container_width=True, type="primary"):
                st.session_state['page'] = 'app'
                st.rerun()
    else:
        st.title("HANDWRITING AI")
        mode = st.sidebar.selectbox("Mode", ["📷 鏡頭 (Live)", "✍️ 手寫板 (Canvas)", "📂 上傳 (Upload)"], index=1)
        with st.sidebar.expander("🔧 Advanced Config"):
            erosion = st.slider("Erosion (切割沾黏)", 0, 5, 0)
            min_conf = st.slider("Confidence (信心門檻)", 0.0, 1.0, 0.5)
        if st.sidebar.button("🏠 回到首頁"):
            st.session_state['page'] = 'welcome'
            st.rerun()
            
        if mode == "📷 鏡頭 (Live)": run_camera_mode(erosion, 0, min_conf)
        elif mode == "✍️ 手寫板 (Canvas)": run_canvas_mode(erosion, 0, min_conf)
        elif mode == "📂 上傳 (Upload)": run_upload_mode(erosion, 0, min_conf)

if __name__ == "__main__":
    main()
