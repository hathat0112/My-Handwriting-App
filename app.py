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

# 自定義 CSS：極簡黑白風格 + Tooltip 優化
st.markdown("""
<style>
    /* 隱藏預設選單與 Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 全局字體優化 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 按鈕風格：黑底白字圓角 */
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
    
    /* 標題置中與風格化 */
    h1 {
        text-align: center;
        font-weight: 300 !important;
        letter-spacing: 2px;
        margin-bottom: 2rem !important;
    }
    
    /* 側邊欄優化 */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #eaeaea;
    }
    
    /* 卡片式容器 */
    div[data-testid="stVerticalBlock"] > div {
        border-radius: 10px;
    }
    
    /* 說明文字風格 */
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
# 1. 共用核心 (V79 BlackHat 邏輯)
# ==========================================
@st.cache_resource
def load_models():
    # 1. CNN
    cnn = None
    model_files = ["cnn_model_robust.h5", "mnist_cnn.h5", "cnn_model.h5"]
    for f in model_files:
        if os.path.exists(f):
            try:
                cnn = load_model(f)
                break
            except: pass
    
    # 2. 訓練資料
    x_flat = None
    y_train = None
    try:
        (x_raw, y_raw), _ = mnist.load_data()
        x_flat = x_raw.reshape(-1, 784)[:10000] / 255.0
        y_train = y_raw[:10000]
    except: pass

    # 3. KNN
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

    # 4. SVM
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
# 2. 鏡頭模式
# ==========================================
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = cnn_model
        self.erosion = 0
        self.dilation = 2
        self.min_conf = 0.5
    def update_params(self, ero, dil, conf):
        self.erosion = ero
        self.dilation = dil
        self.min_conf = conf
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 10)
        binary_proc = v65_morphology(binary, self.erosion, self.dilation)
        cnts, _ = cv2.findContours(binary_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes_data = []
        for c in cnts:
            if cv2.contourArea(c) < 300: continue
            x, y, w, h = cv2.boundingRect(c)
            if x<5 or y<5: continue
            boxes_data.append((x,y,w,h))
        boxes_data.sort(key=lambda b: b[0])
        count_id = 1
        for (x, y, w, h) in boxes_data:
            roi = binary_proc[y:y+h, x:x+w]
            cnn_in, _ = preprocess_input(roi)
            if self.model:
                pred = self.model.predict(cnn_in, verbose=0)[0]
                conf = np.max(pred)
                if conf > 0.99: conf = 0.99
                if conf > self.min_conf:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    draw_label(img, f"#{count_id}", x, y)
                    count_id += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def run_camera_mode(erosion, dilation, min_conf):
    st.caption("請將數字置於鏡頭中央")
    ctx = webrtc_streamer(
        key="v65-cam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveProcessor,
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.update_params(erosion, dilation, min_conf)

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
# 4. 上傳模式 (V79 BlackHat 核心)
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
        h, w = img_origin.shape[:2]
        if w > 1000:
            scale = 1000 / w
            img_origin = cv2.resize(img_origin, (1000, int(h * scale)))
            
        gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
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
            if cv2.contourArea(c) < 150: continue 
            x, y, w, h = cv2.boundingRect(c)
            if w < 20 and h < 20: continue
            if w * h > (h * w * 0.9): continue
            
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
    # [修正] index=1 讓預設值變成 "手寫板" (List 中的第 2 個選項)
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
