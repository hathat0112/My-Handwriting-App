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

# 設定頁面
st.set_page_config(page_title="AI 手寫辨識 (V65 Super Strict)", page_icon="🔢", layout="wide")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. 共用核心 (Shared Core)
# ==========================================
@st.cache_resource
def load_models():
    cnn = None
    model_files = ["cnn_model_robust.h5", "mnist_cnn.h5", "cnn_model.h5"]
    for f in model_files:
        if os.path.exists(f):
            try:
                cnn = load_model(f)
                print(f"✅ CNN 模型載入成功: {f}")
                break
            except: pass
    
    knn = None
    knn_path = "knn_model.pkl"
    if os.path.exists(knn_path):
        try:
            knn = joblib.load(knn_path)
        except: pass
    
    if knn is None:
        try:
            (x_train, y_train), _ = mnist.load_data()
            x_flat = x_train.reshape(-1, 784) / 255.0
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(x_flat[:5000], y_train[:5000])
            joblib.dump(knn, knn_path)
        except: pass
        
    return cnn, knn

cnn_model, knn_model = load_models()

def v65_morphology(binary_img, erosion, dilation):
    res = binary_img.copy()
    if erosion > 0:
        kernel = np.ones((3,3), np.uint8)
        res = cv2.erode(res, kernel, iterations=erosion)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel_rect, iterations=1)
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
    return final.reshape(1, 28, 28, 1).astype('float32') / 255.0

def count_holes(binary_roi):
    """計算二值化圖像中的孔洞數量 (拓撲特徵)"""
    contours, hierarchy = cv2.findContours(binary_roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        # hierarchy 結構: [Next, Previous, First_Child, Parent]
        # 如果 Parent != -1，代表這是內輪廓 (洞)
        for h in hierarchy[0]:
            if h[3] != -1:
                holes += 1
    return holes

# ==========================================
# 2. 模式 A: 鏡頭模式
# ==========================================
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = cnn_model
        self.knn = knn_model
        self.erosion = 0
        self.dilation = 2
        self.min_conf = 0.6
        self.frozen = False
        self.frozen_frame = None
        
    def update_params(self, ero, dil, conf):
        self.erosion = ero
        self.dilation = dil
        self.min_conf = conf

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.frozen and self.frozen_frame is not None:
             return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 10)
        binary_proc = v65_morphology(binary, self.erosion, self.dilation)
        
        cnts, _ = cv2.findContours(binary_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            if cv2.contourArea(c) < 100: continue
            x, y, w, h = cv2.boundingRect(c)
            if x<5 or y<5: continue
            
            roi = binary_proc[y:y+h, x:x+w]
            inp = preprocess_input(roi)
            if self.model:
                pred = self.model.predict(inp, verbose=0)[0]
                conf = np.max(pred)
                lbl = np.argmax(pred)
                
                if conf > self.min_conf:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, f"{lbl}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def run_camera_mode(erosion, dilation, min_conf):
    st.info("📷 將數字置於鏡頭中央，系統會自動辨識")
    ctx = webrtc_streamer(
        key="v65-cam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveProcessor,
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.update_params(erosion, dilation, min_conf)

# ==========================================
# 3. 模式 B: 手寫板模式
# ==========================================
def run_canvas_mode(erosion, dilation, min_conf):
    c1, c2 = st.columns([2, 1])
    with c1:
        canvas_res = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=20,
            stroke_color="#FFF",
            background_color="#000",
            height=350,
            width=600,
            drawing_mode="freedraw",
            key="canvas_v65"
        )
    
    with c2:
        st.markdown("### 辨識結果")
        if canvas_res.image_data is not None and np.max(canvas_res.image_data) > 0:
            raw = canvas_res.image_data.astype(np.uint8)
            img_bgr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR) if raw.shape[2] == 4 else raw
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = v65_morphology(binary, erosion, dilation)
            
            st.image(processed, caption="[Debug] AI 視角", width=200)
            
            cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = sorted([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 50], key=lambda b: b[0])
            
            results_txt = []
            for i, (x, y, w, h) in enumerate(boxes):
                roi = processed[y:y+h, x:x+w]
                inp = preprocess_input(roi)
                pred = cnn_model.predict(inp, verbose=0)[0]
                conf = np.max(pred)
                lbl = np.argmax(pred)
                
                if conf > min_conf:
                    results_txt.append(f"**#{i+1}**: 數字 `{lbl}` ({int(conf*100)}%)")
            
            if results_txt:
                for r in results_txt: st.markdown(r)
            else:
                st.warning("寫得太潦草或信心過低")

# ==========================================
# 4. 模式 C: 上傳圖片 - 超強過濾版
# ==========================================
def run_upload_mode(erosion, dilation, min_conf):
    st.info("支援 JPG/PNG，已啟用【孔洞偵測】來過濾中文字與陰影")
    
    file = st.file_uploader("選擇圖片", type=["jpg", "png", "jpeg"])
    
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_origin = cv2.imdecode(file_bytes, 1)
        h_orig, w_orig = img_origin.shape[:2]
        
        # 1. 影像增強
        gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        # 2. 雙重二值化 (Dual Thresholding)
        thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_combined = cv2.bitwise_and(thresh_adapt, thresh_otsu)
        
        # V65 形態學
        processed = v65_morphology(binary_combined, erosion, dilation)
        
        cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_count = 0
        display_img = img_origin.copy()
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 80: continue 
            x, y, w, h = cv2.boundingRect(c)
            
            # ==========================================
            # 🛑 第一階段：物理尺寸過濾
            # ==========================================
            if x < 5 or y < 5 or (x+w) > w_orig-5 or (y+h) > h_orig-5: continue # 邊緣
            
            aspect_ratio = w / float(h)
            if aspect_ratio > 1.1: continue # 太寬 (通常是中文字或單詞)
            if aspect_ratio < 0.15: continue # 太細 (通常是雜訊線條)
            
            if w * h > (h_orig * w_orig * 0.1): continue # 巨大物件
            
            roi_check = processed[y:y+h, x:x+w]
            density = cv2.countNonZero(roi_check) / (w * h)
            if density < 0.12 or density > 0.60: continue # 密度過高或過低
            
            # ==========================================
            # 🧠 模型預測
            # ==========================================
            roi = processed[y:y+h, x:x+w]
            inp = preprocess_input(roi)
            pred = cnn_model.predict(inp, verbose=0)[0]
            
            conf = np.max(pred)
            lbl = np.argmax(pred)
            
            # ==========================================
            # 🛑 第二階段：邏輯與特徵過濾 (Hole Check)
            # ==========================================
            holes = count_holes(roi)
            
            # 規則 1: 數字 8 必須有洞
            # 如果 AI 說是 8，但沒有洞，那通常是實心陰影或誤判
            if lbl == 8 and holes == 0: continue
            
            # 規則 2: 數字 0, 6, 9 通常有洞 (偶爾手寫會封起來，所以我們只過濾極端情況)
            # 如果 AI 說是 0 且沒洞，且信心度不高，殺掉
            if lbl == 0 and holes == 0 and conf < 0.95: continue
            
            # 規則 3: 數字 1 不應該有大洞
            if lbl == 1 and holes > 0: continue
            
            # 規則 4: 數字 1 必須瘦長
            # 如果 AI 說是 1，但長寬比太胖 (> 0.5)，那通常是中文字的筆畫
            if lbl == 1 and aspect_ratio > 0.55: continue

            # 規則 5: 針對容易誤判的 4, 3, 7 提高門檻
            # 中文字的筆畫最常被看成 4, 3, 7
            final_conf_thresh = min_conf
            if lbl in [3, 4, 7]: final_conf_thresh += 0.20 # 對這些數字要求超高信心
            if lbl in [1, 8]: final_conf_thresh += 0.10

            if conf > final_conf_thresh:
                # 繪圖
                cv2.rectangle(display_img, (x,y), (x+w,y+h), (0,255,0), 2)
                
                # 文字底色
                label_text = f"{lbl}"
                (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(display_img, (x, y-lh-10), (x+lw, y), (0,255,0), -1)
                cv2.putText(display_img, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                detected_count += 1

        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(img_rgb, use_container_width=True, caption="辨識結果")
        with c2:
            st.image(processed, use_container_width=True, caption="[Debug] AI 視角 (二值化)")
            st.markdown(f"**共找到 {detected_count} 個數字**")

# ==========================================
# 5. 主程式分流
# ==========================================
def main():
    st.sidebar.title("🔢 手寫辨識 V65 Fix")
    mode = st.sidebar.radio("選擇模式", ["📷 鏡頭 (Live)", "✍️ 手寫板 (Canvas)", "📂 上傳圖片 (Upload)"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔪 V65 手術刀參數")
    erosion_iter = st.sidebar.slider("切割沾黏 (Erosion)", 0, 5, 0, help="數字黏在一起時調大這個")
    dilation_iter = st.sidebar.slider("筆畫加粗 (Dilation)", 0, 3, 2, help="筆畫太細時調大這個")
    min_conf = st.sidebar.slider("信心門檻", 0.0, 1.0, 0.5)

    if cnn_model is None:
        st.error("❌ 找不到模型檔案 (cnn_model_robust.h5 或 mnist_cnn.h5)")
        st.stop()

    if mode == "📷 鏡頭 (Live)":
        run_camera_mode(erosion_iter, dilation_iter, min_conf)
    elif mode == "✍️ 手寫板 (Canvas)":
        run_canvas_mode(erosion_iter, dilation_iter, min_conf)
    elif mode == "📂 上傳圖片 (Upload)":
        run_upload_mode(erosion_iter, dilation_iter, min_conf)

if __name__ == "__main__":
    main()
