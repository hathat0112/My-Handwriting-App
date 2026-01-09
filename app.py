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
st.set_page_config(page_title="AI 手寫辨識 (V65 Ultimate)", page_icon="🔢", layout="wide")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. 共用核心 (Shared Core) - 所有模式通用
# ==========================================
@st.cache_resource
def load_models():
    """載入 CNN 主模型與 KNN 輔助模型"""
    cnn = None
    # 嘗試載入多種可能的模型檔名
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
    
    # 若無 KNN 則現場訓練一個簡單的
    if knn is None:
        try:
            (x_train, y_train), _ = mnist.load_data()
            x_flat = x_train.reshape(-1, 784) / 255.0
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(x_flat[:5000], y_train[:5000]) # 僅用 5000 筆加速
            joblib.dump(knn, knn_path)
        except: pass
        
    return cnn, knn

# 初始化模型
cnn_model, knn_model = load_models()

def v65_morphology(binary_img, erosion, dilation):
    """
    [V65 核心] 形態學處理：先切割(Erosion)再膨脹(Dilation)
    來自 app (1).py 的手術刀功能
    """
    res = binary_img.copy()
    
    # 1. 手術刀切割 (Erosion)：把黏在一起的切開
    if erosion > 0:
        kernel = np.ones((3,3), np.uint8)
        res = cv2.erode(res, kernel, iterations=erosion)

    # 2. 斷筆修補 (Close)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel_rect, iterations=1)

    # 3. 筆畫加粗 (Dilation)
    if dilation > 0:
        res = cv2.dilate(res, None, iterations=dilation)
        
    return res

def center_by_moments(img):
    """影像重心置中 (提升 MNIST 準確度關鍵)"""
    m = cv2.moments(img, True)
    if m['m00'] < 0.1: return cv2.resize(img, (28, 28))
    cX, cY = m['m10'] / m['m00'], m['m01'] / m['m00']
    tX, tY = 14.0 - cX, 14.0 - cY
    M = np.float32([[1, 0, tX], [0, 1, tY]])
    return cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def preprocess_input(roi):
    """將裁切下來的 ROI 轉為模型可讀格式 (1, 28, 28, 1)"""
    h, w = roi.shape
    # 保持比例縮放
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # 貼到 28x28 畫布
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off, x_off = (28 - nh) // 2, (28 - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    
    # 重心置中與正規化
    final = center_by_moments(canvas)
    return final.reshape(1, 28, 28, 1).astype('float32') / 255.0

# ==========================================
# 2. 模式 A: 鏡頭模式專用邏輯 (Live Camera)
# 結合 app.py 的穩定偵測 + app (1).py 的形態學
# ==========================================
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = cnn_model
        self.knn = knn_model
        self.erosion = 0    # 預設值，會由 update_params 更新
        self.dilation = 2
        self.min_conf = 0.6
        
        # 穩定度與抓拍變數 (來自 app.py)
        self.last_boxes = []
        self.stability_start = None
        self.frozen = False
        self.frozen_frame = None
        self.ui_results = []
        
    def update_params(self, ero, dil, conf):
        self.erosion = ero
        self.dilation = dil
        self.min_conf = conf

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.frozen and self.frozen_frame is not None:
             return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")

        # 1. 前處理 (Adaptive Threshold 適合鏡頭)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 10)
        
        # [V65 Feature] 形態學處理
        binary_proc = v65_morphology(binary, self.erosion, self.dilation)
        
        # 2. 輪廓偵測
        cnts, _ = cv2.findContours(binary_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_boxes = []
        
        for c in cnts:
            if cv2.contourArea(c) < 100: continue
            x, y, w, h = cv2.boundingRect(c)
            if x<5 or y<5: continue # 邊緣過濾
            
            # 預測
            roi = binary_proc[y:y+h, x:x+w]
            inp = preprocess_input(roi)
            if self.model:
                pred = self.model.predict(inp, verbose=0)[0]
                conf = np.max(pred)
                lbl = np.argmax(pred)
                
                if conf > self.min_conf:
                    current_boxes.append({'rect':(x,y,w,h), 'lbl':lbl, 'conf':conf})
                    # 繪圖
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, f"{lbl}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 3. 簡單的穩定度邏輯 (簡化版)
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
# 3. 模式 B: 手寫板專用邏輯 (Canvas)
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
        st.markdown("### 👁️ 辨識結果")
        if canvas_res.image_data is not None and np.max(canvas_res.image_data) > 0:
            # 轉換影像
            raw = canvas_res.image_data.astype(np.uint8)
            img_bgr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR) if raw.shape[2] == 4 else raw
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 手寫板適合 Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # [V65 Feature] 形態學處理
            processed = v65_morphology(binary, erosion, dilation)
            
            # 顯示處理後影像 (Debug)
            st.image(processed, caption="AI 看見的影像 (經切割處理)", width=200)
            
            # 偵測與辨識
            cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 排序
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
# 4. 模式 C: 上傳圖片專用邏輯 (Upload)
# 結合 app.py 的編輯模式 (Edit Mode) 與 防呆機制
# ==========================================
def run_upload_mode(erosion, dilation, min_conf):
    st.info("支援 JPG/PNG，可切換至「編輯模式」修正誤判")
    
    file = st.file_uploader("選擇圖片", type=["jpg", "png", "jpeg"])
    edit_mode = st.toggle("🔧 啟用編輯模式 (點擊刪除/新增)", value=False)
    
    if 'ignored_boxes' not in st.session_state: st.session_state.ignored_boxes = set()
    if 'manual_boxes' not in st.session_state: st.session_state.manual_boxes = []
    
    # 換圖片時重置
    if file and st.session_state.get('last_file') != file.name:
        st.session_state.ignored_boxes = set()
        st.session_state.manual_boxes = []
        st.session_state.last_file = file.name

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_origin = cv2.imdecode(file_bytes, 1)
        
        # 前處理
        gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        # 自動判斷模式：照片用 Adaptive, 截圖用 Otsu
        is_photo = np.mean(gray) < 240 and np.std(gray) > 30
        if is_photo:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 10)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
        # [V65 Feature]
        processed = v65_morphology(binary, erosion, dilation)
        
        # 偵測
        cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_data = []
        
        display_img = img_origin.copy()
        
        # 自動框
        for c in cnts:
            if cv2.contourArea(c) < 50: continue
            x, y, w, h = cv2.boundingRect(c)
            
            # ==========================================
            # 🛑 強化版防呆過濾 (Stricter Filtering)
            # ==========================================
            aspect_ratio = w / float(h)
            
            # 1. 嚴格長寬比：數字通常是瘦的，正方形(1.0)或橫向(>1.0)通常是中文字或背景
            if aspect_ratio > 0.9: 
                continue 
            
            # 2. 邊框大小過濾：過大的框通常是背景 (超過5%)
            img_area = img_origin.shape[0] * img_origin.shape[1]
            if w * h > (img_area * 0.05): 
                continue 
            
            # 3. 密度過濾：數字筆畫細，若密度過高(>0.65)通常是色塊
            roi_check = binary[y:y+h, x:x+w]
            density = cv2.countNonZero(roi_check) / (w * h)
            if density > 0.65: 
                continue 
            # ==========================================

            bid = f"{x}_{y}_{w}_{h}"
            
            if bid in st.session_state.ignored_boxes:
                cv2.rectangle(display_img, (x,y), (x+w,y+h), (128,128,128), 1)
                continue
            
            roi = processed[y:y+h, x:x+w]
            inp = preprocess_input(roi)
            pred = cnn_model.predict(inp, verbose=0)[0]
            
            if np.max(pred) > min_conf:
                lbl = np.argmax(pred)
                cv2.rectangle(display_img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(display_img, str(lbl), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                detected_data.append({'id': bid, 'rect':(x,y,w,h), 'type':'auto'})

        # 手動框
        for mbox in st.session_state.manual_boxes:
            mx, my, mw, mh = mbox['rect']
            cv2.rectangle(display_img, (mx,my), (mx+mw,my+mh), (255,0,255), 2)
            cv2.putText(display_img, str(mbox['lbl']), (mx, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
            detected_data.append({'id': 'manual', 'rect':(mx,my,mw,mh), 'type':'manual'})

        # 顯示
        # 轉換為 RGB
        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        if edit_mode:
            st.warning("點擊綠框可刪除；點擊未偵測到的黑字可新增")
            value = streamlit_image_coordinates(img_rgb, key="click_upload")
            
            if value:
                cx, cy = value['x'], value['y']
                hit = False
                # 刪除邏輯
                for item in detected_data:
                    if item['type'] == 'manual': continue # 簡化：手動框先不刪
                    rx, ry, rw, rh = item['rect']
                    if rx < cx < rx+rw and ry < cy < ry+rh:
                        st.session_state.ignored_boxes.add(item['id'])
                        hit = True; st.rerun(); break
                
                # 新增邏輯
                if not hit:
                    # 在 processed 找點擊的輪廓
                    mcnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for mc in mcnts:
                        if cv2.pointPolygonTest(mc, (cx, cy), False) >= 0:
                            mx, my, mw, mh = cv2.boundingRect(mc)
                            m_roi = processed[my:my+mh, mx:mx+mw]
                            m_pred = cnn_model.predict(preprocess_input(m_roi), verbose=0)[0]
                            st.session_state.manual_boxes.append({'rect':(mx,my,mw,mh), 'lbl':np.argmax(m_pred)})
                            st.rerun(); break
        else:
            st.image(img_rgb, use_container_width=True)
            st.markdown(f"**共找到 {len(detected_data)} 個數字**")

# ==========================================
# 5. 主程式分流 (Main Dispatcher)
# ==========================================
def main():
    st.sidebar.title("🔢 手寫辨識 V65 Ultimate")
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
