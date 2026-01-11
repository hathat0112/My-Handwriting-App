import streamlit as st
import pandas as pd  # 新增: 用於圖表
import cv2
import numpy as np
import os
import time
import av
import joblib
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="Handwriting AI (Dev Lab)", 
    page_icon="🛠️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 修飾 (保留原版風格，並增加儀表板樣式)
st.markdown("""
<style>
    header[data-testid="stHeader"] {background-color: transparent; z-index: 999;}
    .stButton>button {
        background-color: #4a4a4a !important; color: white !important; border: none; transition: all 0.3s ease;
    }
    .stButton>button:hover {background-color: #FF4B4B !important; transform: scale(1.02);}
    .welcome-container {text-align: center; padding: 50px; border-radius: 15px; background: rgba(128, 128, 128, 0.1); margin-top: 50px;}
    .welcome-title {font-size: 3rem; font-weight: 700; margin-bottom: 1rem;}
    
    /* 儀表板樣式 */
    .dashboard-card {background-color: #262730; padding: 10px; border-radius: 5px; border: 1px solid #444; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 模型載入
# ==========================================
@st.cache_resource
def load_models():
    # 1. CNN
    cnn = None
    if os.path.exists("mnist_cnn.h5"):
        try: cnn = load_model("mnist_cnn.h5"); print("CNN Loaded")
        except: pass
    
    # 準備訓練資料給 KNN/SVM
    x_flat = None
    y_train = None
    try:
        (x_raw, y_raw), _ = mnist.load_data()
        # 僅用 5000 筆加速啟動
        x_flat = x_raw.reshape(-1, 784)[:5000] / 255.0
        y_train = y_raw[:5000]
    except: pass

    # 2. KNN
    knn = None
    if x_flat is not None:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_flat, y_train)

    # 3. SVM
    svm = None
    if x_flat is not None:
        svm = SVC(kernel='rbf', probability=True)
        svm.fit(x_flat, y_train)
        
    return cnn, knn, svm

try:
    cnn_model, knn_model, svm_model = load_models()
except Exception as e:
    st.error(f"❌ 模型載入失敗: {e}")
    st.stop()

# ==========================================
# 2. 核心影像處理與預測
# ==========================================

# [新增] 靈活的二值化處理
def get_binary_image(gray, method, block_size, c_val):
    if method == "Adaptive Gaussian":
        if block_size % 2 == 0: block_size += 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_val)
    elif method == "Otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    else: # Simple
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return binary

# [修改] 預處理：回傳更多資訊 (Raw Image)
def preprocess_input(roi):
    h, w = roi.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off, x_off = (28 - nh) // 2, (28 - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    
    # 重心置中
    m = cv2.moments(canvas, True)
    if m['m00'] > 0.1:
        cX, cY = m['m10'] / m['m00'], m['m01'] / m['m00']
        tX, tY = 14.0 - cX, 14.0 - cY
        M = np.float32([[1, 0, tX], [0, 1, tY]])
        canvas = cv2.warpAffine(canvas, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
    cnn_in = canvas.reshape(1, 28, 28, 1).astype('float32') / 255.0
    flat_in = canvas.reshape(1, 784).astype('float32') / 255.0
    
    return cnn_in, flat_in, canvas # 多回傳 canvas (原圖)

# [修改] 預測邏輯：回傳詳細字典 (Dict) 而非 Tuple
def ensemble_predict_advanced(roi, min_conf):
    cnn_in, flat_in, raw_img = preprocess_input(roi)
    
    # CNN 預測
    pred_probs = cnn_model.predict(cnn_in, verbose=0)[0]
    cnn_lbl = np.argmax(pred_probs)
    cnn_conf = np.max(pred_probs)
    
    # KNN & SVM
    knn_lbl = knn_model.predict(flat_in)[0] if knn_model else -1
    svm_lbl = svm_model.predict(flat_in)[0] if svm_model else -1
    
    # 簡單投票邏輯 (用於 Live 模式快速判斷)
    final_lbl = cnn_lbl
    if cnn_conf < 0.8 and (knn_lbl == svm_lbl) and (knn_lbl != cnn_lbl):
        final_lbl = knn_lbl
    
    return {
        "final_label": int(final_lbl),
        "conf": float(cnn_conf),
        "probs": pred_probs, # 機率分佈
        "preds": {"CNN": int(cnn_lbl), "KNN": int(knn_lbl), "SVM": int(svm_lbl)},
        "raw_img": raw_img, # 28x28 像素圖
        "models_agree": (cnn_lbl == knn_lbl == svm_lbl)
    }

# 輔助：判斷複雜度 (避免雜訊)
def check_complexity(roi):
    cnts, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(cnts) > 0

# 輔助：合併框
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
                if (max(0, x1-ox2)+max(0, ox1-x2) < distance_threshold) and (max(0, y1-oy2)+max(0, oy1-y2) < distance_threshold):
                    x1, y1, x2, y2 = min(x1, ox1), min(y1, oy1), max(x2, ox2), max(y2, oy2)
                    used[j] = True; merged = True
            new_rects.append([x1, y1, x2, y2])
        if not merged: break
        rects = np.array(new_rects)
    return [(x, y, x2-x, y2-y) for (x, y, x2, y2) in rects]

# ==========================================
# 3. 儀表板顯示組件 (Dashboard UI)
# ==========================================
def display_dashboard(results):
    """將 Canvas 和 Upload 模式的結果以儀表板形式顯示"""
    if not results:
        st.info("尚未偵測到數字")
        return

    st.markdown("### 🔬 開發者分析儀表板 (Developer Dashboard)")
    
    for i, res in enumerate(results):
        lbl = res['final_label']
        conf = res['conf']
        agree = res['models_agree']
        icon = "🟢" if agree and conf > 0.8 else "🔴" if not agree else "⚠️"
        
        with st.expander(f"{icon} 數字 #{i+1}: 預測為 **{lbl}** (信心度 {int(conf*100)}%)", expanded=True):
            c_img, c_table, c_chart = st.columns([1, 2, 3])
            
            # 1. 顯示 AI 看到的 28x28 原圖
            with c_img:
                st.caption("AI 看到的 (Raw Input)")
                # 放大顯示以便觀察像素
                big_img = cv2.resize(res['raw_img'], (150, 150), interpolation=cv2.INTER_NEAREST)
                st.image(big_img, clamp=True, output_format="PNG")
            
            # 2. 模型競技場 (比較不同模型)
            with c_table:
                st.caption("模型投票 (Model Arena)")
                df_vote = pd.DataFrame([res['preds']])
                st.dataframe(df_vote, hide_index=True, use_container_width=True)
                if not agree:
                    st.error("⚠️ 模型意見分歧！")
                else:
                    st.success("✅ 模型意見一致")

            # 3. 機率分佈圖
            with c_chart:
                st.caption("CNN 猶豫程度 (Probability)")
                df_chart = pd.DataFrame({
                    "Digit": range(10),
                    "Prob": res['probs']
                })
                st.bar_chart(df_chart, x="Digit", y="Prob", height=150)

# ==========================================
# 4. 鏡頭模式 (Live)
# ==========================================
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.params = {"method": "Adaptive Gaussian", "block": 15, "c": 10, "conf": 0.5, "erosion": 0}
        self.last_results = []
        
    def update_params(self, new_params):
        self.params = new_params

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            display_img = img.copy()
            
            # 使用側邊欄設定的參數
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = get_binary_image(gray, self.params['method'], self.params['block'], self.params['c'])
            
            # 形態學
            if self.params['erosion'] > 0:
                binary = cv2.erode(binary, np.ones((3,3), np.uint8), iterations=self.params['erosion'])
            
            # 抓輪廓
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_boxes = []
            for c in cnts:
                if cv2.contourArea(c) < 100: continue
                x, y, w, h = cv2.boundingRect(c)
                if h < 20: continue
                raw_boxes.append((x,y,w,h))
                
            merged = merge_nearby_boxes(raw_boxes)
            
            # 預測並繪圖
            for (x, y, w, h) in merged:
                pad = 10
                # 邊界檢查
                y1, y2 = max(0, y-pad), min(binary.shape[0], y+h+pad)
                x1, x2 = max(0, x-pad), min(binary.shape[1], x+w+pad)
                roi = binary[y1:y2, x1:x2]
                
                if roi.size == 0 or not check_complexity(roi): continue
                
                # 呼叫新的預測函式
                res = ensemble_predict_advanced(roi, self.params['conf'])
                
                if res['conf'] > self.params['conf']:
                    color = (0, 255, 0) if res['models_agree'] else (0, 165, 255)
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 2)
                    label = f"{res['final_label']} ({int(res['conf']*100)}%)"
                    cv2.putText(display_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return av.VideoFrame.from_ndarray(display_img, format="bgr24")
        except Exception as e:
            print(f"Error: {e}")
            return frame

def run_camera_mode(params):
    st.info("🎥 鏡頭模式：即時預覽參數調整的效果")
    col1, col2 = st.columns([3, 1])
    with col1:
        ctx = webrtc_streamer(
            key="dev-cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=LiveProcessor,
            async_processing=True,
        )
    with col2:
        if ctx.video_processor:
            ctx.video_processor.update_params(params)
            st.success("✅ 參數已同步至鏡頭")
        st.markdown("**說明：**\n鏡頭模式僅顯示簡化結果，若需詳細圖表分析，請使用手寫板或上傳模式。")

# ==========================================
# 5. 手寫板模式 (Canvas)
# ==========================================
def run_canvas_mode(params):
    c1, c2 = st.columns([1.5, 2], gap="large")
    
    with c1:
        st.subheader("✍️ 繪圖區")
        if st.button("🗑️ 清空畫布"): st.session_state['canvas_key'] = f"canvas_{time.time()}"
        
        canvas_res = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=15, stroke_color="#FFFFFF", background_color="#000000",
            height=300, width=400, drawing_mode="freedraw",
            key=st.session_state.get('canvas_key', 'canvas_0'),
            display_toolbar=True
        )

    with c2:
        if canvas_res.image_data is not None and np.max(canvas_res.image_data) > 0:
            # 處理畫布影像
            raw = canvas_res.image_data.astype(np.uint8)
            img_bgr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR) if raw.shape[2] == 4 else raw
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 使用側邊欄參數二值化
            # 注意：手寫板通常已經是黑底白字，不需要複雜閾值，但為了實驗一致性，我們還是跑一次流程
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if params['erosion'] > 0:
                binary = cv2.erode(binary, np.ones((3,3), np.uint8), iterations=params['erosion'])

            # 抓輪廓
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_boxes = []
            for c in cnts:
                if cv2.contourArea(c) < 50: continue
                x, y, w, h = cv2.boundingRect(c)
                raw_boxes.append((x,y,w,h))
            
            merged = merge_nearby_boxes(raw_boxes)
            
            results = []
            draw_img = img_bgr.copy()
            
            for (x, y, w, h) in merged:
                pad = 10
                roi = binary[max(0, y-pad):min(binary.shape[0], y+h+pad), 
                             max(0, x-pad):min(binary.shape[1], x+w+pad)]
                
                if roi.size == 0: continue
                
                # 取得詳細預測
                res = ensemble_predict_advanced(roi, params['conf'])
                if res['conf'] > 0.1: # 顯示所有可能的結果
                    results.append(res)
                    cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(draw_img, str(res['final_label']), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            st.image(draw_img, caption="偵測位置", channels="BGR", width=300)
            
            # 顯示儀表板
            if results:
                st.divider()
                display_dashboard(results)
        else:
            st.info("請在左側書寫...")

# ==========================================
# 6. 上傳模式 (Upload)
# ==========================================
def run_upload_mode(params):
    st.subheader("📂 圖片實驗室")
    file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
    
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # 顯示原始與預處理圖 (Debug)
        c_orig, c_bin = st.columns(2)
        with c_orig: st.image(img, caption="原始圖片", use_container_width=True, channels="BGR")
        
        # 預處理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = get_binary_image(gray, params['method'], params['block'], params['c'])
        if params['erosion'] > 0:
            binary = cv2.erode(binary, np.ones((3,3), np.uint8), iterations=params['erosion'])
            
        with c_bin: st.image(binary, caption=f"二值化結果 ({params['method']})", use_container_width=True)
        
        # 抓數字
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = []
        for c in cnts:
            if cv2.contourArea(c) < 50: continue
            x, y, w, h = cv2.boundingRect(c)
            # 簡單過濾
            if w*h > img.shape[0]*img.shape[1]*0.9: continue
            raw_boxes.append((x,y,w,h))
            
        merged = merge_nearby_boxes(raw_boxes)
        merged.sort(key=lambda b: (b[1]//50, b[0])) # 排序
        
        results = []
        for (x, y, w, h) in merged:
            pad = params['erosion'] * 2
            roi = binary[max(0, y-pad):min(binary.shape[0], y+h+pad), 
                         max(0, x-pad):min(binary.shape[1], x+w+pad)]
            
            if roi.size == 0 or not check_complexity(roi): continue
            
            res = ensemble_predict_advanced(roi, params['conf'])
            if res['conf'] > params['conf']:
                results.append(res)
        
        # 顯示儀表板
        st.divider()
        if results:
            st.success(f"共偵測到 {len(results)} 個數字")
            display_dashboard(results)
        else:
            st.warning("未偵測到數字，請嘗試調整左側『二值化』或『侵蝕』參數。")

# ==========================================
# 7. 主程式入口
# ==========================================
def main():
    if 'page' not in st.session_state: st.session_state['page'] = 'welcome'

    # --- 歡迎頁面 (保留) ---
    if st.session_state['page'] == 'welcome':
        st.markdown("<br><br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("""
            <div class="welcome-container">
                <div class="welcome-title">🛠️ AI Developer Lab</div>
                <p>這是 App(3) 的增強版，專為開發者設計。<br>
                包含模型投票分析、機率可視化與參數調校功能。</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Enter Lab", use_container_width=True, type="primary"):
                st.session_state['page'] = 'app'
                st.rerun()

    # --- 主程式頁面 ---
    elif st.session_state['page'] == 'app':
        st.title("🛠️ AI Developer Dashboard")
        
        # --- 側邊欄：開發者參數控制台 (新增功能) ---
        with st.sidebar:
            st.header("⚙️ 參數實驗室")
            mode = st.radio("模式選擇", ["📷 鏡頭 (Live)", "✍️ 手寫板 (Canvas)", "📂 上傳 (Upload)"])
            
            st.divider()
            st.subheader("1. 影像處理 (Preprocessing)")
            
            # [新增] 二值化演算法選擇
            thresh_method = st.selectbox("二值化演算法", ["Adaptive Gaussian", "Otsu", "Simple"], 
                                       help="Adaptive: 適合光影不均\nOtsu: 自動尋找最佳閾值\nSimple: 固定閾值")
            
            block_size = 15
            c_val = 10
            
            if thresh_method == "Adaptive Gaussian":
                block_size = st.slider("Block Size (奇數)", 3, 51, 15, step=2)
                c_val = st.slider("C Constant", 0, 50, 10)
            
            erosion = st.slider("Erosion (切割沾黏)", 0, 5, 0)
            
            st.subheader("2. 預測門檻")
            min_conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
            
            if st.button("🏠 Back to Home"):
                st.session_state['page'] = 'welcome'
                st.rerun()

            # 參數打包
            params = {
                "method": thresh_method, 
                "block": block_size, 
                "c": c_val, 
                "erosion": erosion, 
                "conf": min_conf
            }

        # --- 模式分流 ---
        if cnn_model is None:
            st.error("Model Missing! 請確認目錄下有 mnist_cnn.h5")
        else:
            if mode == "📷 鏡頭 (Live)":
                run_camera_mode(params)
            elif mode == "✍️ 手寫板 (Canvas)":
                run_canvas_mode(params)
            elif mode == "📂 上傳 (Upload)":
                run_upload_mode(params)

if __name__ == "__main__":
    main()
