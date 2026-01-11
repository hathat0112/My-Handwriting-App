import streamlit as st
import pandas as pd # 新增 pandas 用於圖表
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

# ==========================================
# 0. 頁面設定 (開發者風格)
# ==========================================
st.set_page_config(
    page_title="AI Model Lab (Dev Tool)", 
    page_icon="🛠️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CSS: 讓介面看起來更像儀表板
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .reportview-container {background: #0e1117;}
    .sidebar .sidebar-content {background: #262730;}
    h1, h2, h3 {font-family: 'Courier New', monospace;}
    .metric-card {background-color: #1f2937; padding: 15px; border-radius: 8px; border: 1px solid #374151;}
    .stDataFrame {border: 1px solid #374151;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 模型載入
# ==========================================
@st.cache_resource
def load_models():
    # 載入 CNN
    cnn = None
    if os.path.exists("mnist_cnn.h5"):
        try: cnn = load_model("mnist_cnn.h5")
        except: pass
    
    # 準備訓練資料給 KNN/SVM
    x_flat = None
    y_train = None
    try:
        (x_raw, y_raw), _ = mnist.load_data()
        x_flat = x_raw.reshape(-1, 784)[:5000] / 255.0 # 僅用 5000 筆加速
        y_train = y_raw[:5000]
    except: pass

    # 載入或訓練 KNN
    knn = None
    knn_path = "knn_model.pkl"
    if os.path.exists(knn_path):
        try: knn = joblib.load(knn_path)
        except: pass
    if knn is None and x_flat is not None:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_flat, y_train)
        # joblib.dump(knn, knn_path) # 開發版不強制存檔

    # 載入或訓練 SVM
    svm = None
    svm_path = "svm_model.pkl"
    if os.path.exists(svm_path):
        try: svm = joblib.load(svm_path)
        except: pass
    if svm is None and x_flat is not None:
        svm = SVC(kernel='rbf', probability=True)
        svm.fit(x_flat, y_train)
        # joblib.dump(svm, svm_path)

    return cnn, knn, svm

try:
    cnn_model, knn_model, svm_model = load_models()
except Exception as e:
    st.error(f"❌ 模型載入失敗: {e}")
    st.stop()

# ==========================================
# 2. 核心分析函式 (Deep Analysis)
# ==========================================
def preprocess_for_analysis(roi):
    """將圖片轉為 28x28 並進行標準化，保留原始特徵供檢視"""
    h, w = roi.shape
    # 保持長寬比縮放
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # 填補至 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off, x_off = (28 - nh) // 2, (28 - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    
    # 重心置中 (Center by Moments) - 這是標準 MNIST 處理
    m = cv2.moments(canvas, True)
    if m['m00'] > 0.1:
        cX, cY = m['m10'] / m['m00'], m['m01'] / m['m00']
        tX, tY = 14.0 - cX, 14.0 - cY
        M = np.float32([[1, 0, tX], [0, 1, tY]])
        canvas = cv2.warpAffine(canvas, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
    cnn_in = canvas.reshape(1, 28, 28, 1).astype('float32') / 255.0
    flat_in = canvas.reshape(1, 784).astype('float32') / 255.0
    return cnn_in, flat_in, canvas

def analyze_digit(roi):
    """執行多模型分析，回傳詳細數據"""
    cnn_in, flat_in, raw_img = preprocess_for_analysis(roi)
    
    # 1. CNN 預測 (含機率分佈)
    pred_prob = cnn_model.predict(cnn_in, verbose=0)[0]
    cnn_lbl = np.argmax(pred_prob)
    cnn_conf = float(np.max(pred_prob))
    
    # 2. KNN 預測
    knn_lbl = knn_model.predict(flat_in)[0] if knn_model else -1
    
    # 3. SVM 預測
    svm_lbl = svm_model.predict(flat_in)[0] if svm_model else -1
    
    return {
        "cnn_label": int(cnn_lbl),
        "cnn_conf": cnn_conf,
        "probs": pred_prob,
        "knn_label": int(knn_lbl),
        "svm_label": int(svm_lbl),
        "raw_img": raw_img # 這是 28x28 的原始圖
    }

# ==========================================
# 3. 輔助函式 (影像處理)
# ==========================================
def get_binary_image(gray, method, block_size, c_val):
    """根據開發者設定的參數進行二值化"""
    if method == "Adaptive Gaussian":
        # Block size 必須是奇數
        if block_size % 2 == 0: block_size += 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_val)
    elif method == "Otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    else: # Simple
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return binary

# ==========================================
# 4. 上傳與實驗模式 (Lab Mode)
# ==========================================
def run_lab_mode(thresh_method, block_size, c_val, show_intermediate):
    st.markdown("### 🧪 影像實驗室 (Image Lab)")
    
    file = st.file_uploader("上傳圖片進行深度分析", type=["jpg", "png", "jpeg"])
    
    if file:
        # 讀取圖片
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_origin = cv2.imdecode(file_bytes, 1)
        
        # 縮放過大圖片
        h, w = img_origin.shape[:2]
        if w > 800:
            scale = 800 / w
            img_origin = cv2.resize(img_origin, (800, int(h * scale)))
        
        gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        # --- 步驟 1: 使用開發者參數進行二值化 ---
        binary = get_binary_image(gray, thresh_method, block_size, c_val)
        
        # 顯示中間產物 (Debug View)
        if show_intermediate:
            c_debug1, c_debug2 = st.columns(2)
            with c_debug1: st.image(gray, caption="原始灰階", use_container_width=True)
            with c_debug2: st.image(binary, caption=f"二值化 ({thresh_method})", use_container_width=True)

        # --- 步驟 2: 輪廓偵測與切割 ---
        # 這裡不使用過多的形態學操作，保留原始雜訊以供測試
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_candidates = []
        display_img = img_origin.copy()
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 100: continue # 過濾極小噪點
            if h < 20: continue
            
            # 在原圖畫框
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_img, f"#{len(digit_candidates)+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            roi = binary[y:y+h, x:x+w]
            digit_candidates.append({"id": len(digit_candidates)+1, "roi": roi, "rect": (x,y,w,h)})

        # 排序 (從左到右，從上到下)
        digit_candidates.sort(key=lambda k: (k['rect'][1] // 50, k['rect'][0]))

        # --- 步驟 3: 顯示全域結果 ---
        st.image(display_img, caption=f"偵測到 {len(digit_candidates)} 個潛在區域", use_container_width=True, channels="BGR")
        
        if not digit_candidates:
            st.warning("⚠️ 未偵測到任何數字，請調整左側『二值化參數』。")
            return

        st.divider()
        st.markdown("### 🔬 深度分析報告 (Deep Analysis Report)")
        
        # --- 步驟 4: 逐一分析並顯示詳細儀表板 ---
        # 為了避免畫面過長，如果超過 10 個只顯示前 10 個
        limit = 20
        for i, item in enumerate(digit_candidates[:limit]):
            res = analyze_digit(item['roi'])
            
            # 判斷模型是否衝突
            models_agree = (res['cnn_label'] == res['knn_label'] == res['svm_label'])
            status_icon = "🟢" if models_agree else "🔴"
            if res['cnn_conf'] < 0.7: status_icon = "⚠️"

            with st.container():
                st.markdown(f"#### {status_icon} Digit #{item['id']} (Prediction: **{res['cnn_label']}**)")
                
                c_visual, c_stats, c_chart = st.columns([1, 2, 3])
                
                # [Col 1] 視覺化：模型看到的 28x28 Raw Input
                with c_visual:
                    # 放大顯示像素圖
                    enlarged = cv2.resize(res['raw_img'], (150, 150), interpolation=cv2.INTER_NEAREST)
                    st.image(enlarged, caption="28x28 Input (Raw)", clamp=True)
                    st.caption(f"Conf: {res['cnn_conf']:.2f}")

                # [Col 2] 數據：模型競技場
                with c_stats:
                    st.markdown("**Model Arena:**")
                    match_data = {
                        "Model": ["CNN", "KNN", "SVM"],
                        "Pred": [res['cnn_label'], res['knn_label'], res['svm_label']]
                    }
                    st.dataframe(pd.DataFrame(match_data), hide_index=True, use_container_width=True)
                    if not models_agree:
                        st.error("模型意見分歧！")

                # [Col 3] 圖表：機率分佈
                with c_chart:
                    st.markdown("**Probability Distribution (CNN):**")
                    chart_df = pd.DataFrame({
                        "Digit": list(range(10)),
                        "Probability": res['probs']
                    })
                    st.bar_chart(chart_df, x="Digit", y="Probability", height=150)
                
                st.markdown("---")
        
        if len(digit_candidates) > limit:
            st.info(f"還有 {len(digit_candidates) - limit} 個數字未顯示...")

# ==========================================
# 5. 主程式入口
# ==========================================
def main():
    st.title("🛠️ AI Developer Dashboard")
    st.markdown("此工具專為 **開發者與資料科學家** 設計，用於分析模型行為、調整前處理參數與除錯。")

    # --- 側邊欄：開發者參數控制台 ---
    with st.sidebar:
        st.header("⚙️ Config Lab")
        
        st.subheader("1. 影像前處理 (Preprocessing)")
        thresh_method = st.selectbox("二值化演算法", ["Adaptive Gaussian", "Otsu", "Simple"], index=0)
        
        block_size = 11
        c_val = 10
        if thresh_method == "Adaptive Gaussian":
            block_size = st.slider("Block Size (奇數)", 3, 51, 15, step=2, help="決定局部閾值的區域大小")
            c_val = st.slider("C Constant", 0, 50, 10, help="從平均值減去的常數")
        
        show_intermediate = st.checkbox("顯示中間運算圖 (Binary Output)", value=True)
        
        st.divider()
        st.subheader("2. 模型資訊")
        st.info(f"CNN: {'✅ Loaded' if cnn_model else '❌ Missing'}")
        st.info(f"KNN: {'✅ Loaded' if knn_model else '⚠️ Training...'}")
        st.info(f"SVM: {'✅ Loaded' if svm_model else '⚠️ Training...'}")

    # 目前僅開放 Lab Mode (因為這是 Developer Tool)
    run_lab_mode(thresh_method, block_size, c_val, show_intermediate)

if __name__ == "__main__":
    main()
