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

# 設定頁面
st.set_page_config(page_title="AI 手寫辨識 (V79 Shadow Hunter)", page_icon="🔢", layout="wide")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. 共用核心
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
                print(f"✅ CNN 模型載入成功: {f}")
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

# 投票機制
def ensemble_predict(roi, min_conf):
    cnn_in, flat_in = preprocess_input(roi)
    
    # 1. CNN
    pred_cnn = cnn_model.predict(cnn_in, verbose=0)[0]
    lbl_cnn = np.argmax(pred_cnn)
    conf_cnn = np.max(pred_cnn)
    
    # 2. KNN
    lbl_knn = -1
    if knn_model: lbl_knn = knn_model.predict(flat_in)[0]
    
    # 3. SVM
    lbl_svm = -1
    if svm_model: lbl_svm = svm_model.predict(flat_in)[0]
    
    # 投票
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
    with st.expander("📖 鏡頭模式使用說明", expanded=True):
        st.markdown("1. 點擊 `START`。 2. 對準數字。 3. 系統自動框選。")
    st.info("📷 鏡頭模式")
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
    with st.expander("📖 手寫板使用說明 (點擊展開)", expanded=False):
        st.markdown("""
        * **書寫**：在下方直接寫字。
        * **修正**：使用復原或橡皮擦。
        * **辨識**：已啟用三重驗證，若模型意見不合會在清單中顯示。
        """)

    if 'canvas_json' not in st.session_state: st.session_state['canvas_json'] = None
    if 'initial_drawing' not in st.session_state: st.session_state['initial_drawing'] = None

    c1, c2 = st.columns([2, 1.5])
    
    with c1:
        st.markdown("### ✍️ 請在此書寫")
        c_tool, c_acts = st.columns([1.5, 2])
        with c_tool:
            tool_mode = st.radio("🖊️ 工具", ["✏️ 畫筆", "🧽 橡皮擦"], horizontal=True, label_visibility="collapsed")
        
        with c_acts:
            b_undo, b_clear = st.columns(2)
            with b_undo:
                if st.button("↩️ 復原一筆", use_container_width=True):
                    if st.session_state['canvas_json'] is not None:
                        data = st.session_state['canvas_json']
                        if "objects" in data and len(data["objects"]) > 0:
                            data["objects"].pop()
                            st.session_state['initial_drawing'] = data
                            st.session_state['canvas_key'] = f"canvas_{time.time()}"
                            st.rerun()
            with b_clear:
                if st.button("🗑️ 清除全部", use_container_width=True):
                    st.session_state['canvas_key'] = f"canvas_{time.time()}"
                    st.session_state['initial_drawing'] = None
                    st.rerun()

        canvas_res = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=15 if tool_mode == "✏️ 畫筆" else 40,
            stroke_color="#FFFFFF" if tool_mode == "✏️ 畫筆" else "#000000",
            background_color="#000000",
            height=400, width=650, drawing_mode="freedraw",
            initial_drawing=st.session_state['initial_drawing'],
            key=st.session_state.get('canvas_key', 'canvas_0'),
            display_toolbar=False 
        )
        if canvas_res.json_data is not None: st.session_state['canvas_json'] = canvas_res.json_data
    
    with c2:
        st.markdown("### 👁️ 分析與編號")
        result_container = st.container(height=400, border=True)
        
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
                x, y, w, h = cv2.boundingRect(c)
                if area < 150: continue 
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
                    results_list.append({"編號": f"#{valid_count}", "預測數字": str(final_lbl), "狀態": status_text})
                    valid_count += 1
            
            st.image(draw_img, caption="編號對照圖", channels="BGR", use_container_width=True)

            with result_container:
                if results_list: st.dataframe(results_list, hide_index=True, use_container_width=True)
                else: st.info("尚未偵測到有效數字")
        else:
            with result_container: st.info("請在左側書寫...")

# ==========================================
# 4. 上傳模式 (V79 BlackHat Shadow Hunter)
# ==========================================
def run_upload_mode(erosion, dilation, min_conf):
    with st.expander("📖 上傳模式使用指南", expanded=True):
        st.markdown("""
        **1. 上傳**：選擇圖片。 **2. 檢視**：系統會自動過濾雜訊並辨識。
        * **黑帽運算**：使用專業演算法分離陰影與筆跡，Debug 圖將不再有滿天星斗。
        """)

    st.info("✅ 已啟用【V79 黑帽獵影引擎】，無視背景紋路與陰影")
    
    file = st.file_uploader("選擇圖片", type=["jpg", "png", "jpeg"])
    
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_origin = cv2.imdecode(file_bytes, 1)
        
        # 1. 圖片瘦身 (加速)
        h, w = img_origin.shape[:2]
        if w > 1000:
            scale = 1000 / w
            img_origin = cv2.resize(img_origin, (1000, int(h * scale)))
            
        gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        # 2. [核心變革] 黑帽運算 (BlackHat)
        # 用一個比筆畫稍大的 kernel (15x15) 去掃描
        # 它可以把 "比周圍暗的東西" (筆畫) 抓出來，並把 "平滑的背景" (紙張/陰影) 扣除
        # 這是去除光影不均與紋路的最終兵器
        kernel_hat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_hat)
        
        # 3. 對比拉伸 (Normalize)
        # 因為黑帽運算後的筆跡可能很淡，我們把它拉到 0~255 最亮
        blackhat_enhanced = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        
        # 4. Otsu 二值化 (現在背景是純黑的，Otsu 會非常準)
        _, binary = cv2.threshold(blackhat_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. 形態學處理 (稍微連接斷字)
        # 這裡不需要腐蝕，因為黑帽已經幫我們把雜訊殺光了
        kernel_link = np.ones((3,3), np.uint8)
        processed = cv2.dilate(binary, kernel_link, iterations=1)
        
        # 使用者想加粗可以再加
        if dilation > 0:
            processed = cv2.dilate(processed, None, iterations=dilation)
        
        cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_count = 0
        display_img = img_origin.copy()
        valid_boxes_data = []
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 150: continue 
            
            x, y, w, h = cv2.boundingRect(c)
            
            # 尺寸雙重鎖定
            if w < 20 and h < 20: continue
            if w * h > (h * w * 0.9): continue
            
            roi = processed[y:y+h, x:x+w]
            final_lbl, final_conf, details = ensemble_predict(roi, min_conf)
            
            if final_conf > min_conf:
                valid_boxes_data.append({
                    'rect': (x, y, w, h),
                    'lbl': final_lbl,
                    'conf': final_conf,
                    'details': details
                })

        valid_boxes_data.sort(key=lambda item: (item['rect'][1]//50, item['rect'][0]))
        results_list = []
        valid_count = 1

        for idx, item in enumerate(valid_boxes_data):
            x, y, w, h = item['rect']
            lbl = item['lbl']
            conf = item['conf']
            
            cv2.rectangle(display_img, (x,y), (x+w,y+h), (0,255,0), 2)
            draw_label(display_img, f"#{valid_count}", x, y)
            results_list.append(f"**#{valid_count}**: 數字 `{lbl}` ({int(conf*100)}%){item['details']}")
            valid_count += 1
            detected_count += 1

        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(img_rgb, use_container_width=True, caption="辨識結果 (僅編號)")
        with c2:
            st.image(processed, use_container_width=True, caption="[Debug] AI 視角 (黑帽運算)")
            st.markdown(f"**共找到 {detected_count} 個數字**")
            if results_list:
                st.markdown("---")
                st.markdown("#### 📝 詳細清單")
                for r in results_list: st.markdown(r)

# ==========================================
# 5. 主程式分流
# ==========================================
def main():
    st.sidebar.title("🔢 手寫辨識 (V79 Shadow Hunter)")
    mode = st.sidebar.radio("選擇模式", ["📷 鏡頭 (Live)", "✍️ 手寫板 (Canvas)", "📂 上傳圖片 (Upload)"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔪 V65 手術刀參數")
    
    with st.sidebar.expander("❓ 參數調整指南"):
        st.markdown("""
        **1. 切割沾黏 (Erosion)**
        * **功能**：把變粗的線條「削細」。
        
        **2. 筆畫加粗 (Dilation)**
        * **功能**：把變細的線條「變粗」。
        
        **3. 信心門檻**
        * **功能**：AI 多有把握才敢顯示出來。
        * **設定**：預設已降至 **0.5** 以確保不漏字。
        """)

    erosion_iter = st.sidebar.slider("切割沾黏 (Erosion)", 0, 5, 0)
    dilation_iter = st.sidebar.slider("筆畫加粗 (Dilation)", 0, 3, 2)
    min_conf = st.sidebar.slider("信心門檻", 0.0, 1.0, 0.50) 

    if cnn_model is None:
        st.error("❌ 找不到模型檔案")
        st.stop()

    if mode == "📷 鏡頭 (Live)":
        run_camera_mode(erosion_iter, dilation_iter, min_conf)
    elif mode == "✍️ 手寫板 (Canvas)":
        run_canvas_mode(erosion_iter, dilation_iter, min_conf)
    elif mode == "📂 上傳圖片 (Upload)":
        run_upload_mode(erosion_iter, dilation_iter, min_conf)

if __name__ == "__main__":
    main()
