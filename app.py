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
st.set_page_config(page_title="AI 手寫辨識 (Guide Ver.)", page_icon="🔢", layout="wide")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. 共用核心
# ==========================================
@st.cache_resource
def load_models():
    # 1. 載入 CNN (主模型)
    cnn = None
    model_files = ["cnn_model_robust.h5", "mnist_cnn.h5", "cnn_model.h5"]
    for f in model_files:
        if os.path.exists(f):
            try:
                cnn = load_model(f)
                print(f"✅ CNN 模型載入成功: {f}")
                break
            except: pass
    
    # 2. 載入或訓練 KNN (輔助模型)
    knn = None
    knn_path = "knn_model.pkl"
    if os.path.exists(knn_path):
        try:
            knn = joblib.load(knn_path)
            print("✅ KNN 模型載入成功")
        except: pass
    
    if knn is None:
        st.toast("正在訓練輔助用 KNN 模型 (初次執行較慢)...")
        try:
            (x_train, y_train), _ = mnist.load_data()
            x_flat = x_train.reshape(-1, 784) / 255.0
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(x_flat[:10000], y_train[:10000])
            joblib.dump(knn, knn_path)
            print("✅ KNN 模型訓練完成")
        except: pass
        
    return cnn, knn

cnn_model, knn_model = load_models()

def v65_morphology(binary_img, erosion, dilation):
    res = binary_img.copy()
    kernel_noise = np.ones((2,2), np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel_noise)

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
    cnn_in = final.reshape(1, 28, 28, 1).astype('float32') / 255.0
    knn_in = final.reshape(1, 784).astype('float32') / 255.0
    return cnn_in, knn_in

def count_holes(binary_roi):
    contours, hierarchy = cv2.findContours(binary_roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for h in hierarchy[0]:
            if h[3] != -1:
                holes += 1
    return holes

def check_multiline_complexity(binary_roi):
    h, w = binary_roi.shape
    max_strokes = 0
    for r_ratio in [0.25, 0.5, 0.75]:
        row = binary_roi[int(h * r_ratio), :] / 255
        transitions = np.sum(np.abs(np.diff(row)))
        strokes = (transitions + 1) // 2
        max_strokes = max(max_strokes, strokes)
    for c_ratio in [0.25, 0.5, 0.75]:
        col = binary_roi[:, int(w * c_ratio)] / 255
        transitions = np.sum(np.abs(np.diff(col)))
        strokes = (transitions + 1) // 2
        max_strokes = max(max_strokes, strokes)
    return max_strokes

def draw_label(img, text, x, y, color=(0, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (lw, lh), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - lh - 10), (x + lw, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - 5), font, scale, color, thickness)

# ==========================================
# 2. 鏡頭模式
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
        boxes_data = []
        for c in cnts:
            if cv2.contourArea(c) < 100: continue
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
                lbl = np.argmax(pred)
                
                if conf > self.min_conf:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    draw_label(img, f"#{count_id}", x, y)
                    count_id += 1

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def run_camera_mode(erosion, dilation, min_conf):
    # [新增] 詳細使用說明
    with st.expander("📖 鏡頭模式使用指南 & 注意事項 (點擊展開)", expanded=True):
        st.markdown("""
        ### 🎯 使用步驟
        1. **啟動**：點擊下方 `START` 按鈕，瀏覽器會請求攝影機權限，請點選「允許」。
        2. **對準**：將寫有數字的紙張或物體，平穩地置於畫面中央。
        3. **判讀**：系統會即時框選看到的數字，並顯示綠色框框與編號。

        ### ⚠️ 注意事項與技巧
        * **💡 光線是關鍵**：請確保環境**光線充足**。太暗或有強烈陰影（例如手機遮住光線）會導致誤判。
        * **💡 背景要乾淨**：最理想的情況是 **「白紙黑字」**。如果背景太雜亂（例如有格子、花紋），系統可能會混淆。
        * **💡 距離要適中**：數字太小（離鏡頭太遠）會看不清楚；數字太大（爆框）也會無法辨識。
        * **💡 避免手震**：手持鏡頭時請盡量保持穩定，模糊的影像會讓 AI 看成一團霧。
        """)

    st.info("📷 鏡頭模式 (為求流暢，此模式主要使用 CNN)")
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
    # [新增] 詳細使用說明
    with st.expander("📖 手寫板模式使用指南 & 注意事項 (點擊展開)", expanded=True):
        st.markdown("""
        ### 🎯 使用步驟
        1. **書寫**：在下方的黑色畫布區，用滑鼠或觸控筆直接寫下 0-9 的數字。
        2. **工具**：
            * **✏️ 畫筆**：預設工具，用來寫字。
            * **🧽 橡皮擦**：擦掉寫錯的部分。
            * **↩️ 復原一筆**：寫壞了？按一下稍微回溯，不用全部重寫。
            * **🗑️ 清除全部**：一鍵清空畫布，重新開始。
        
        ### ⚠️ 注意事項與技巧
        * **💡 字體端正**：雖然 AI 看得懂潦草字，但寫得端正準確度最高。
        * **💡 不要黏在一起**：請將每個數字分開寫，**不要連筆**或重疊，否則 AI 會把它們看成同一個奇怪的符號。
        * **💡 筆劃完整**：例如數字 `0` 或 `8`，請盡量把圈圈封好，不要留太大的缺口。
        """)

    if 'canvas_json' not in st.session_state: st.session_state['canvas_json'] = None
    if 'initial_drawing' not in st.session_state: st.session_state['initial_drawing'] = None

    c1, c2 = st.columns([3, 1.5])
    
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
            display_toolbar=True
        )
        if canvas_res.json_data is not None: st.session_state['canvas_json'] = canvas_res.json_data
    
    with c2:
        st.markdown("### 📊 辨識清單")
        result_container = st.container(height=400, border=True)
        
        if canvas_res.image_data is not None and np.max(canvas_res.image_data) > 0:
            raw = canvas_res.image_data.astype(np.uint8)
            img_bgr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR) if raw.shape[2] == 4 else raw
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = v65_morphology(binary, erosion, dilation)
            
            with st.expander("👁️ Debug (AI 視角)"):
                st.image(processed, caption="二值化影像", use_container_width=True)
            
            cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = sorted([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 50], key=lambda b: b[0])
            draw_img = img_bgr.copy()
            results_list = []
            
            for i, (x, y, w, h) in enumerate(boxes):
                roi = processed[y:y+h, x:x+w]
                cnn_in, _ = preprocess_input(roi)
                
                pred = cnn_model.predict(cnn_in, verbose=0)[0]
                conf = np.max(pred)
                lbl = np.argmax(pred)
                
                if conf > min_conf:
                    cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    draw_label(draw_img, f"#{i+1}", x, y)
                    results_list.append({"編號": f"#{i+1}", "預測數字": str(lbl), "信心度": f"{int(conf*100)}%"})
            
            with result_container:
                if results_list: st.dataframe(results_list, hide_index=True, use_container_width=True)
                else: st.info("尚未偵測到數字")
        else:
            with result_container: st.info("請在左側書寫...")

# ==========================================
# 4. 上傳模式 (整合 CNN + KNN 雙重驗證)
# ==========================================
def run_upload_mode(erosion, dilation, min_conf):
    # [新增] 詳細使用說明
    with st.expander("📖 上傳模式使用指南 & 疑難排解 (點擊展開)", expanded=True):
        st.markdown("""
        ### 🎯 使用步驟
        1. **上傳**：點擊 `Browse files` 選擇一張含有數字的圖片 (JPG/PNG)。
        2. **等待**：系統會自動進行影像處理、切割、與雙重模型驗證。
        3. **檢視**：圖片上會顯示綠色框與編號，右側清單會列出詳細結果。

        ### ⚠️ 為什麼有些字沒抓到？(系統過濾機制)
        為了避免把「國字」、「插圖」或「陰影」誤判成數字，本模式啟用了 **嚴格過濾**：
        * **🚫 形狀不對**：如果框框太細長（像 "l"）或太寬扁（像 "一"），會被視為雜訊。
        * **🚫 結構太複雜**：系統會掃描筆畫，如果發現線條縱橫交錯（像「法」、「則」等中文字），會直接忽略。
        * **🚫 密度異常**：如果一個框框裡黑色填滿的比例太高（像實心方塊）或太低（像空心圓圈），也會被過濾。
        * **🚫 雙重驗證失敗**：如果 **CNN** 模型說是 8，但 **KNN** 模型說是 6，系統會判定為「有爭議」並扣分，信心不足就會隱藏。

        ### 💡 提升準確率的小撇步
        * 盡量使用 **白底黑字** 的圖片。
        * 如果字黏在一起，試著調整左側的 **「切割沾黏 (Erosion)」** 參數。
        * 如果字筆畫斷掉，試著調整左側的 **「筆畫加粗 (Dilation)」** 參數。
        """)

    st.info("✅ 已啟用【雙重模型驗證】+【結構複雜度過濾】，強力排除非數字干擾")
    
    file = st.file_uploader("選擇圖片", type=["jpg", "png", "jpeg"])
    
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_origin = cv2.imdecode(file_bytes, 1)
        h_orig, w_orig = img_origin.shape[:2]
        gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_combined = cv2.bitwise_and(thresh_adapt, thresh_otsu)
        processed = v65_morphology(binary_combined, erosion, dilation)
        
        cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_count = 0
        display_img = img_origin.copy()
        valid_boxes_data = []
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 100: continue 
            x, y, w, h = cv2.boundingRect(c)
            
            # 物理過濾
            if x < 10 or y < 10 or (x+w) > w_orig-10 or (y+h) > h_orig-10: continue
            if w * h > (h_orig * w_orig * 0.15): continue
            
            roi_check = processed[y:y+h, x:x+w]
            density = cv2.countNonZero(roi_check) / (w * h)
            if density < 0.15 or density > 0.55: continue 
            
            aspect_ratio = w / float(h)
            if aspect_ratio > 1.2: continue 
            if aspect_ratio < 0.15: continue
            
            max_strokes = check_multiline_complexity(roi_check)
            if max_strokes > 3: continue 
            
            # 雙重模型預測
            roi = processed[y:y+h, x:x+w]
            cnn_in, knn_in = preprocess_input(roi)
            
            pred_cnn = cnn_model.predict(cnn_in, verbose=0)[0]
            conf_cnn = np.max(pred_cnn)
            lbl_cnn = np.argmax(pred_cnn)
            
            lbl_knn = -1
            if knn_model:
                lbl_knn = knn_model.predict(knn_in)[0]
            
            final_conf = conf_cnn
            is_disagree = False
            
            if knn_model and lbl_cnn != lbl_knn:
                final_conf -= 0.30 
                is_disagree = True
            
            holes = count_holes(roi)
            if lbl_cnn != 1 and aspect_ratio < 0.35: continue
            if lbl_cnn == 1 and aspect_ratio > 0.6: continue
            if lbl_cnn in [8, 0, 6, 9] and holes == 0: continue
            if lbl_cnn in [1, 2, 3, 5, 7] and holes > 0: continue

            target_thresh = min_conf
            if lbl_cnn in [4, 7]: target_thresh += 0.20
            
            if final_conf > target_thresh:
                status_note = ""
                if is_disagree: status_note = " (⚠️爭議)"
                
                valid_boxes_data.append({
                    'rect': (x, y, w, h),
                    'lbl': lbl_cnn,
                    'conf': final_conf,
                    'note': status_note
                })

        valid_boxes_data.sort(key=lambda item: (item['rect'][1]//50, item['rect'][0]))
        results_list = []

        for idx, item in enumerate(valid_boxes_data):
            x, y, w, h = item['rect']
            lbl = item['lbl']
            conf = item['conf']
            
            cv2.rectangle(display_img, (x,y), (x+w,y+h), (0,255,0), 2)
            draw_label(display_img, f"#{idx+1}", x, y)
            results_list.append(f"**#{idx+1}**: 數字 `{lbl}` ({int(conf*100)}%){item['note']}")
            detected_count += 1

        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(img_rgb, use_container_width=True, caption="辨識結果 (僅編號)")
        with c2:
            st.image(processed, use_container_width=True, caption="[Debug] AI 視角")
            st.markdown(f"**共找到 {detected_count} 個數字**")
            if results_list:
                st.markdown("---")
                st.markdown("#### 📝 詳細清單")
                for r in results_list: st.markdown(r)

# ==========================================
# 5. 主程式分流
# ==========================================
def main():
    st.sidebar.title("🔢 手寫辨識 (Guide Ver.)")
    mode = st.sidebar.radio("選擇模式", ["📷 鏡頭 (Live)", "✍️ 手寫板 (Canvas)", "📂 上傳圖片 (Upload)"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔪 V65 手術刀參數")
    
    # 詳細的參數調整指南
    with st.sidebar.expander("❓ 參數調整指南"):
        st.markdown("""
        **1. 切割沾黏 (Erosion)**
        * **功能**：把變粗的線條「削細」。
        * **何時用**：當兩個數字靠太近，被框在同一個框框時，**調大**此數值。
        
        **2. 筆畫加粗 (Dilation)**
        * **功能**：把變細的線條「變粗」。
        * **何時用**：當一個數字斷成兩截 (例如 5 的上面斷掉)，被認成兩個字時，**調大**此數值。
        
        **3. 信心門檻**
        * **功能**：AI 多有把握才敢顯示出來。
        * **何時用**：畫面雜訊太多、出現很多誤判時，**調高**此值；字都抓不到時，**調低**此值。
        """)

    erosion_iter = st.sidebar.slider("切割沾黏 (Erosion)", 0, 5, 0)
    dilation_iter = st.sidebar.slider("筆畫加粗 (Dilation)", 0, 3, 2)
    min_conf = st.sidebar.slider("信心門檻", 0.0, 1.0, 0.80) 

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
