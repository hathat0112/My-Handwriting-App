import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ==========================================
#              設定與模型載入
# ==========================================
st.set_page_config(page_title="AI 手寫數字辨識 (V31 Tuned)", page_icon="🔢", layout="wide")

MODEL_FILE = "cnn_model_robust.h5"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return tf.keras.models.load_model(MODEL_FILE)
    return None

if not os.path.exists(MODEL_FILE):
    st.error(f"找不到模型檔案: {MODEL_FILE}")
    st.stop()

cnn_model = load_model()

# ==========================================
#              核心演算法
# ==========================================
def center_by_moments_cnn(src):
    """將影像重心對齊"""
    img = src.copy()
    m = cv2.moments(img, True)
    if m['m00'] < 0.1: return cv2.resize(img, (28, 28))
    cX, cY = m['m10'] / m['m00'], m['m01'] / m['m00']
    tX, tY = 14.0 - cX, 14.0 - cY
    M = np.float32([[1, 0, tX], [0, 1, tY]])
    return cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def split_touching_digits(roi_binary):
    """切割連字"""
    h, w = roi_binary.shape
    if w / h < 1.2: return [(0, roi_binary)]
    projection = np.sum(roi_binary, axis=0)
    mid_start, mid_end = int(w * 0.25), int(w * 0.75)
    if mid_end <= mid_start: return [(0, roi_binary)]
    split_x = mid_start + np.argmin(projection[mid_start:mid_end])
    if projection[split_x] > (h * 255 * 0.5): return [(0, roi_binary)]
    part1 = roi_binary[:, :split_x]
    part2 = roi_binary[:, split_x:]
    if part1.shape[1] < 5 or part2.shape[1] < 5: return [(0, roi_binary)]
    return [(0, part1), (split_x, part2)]

def analyze_hole_geometry(binary_roi):
    """分析洞的數量與位置"""
    roi_copy = binary_roi.copy()
    contours, hierarchy = cv2.findContours(roi_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None: return 0, None
    valid_holes = []
    h_img, w_img = roi_copy.shape
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1: 
            area = cv2.contourArea(contours[i])
            if area > 15: 
                M = cv2.moments(contours[i])
                if M['m00'] != 0:
                    cy = int(M['m01'] / M['m00'])
                    norm_y = cy / float(h_img)
                    valid_holes.append((area, norm_y))
    if not valid_holes: return 0, None
    valid_holes.sort(key=lambda x: x[0], reverse=True)
    largest_hole_y = valid_holes[0][1]
    return len(valid_holes), largest_hole_y

def process_and_predict(image_bgr, min_area, min_density, show_debug=False):
    result_img = image_bgr.copy()
    
    # 1. 轉灰階 & 總亮度檢查
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    max_val = np.max(gray)
    if max_val < 50:
        if show_debug: st.warning("⚠️ 畫面太暗，忽略處理")
        return result_img, []

    # 2. Otsu 二值化
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_proc = cv2.dilate(thresh, None, iterations=1)
    
    if show_debug:
        st.image(binary_proc, caption="AI 看到的二值化影像", width=300)
    
    # 3. 連通域分析
    nb, output, stats_cc, _ = cv2.connectedComponentsWithStats(binary_proc, connectivity=8)
    raw_boxes = sorted([stats_cc[i, :4] for i in range(1, nb) if stats_cc[i, cv2.CC_STAT_AREA] > min_area], key=lambda b: b[0])

    rois_to_pred = []
    coords_to_draw = []
    h_img, w_img = binary_proc.shape 

    for box in raw_boxes:
        x, y, w, h = box
        if x < 5 or y < 5 or (x + w) > w_img - 5 or (y + h) > h_img - 5: continue
        if h < 20: continue 

        split_results = split_touching_digits(binary_proc[y:y+h, x:x+w])
        
        for offset_x, sub_roi in split_results:
            sh, sw = sub_roi.shape
            if sw == 0 or sh == 0: continue
            
            n_white_pix = cv2.countNonZero(sub_roi)
            box_area = sw * sh
            density = n_white_pix / float(box_area)
            if density < min_density: continue
            
            side = max(sw, sh)
            container = np.zeros((side+40, side+40), dtype=np.uint8)
            offset_y, offset_x_c = 20 + (side-sh)//2, 20 + (side-sw)//2
            container[offset_y:offset_y+sh, offset_x_c:offset_x_c+sw] = sub_roi
            
            final_roi = center_by_moments_cnn(cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA))
            final_roi_norm = np.expand_dims(final_roi.astype('float32') / 255.0, axis=-1)
            
            rois_to_pred.append(final_roi_norm)
            coords_to_draw.append((x + offset_x, y, sw, sh, sub_roi))

    detected_numbers = []
    if len(rois_to_pred) > 0:
        predictions = cnn_model.predict(np.array(rois_to_pred), verbose=0)
        
        for i, pred_probs in enumerate(predictions):
            res_id = np.argmax(pred_probs)
            confidence = np.max(pred_probs)
            rx, ry, w, h, roi_original = coords_to_draw[i]
            
            display_text = str(res_id)
            color = (0, 255, 0)
            
            # Hybrid 邏輯修正
            num_holes, hole_y = analyze_hole_geometry(roi_original)
            aspect_ratio = w / float(h)
            pixel_count = cv2.countNonZero(roi_original)
            density = pixel_count / float(w * h)

            if res_id == 6:
                if hole_y is not None and hole_y < 0.58: res_id, display_text, color = 0, "0*", (0, 255, 255)
            elif res_id == 8:
                if num_holes == 1: res_id, display_text, color = 0, "0*", (0, 255, 255)
            
            elif res_id == 2:
                # [V31 修改] 
                # 原本這裡有一行 code 會把太瘦的 2 強制變成 1
                # 現在已經移除，讓它保持是 2
                
                # 保留對 7 的檢查 (如果 2 的底部太短，可能是 7)
                h_r, w_r = roi_original.shape
                pts = cv2.findNonZero(roi_original[int(h_r*0.7):, :])
                if pts is not None and cv2.boundingRect(pts)[2] < w_r * 0.5:
                    res_id, display_text, color = 7, "7*", (0, 255, 255)
            
            elif res_id == 7:
                if aspect_ratio < 0.5 or density < 0.25: res_id, display_text, color = 1, "1*", (0, 255, 255)
            elif res_id == 4 or res_id == 9:
                has_hole = (num_holes > 0)
                if res_id == 9 and not has_hole: res_id, display_text, color = 4, "4*", (0, 255, 255)
                elif res_id == 4 and has_hole and confidence < 0.95: res_id, display_text, color = 9, "9*", (0, 255, 255)
            
            detected_numbers.append(str(res_id))
            cv2.rectangle(result_img, (rx, ry), (rx+w, ry+h), color, 2)
            cv2.putText(result_img, display_text, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
    return result_img, detected_numbers

# ==========================================
#              Streamlit UI 介面
# ==========================================
st.title("🔢 AI 多數字辨識系統 (V31 Tuned)")

st.sidebar.header("🔧 設定")
mode_option = st.sidebar.selectbox("輸入模式", ("✍️ 手寫板", "📷 拍照辨識", "📂 上傳圖片"))
show_debug = st.sidebar.checkbox("👁️ 顯示二值化影像 (Debug)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 靈敏度 (可調整)")
stroke_width = st.sidebar.slider("筆刷粗細", 5, 30, 20)
min_area = st.sidebar.slider("最小面積", 20, 500, 100)
min_density = st.sidebar.slider("最小密度", 0.05, 0.3, 0.10)

if mode_option == "✍️ 手寫板":
    st.markdown("### 請在下方寫出一串數字")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300,
            width=600,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        if st.button("開始辨識", type="primary"):
            if canvas_result.image_data is not None:
                img_data = canvas_result.image_data.astype(np.uint8)
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                result_img, nums = process_and_predict(img_bgr, min_area, min_density, show_debug)
                
                st.image(result_img, channels="BGR", use_container_width=True)
                if nums:
                    st.success("✅ 辨識成功！")
                    st.metric(label="偵測結果", value=" ".join(nums))
                else:
                    st.warning("⚠️ 未偵測到數字")

elif mode_option == "📷 拍照辨識":
    img_file = st.camera_input("拍照")
    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        result_img, nums = process_and_predict(cv2_img, min_area, min_density, show_debug)
        st.image(result_img, channels="BGR")
        if nums: st.metric(label="偵測結果", value=" ".join(nums))
        else: st.error("無法辨識")

elif mode_option == "📂 上傳圖片":
    uploaded_file = st.file_uploader("選擇圖片", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        if img_array.shape[-1] == 3: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        st.image(img_array, caption="原始圖", width=300)
        if st.button("辨識"):
            result_img, nums = process_and_predict(img_bgr, min_area, min_density, show_debug)
            st.image(result_img, channels="BGR")
            if nums: st.metric(label="偵測結果", value=" ".join(nums))
