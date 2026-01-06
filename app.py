import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pandas as pd # 用來做漂亮的表格

# ==========================================
#              設定與模型載入
# ==========================================
st.set_page_config(page_title="AI 手寫數字辨識 (V34 Full Debug)", page_icon="🔢", layout="wide")

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
    img = src.copy()
    m = cv2.moments(img, True)
    if m['m00'] < 0.1: return cv2.resize(img, (28, 28))
    cX, cY = m['m10'] / m['m00'], m['m01'] / m['m00']
    tX, tY = 14.0 - cX, 14.0 - cY
    M = np.float32([[1, 0, tX], [0, 1, tY]])
    return cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def split_touching_digits(roi_binary):
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

def process_and_predict(image_bgr, min_area, min_density, min_confidence, show_debug=False):
    """
    V34 更新：全方位診斷邏輯
    會畫出被過濾掉的框框：紫色(面積太小)、藍色(密度太低)、紅色(信心不足)
    """
    result_img = image_bgr.copy()
    
    # 1. 轉灰階 & 亮度檢查
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    max_val = np.max(gray)
    if max_val < 50:
        if show_debug: st.warning(f"⚠️ 畫面太暗 (最高亮度: {max_val})")
        return result_img, []

    # 2. 二值化
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_proc = cv2.dilate(thresh, None, iterations=1)
    
    if show_debug:
        st.image(binary_proc, caption="【Debug】二值化影像 (AI 眼中的世界)", width=300)
    
    # 3. 抓取物件 (注意：這裡先不濾掉面積，為了把太小的框也抓出來畫紫色框)
    nb, output, stats_cc, _ = cv2.connectedComponentsWithStats(binary_proc, connectivity=8)
    raw_boxes = sorted([stats_cc[i, :4] for i in range(1, nb)], key=lambda b: b[0])

    rois_to_pred = []
    coords_to_draw = []
    h_img, w_img = binary_proc.shape 

    for box in raw_boxes:
        x, y, w, h = box
        # 邊緣移除
        if x < 5 or y < 5 or (x + w) > w_img - 5 or (y + h) > h_img - 5: continue
        # 高度太扁的直接忽略，通常是雜訊線條
        if h < 20: continue 

        # 切割連字
        split_results = split_touching_digits(binary_proc[y:y+h, x:x+w])
        
        for offset_x, sub_roi in split_results:
            sh, sw = sub_roi.shape
            if sw == 0 or sh == 0: continue
            
            # --- [診斷邏輯開始] ---

            # 計算真實面積與密度
            n_white_pix = cv2.countNonZero(sub_roi)
            box_area = sw * sh
            density = n_white_pix / float(box_area)

            # 1. 檢查面積 (太小 -> 紫色框)
            if n_white_pix < min_area:
                if show_debug:
                    cv2.rectangle(result_img, (x+offset_x, y), (x+offset_x+sw, y+sh), (255, 0, 255), 1) # 紫色
                    cv2.putText(result_img, "Small", (x+offset_x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                continue # 跳過，不辨識

            # 2. 檢查密度 (太低 -> 藍色框)
            if density < min_density:
                if show_debug:
                    cv2.rectangle(result_img, (x+offset_x, y), (x+offset_x+sw, y+sh), (255, 0, 0), 1) # 藍色
                    cv2.putText(result_img, "Noise", (x+offset_x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                continue # 跳過，不辨識
            
            # --- [通過基本檢查，準備辨識] ---
            
            side = max(sw, sh)
            container = np.zeros((side+40, side+40), dtype=np.uint8)
            offset_y, offset_x_c = 20 + (side-sh)//2, 20 + (side-sw)//2
            container[offset_y:offset_y+sh, offset_x_c:offset_x_c+sw] = sub_roi
            
            final_roi = center_by_moments_cnn(cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA))
            final_roi_norm = np.expand_dims(final_roi.astype('float32') / 255.0, axis=-1)
            
            rois_to_pred.append(final_roi_norm)
            coords_to_draw.append((x + offset_x, y, sw, sh, sub_roi))

    detected_info = [] # 存詳細資料

    if len(rois_to_pred) > 0:
        predictions = cnn_model.predict(np.array(rois_to_pred), verbose=0)
        
        for i, pred_probs in enumerate(predictions):
            res_id = np.argmax(pred_probs)
            confidence = np.max(pred_probs)
            rx, ry, w, h, roi_original = coords_to_draw[i]
            
            # 3. 檢查信心 (太低 -> 紅色框)
            if confidence < min_confidence:
                if show_debug:
                    cv2.rectangle(result_img, (rx, ry), (rx+w, ry+h), (0, 0, 255), 1) # 紅色
                    label = f"{res_id}? ({int(confidence*100)}%)"
                    cv2.putText(result_img, label, (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            # --- [辨識成功] ---
            display_text = str(res_id)
            color = (0, 255, 0) # 綠色
            
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
            
            # 格式化信心 (轉成 %)
            conf_str = f"{int(confidence * 100)}%"
            
            # 收集詳細資訊
            detected_info.append({"數字": str(res_id), "信心度": conf_str, "修正": "*" in display_text})
            
            # 畫圖時加上信心度
            label = f"{display_text} ({conf_str})"
            cv2.rectangle(result_img, (rx, ry), (rx+w, ry+h), color, 2)
            cv2.putText(result_img, label, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return result_img, detected_info

# ==========================================
#              Streamlit UI 介面
# ==========================================
st.title("🔢 AI 手寫辨識 (含信心度分析)")

st.sidebar.header("🔧 設定")
mode_option = st.sidebar.selectbox("輸入模式", ("✍️ 手寫板", "📷 拍照辨識", "📂 上傳圖片"))

# 這裡的選項說明改了一下，強調會顯示忽略區域
show_debug = st.sidebar.checkbox("👁️ 顯示二值化/忽略區域 (Debug)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 靈敏度")
stroke_width = st.sidebar.slider("筆刷粗細", 5, 30, 20)
min_area = st.sidebar.slider("最小面積", 20, 500, 100)
min_density = st.sidebar.slider("最小密度", 0.05, 0.3, 0.10)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI 信心門檻")
st.sidebar.info("信心低於此數值的字會被忽略")
min_confidence = st.sidebar.slider("信心過濾器 (Confidence)", 0.5, 1.0, 0.60) 

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
                result_img, info_list = process_and_predict(img_bgr, min_area, min_density, min_confidence, show_debug)
                
                st.image(result_img, channels="BGR", use_container_width=True)
                
                if info_list:
                    st.success("✅ 辨識完成！")
                    nums_str = " ".join([item["數字"] for item in info_list])
                    st.metric(label="偵測結果", value=nums_str)
                    
                    st.markdown("##### 📊 詳細數據分析")
                    df = pd.DataFrame(info_list)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("⚠️ 未偵測到數字 (或信心不足，請開啟 Debug 檢查)")

elif mode_option == "📷 拍照辨識":
    img_file = st.camera_input("拍照")
    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        result_img, info_list = process_and_predict(cv2_img, min_area, min_density, min_confidence, show_debug)
        
        st.image(result_img, channels="BGR")
        if info_list:
             nums_str = " ".join([item["數字"] for item in info_list])
             st.metric(label="偵測結果", value=nums_str)
             
             st.markdown("##### 📊 詳細數據分析")
             st.dataframe(pd.DataFrame(info_list), use_container_width=True)
        else:
             st.error("無法辨識 (信心不足，請開啟 Debug 檢查)")

elif mode_option == "📂 上傳圖片":
    uploaded_file = st.file_uploader("選擇圖片", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        if img_array.shape[-1] == 3: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        st.image(img_array, caption="原始圖", width=300)
        
        if st.button("辨識"):
            result_img, info_list = process_and_predict(img_bgr, min_area, min_density, min_confidence, show_debug)
            st.image(result_img, channels="BGR")
            if info_list:
                nums_str = " ".join([item["數字"] for item in info_list])
                st.metric(label="偵測結果", value=nums_str)
                st.markdown("##### 📊 詳細數據分析")
                st.dataframe(pd.DataFrame(info_list), use_container_width=True)
