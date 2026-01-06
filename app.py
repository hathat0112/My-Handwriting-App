import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pandas as pd

# ==========================================
#              設定與模型載入
# ==========================================
st.set_page_config(page_title="AI 手寫數字辨識 (V37 Inspection)", page_icon="🔢", layout="wide")

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

def process_and_predict(image_bgr, min_area, min_density, min_confidence, proc_mode="adaptive", manual_thresh=127, show_debug=False):
    result_img = image_bgr.copy()
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # V35/V36 處理模式
    if proc_mode == "adaptive":
        binary_proc = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 10)
    elif proc_mode == "manual":
        _, thresh = cv2.threshold(blur, manual_thresh, 255, cv2.THRESH_BINARY_INV)
        binary_proc = thresh
    else: # "otsu"
        if np.mean(gray) > 127:
            flag = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        else:
            flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        _, thresh = cv2.threshold(blur, 0, 255, flag)
        binary_proc = thresh

    binary_proc = cv2.dilate(binary_proc, None, iterations=1)
    
    if show_debug:
        st.image(binary_proc, caption=f"【Debug】二值化影像", width=300)
    
    nb, output, stats_cc, _ = cv2.connectedComponentsWithStats(binary_proc, connectivity=8)
    raw_boxes = sorted([stats_cc[i, :4] for i in range(1, nb)], key=lambda b: b[0])

    rois_to_pred = []
    coords_to_draw = []
    detected_info = []

    for box in raw_boxes:
        x, y, w, h = box
        if x < 5 or y < 5 or (x + w) > binary_proc.shape[1] - 5 or (y + h) > binary_proc.shape[0] - 5: continue
        if h < 20: continue 

        split_results = split_touching_digits(binary_proc[y:y+h, x:x+w])
        
        for offset_x, sub_roi in split_results:
            sh, sw = sub_roi.shape
            if sw == 0 or sh == 0: continue
            
            n_white_pix = cv2.countNonZero(sub_roi)
            box_area = sw * sh
            density = n_white_pix / float(box_area)

            # Debug 畫框
            if n_white_pix < min_area:
                if show_debug: cv2.rectangle(result_img, (x+offset_x, y), (x+offset_x+sw, y+sh), (255, 0, 255), 1)
                continue
            if density < min_density:
                if show_debug: cv2.rectangle(result_img, (x+offset_x, y), (x+offset_x+sw, y+sh), (255, 0, 0), 1)
                continue
            
            side = max(sw, sh)
            container = np.zeros((side+40, side+40), dtype=np.uint8)
            offset_y, offset_x_c = 20 + (side-sh)//2, 20 + (side-sw)//2
            container[offset_y:offset_y+sh, offset_x_c:offset_x_c+sw] = sub_roi
            
            final_roi = center_by_moments_cnn(cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA))
            final_roi_norm = np.expand_dims(final_roi.astype('float32') / 255.0, axis=-1)
            
            rois_to_pred.append(final_roi_norm)
            coords_to_draw.append((x + offset_x, y, sw, sh, sub_roi))

    if len(rois_to_pred) > 0:
        predictions = cnn_model.predict(np.array(rois_to_pred), verbose=0)
        
        for i, pred_probs in enumerate(predictions):
            res_id = np.argmax(pred_probs)
            confidence = np.max(pred_probs)
            rx, ry, w, h, roi_original = coords_to_draw[i]
            
            if confidence < min_confidence:
                if show_debug:
                    cv2.rectangle(result_img, (rx, ry), (rx+w, ry+h), (0, 0, 255), 1)
                continue

            display_text = str(res_id)
            color = (0, 255, 0)
            
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
            
            # [V37] 儲存切片圖片，為了在介面上顯示
            # 必須把 roi_original (只有黑白) 轉成 RGB 格式方便顯示
            roi_display = cv2.cvtColor(roi_original, cv2.COLOR_GRAY2RGB)
            roi_display = cv2.bitwise_not(roi_display) # 反轉顏色變成白底黑字，比較好閱讀

            detected_info.append({
                "id": len(detected_info) + 1,
                "digit": str(res_id), 
                "confidence": float(confidence),
                "is_corrected": "*" in display_text,
                "roi_img": roi_display # 存下圖片
            })
            
            label = f"{display_text} ({int(confidence*100)}%)"
            cv2.rectangle(result_img, (rx, ry), (rx+w, ry+h), color, 2)
            cv2.putText(result_img, label, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return result_img, detected_info

# ==========================================
#              Streamlit UI 介面
# ==========================================
st.title("🔢 AI 手寫辨識 (V37 詳細檢測)")

st.sidebar.header("🔧 設定")
mode_option = st.sidebar.selectbox("輸入模式", ("✍️ 手寫板", "📷 拍照辨識", "📂 上傳圖片"))

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ 影像處理模式")
proc_mode_sel = st.sidebar.radio(
    "選擇演算法",
    ("otsu", "adaptive", "manual"),
    format_func=lambda x: {
        "otsu": "標準模式 (適合純黑手寫板)",
        "adaptive": "📄 紙張/拍照模式 (抗陰影)",
        "manual": "🎚️ 手動門檻"
    }[x],
    index=1 if mode_option != "✍️ 手寫板" else 0
)
if proc_mode_sel == "manual":
    manual_thresh = st.sidebar.slider("二值化門檻", 0, 255, 127)
else:
    manual_thresh = 127

show_debug = st.sidebar.checkbox("👁️ 顯示 Debug 資訊", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 靈敏度")
stroke_width = st.sidebar.slider("筆刷粗細", 5, 30, 20)
min_area = st.sidebar.slider("最小面積", 20, 500, 100)
min_density = st.sidebar.slider("最小密度", 0.05, 0.3, 0.10)
min_confidence = st.sidebar.slider("信心過濾器", 0.5, 1.0, 0.60) 

def run_app(source_image):
    result_img, info_list = process_and_predict(source_image, min_area, min_density, min_confidence, proc_mode_sel, manual_thresh, show_debug)
    
    st.image(result_img, channels="BGR", use_container_width=True)
    
    if info_list:
        st.success(f"✅ 成功辨識出 {len(info_list)} 個數字！")
        st.markdown("---")
        st.subheader("🔍 詳細檢測報告")

        # [V37] 顯示詳細的清單列表
        for item in info_list:
            with st.container():
                # 分成三欄：[編號/圖片] - [預測結果] - [進度條]
                c1, c2, c3 = st.columns([1, 1, 3])
                
                with c1:
                    st.caption(f"編號 #{item['id']}")
                    # 顯示 AI 切下來的那個字的圖片
                    st.image(item['roi_img'], width=60, clamp=True)
                
                with c2:
                    # 顯示大大的數字
                    st.metric("預測數字", item['digit'], delta="邏輯修正" if item['is_corrected'] else None)
                
                with c3:
                    # 顯示進度條
                    conf = item['confidence']
                    st.markdown(f"**信心度: {int(conf*100)}%**")
                    st.progress(conf)
                    
                    # 給一點文字評語
                    if conf > 0.9:
                        st.caption("🌟 信心十足")
                    elif conf > 0.7:
                        st.caption("✅ 還算確定")
                    else:
                        st.caption("⚠️ 有點猶豫，建議重寫")
                
                st.divider() # 分隔線

    else:
        st.warning("⚠️ 未偵測到數字，請調整模式或靈敏度。")

# 介面渲染
if mode_option == "✍️ 手寫板":
    col1, col2 = st.columns([2, 1])
    with col1:
        canvas_result = st_canvas(fill_color="rgba(255, 165, 0, 0.3)", stroke_width=stroke_width, stroke_color="#FFFFFF", background_color="#000000", height=300, width=600, drawing_mode="freedraw", key="canvas")
    with
