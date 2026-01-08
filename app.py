import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pandas as pd
import math

# ==========================================
#              設定與模型載入
# ==========================================
st.set_page_config(page_title="AI 手寫數字辨識 (V58 Solidity)", page_icon="🔢", layout="wide")

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
#              狀態管理 (追蹤器)
# ==========================================
if 'tracker' not in st.session_state:
    st.session_state.tracker = {
        'next_id': 1,       
        'objects': []       
    }

def reset_tracker():
    st.session_state.tracker = {'next_id': 1, 'objects': []}

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

def merge_nearby_boxes(boxes, threshold=20):
    if not boxes: return []
    boxes.sort(key=lambda b: b[0])
    merged = []
    current_box = boxes[0] 
    for next_box in boxes[1:]:
        cx, cy, cw, ch = current_box
        nx, ny, nw, nh = next_box
        distance = nx - (cx + cw)
        vertical_overlap = (ny < cy + ch) and (ny + nh > cy)
        if distance < threshold and vertical_overlap:
            new_x = min(cx, nx)
            new_y = min(cy, ny)
            new_w = max(cx + cw, nx + nw) - new_x
            new_h = max(cy + ch, ny + nh) - new_y
            current_box = [new_x, new_y, new_w, new_h]
        else:
            merged.append(current_box)
            current_box = next_box
    merged.append(current_box)
    return merged

def update_tracker(current_boxes_coords):
    tracked_objects = st.session_state.tracker['objects']
    next_id = st.session_state.tracker['next_id']
    new_tracked_objects = []
    assigned_ids = [] 
    final_ids_for_boxes = []

    for box in current_boxes_coords:
        x, y, w, h = box
        cx, cy = x + w/2, y + h/2
        best_match_id = None
        min_dist = 999999
        for old_obj in tracked_objects:
            ox, oy = old_obj['center']
            dist = math.sqrt((cx - ox)**2 + (cy - oy)**2)
            if dist < 50 and old_obj['id'] not in assigned_ids:
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = old_obj['id']
        
        if best_match_id is not None:
            final_id = best_match_id
            assigned_ids.append(final_id)
        else:
            final_id = next_id
            next_id += 1
            
        final_ids_for_boxes.append(final_id)
        new_tracked_objects.append({'id': final_id, 'center': (cx, cy)})
    
    st.session_state.tracker['objects'] = new_tracked_objects
    st.session_state.tracker['next_id'] = next_id
    return final_ids_for_boxes

# [V58] 新增：扎實度 (Solidity) 與 凸包檢查
def is_valid_digit_shape(roi_binary, show_debug_info=False):
    contours, hierarchy = cv2.findContours(roi_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    
    # 找出最大的輪廓
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    if area < 10: return False # 太小
    
    # 1. 扎實度檢查 (Solidity)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False
    solidity = float(area) / hull_area
    
    # 中文字的筆劃通常很散，Solidity 會很低
    # 數字通常比較飽滿，Solidity 較高 (除了 1 和 7 可能較低，但通常也在 0.25 以上)
    if solidity < 0.25: 
        return False 

    # 2. 破洞檢查
    holes = 0
    if hierarchy is not None:
        for h in hierarchy[0]:
            if h[3] != -1:
                holes += 1
    if holes > 2: return False 

    # 3. 線條複雜度 (Crossing Number)
    h, w = roi_binary.shape
    check_rows = [int(h*0.25), int(h*0.5), int(h*0.75)]
    for r in check_rows:
        row_pixels = roi_binary[r, :]
        transitions = 0
        prev_val = 0
        for val in row_pixels:
            if val > 127 and prev_val <= 127:
                transitions += 1
            prev_val = val
        if transitions > 3: return False

    check_cols = [int(w*0.25), int(w*0.5), int(w*0.75)]
    for c in check_cols:
        col_pixels = roi_binary[:, c]
        transitions = 0
        prev_val = 0
        for val in col_pixels:
            if val > 127 and prev_val <= 127:
                transitions += 1
            prev_val = val
        if transitions > 3: return False

    return True

def process_and_predict(image_bgr, min_area, min_density, min_confidence, box_padding, proc_mode, manual_thresh, dilation_iter, use_morph_close, merge_dist, use_tracking, use_strict_filter, show_debug):
    result_img = image_bgr.copy()
    h_img_full, w_img_full = result_img.shape[:2]
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

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

    if use_morph_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_proc = cv2.morphologyEx(binary_proc, cv2.MORPH_CLOSE, kernel, iterations=1)

    if dilation_iter > 0:
        binary_proc = cv2.dilate(binary_proc, None, iterations=dilation_iter)
    
    if show_debug:
        st.image(binary_proc, caption=f"【Debug】二值化影像", width=300)
    
    nb, output, stats_cc, _ = cv2.connectedComponentsWithStats(binary_proc, connectivity=8)
    
    raw_boxes = []
    for i in range(1, nb):
        x, y, w, h = stats_cc[i, :4]
        
        # [V58] 邊界過濾 (Border Check)
        # 如果框框貼到圖片的最邊緣 (誤差 2 pixel)，很有可能是切割雜訊或滿版文字，直接丟掉
        if x <= 2 or y <= 2 or (x + w) >= w_img_full - 2 or (y + h) >= h_img_full - 2:
            continue
            
        # 形狀過濾
        if use_strict_filter:
            aspect_ratio = w / float(h)
            if aspect_ratio > 3.0 or aspect_ratio < 0.1:
                continue

        raw_boxes.append([x, y, w, h])

    if merge_dist > 0:
        processing_boxes = merge_nearby_boxes(raw_boxes, threshold=merge_dist)
    else:
        processing_boxes = raw_boxes

    if not use_tracking:
        processing_boxes.sort(key=lambda b: b[0])

    rois_to_pred = []
    coords_to_draw = []
    valid_boxes = [] 

    for box in processing_boxes:
        x, y, w, h = box
        if w * h < min_area: continue

        sub_roi = binary_proc[y:y+h, x:x+w]
        sh, sw = sub_roi.shape
        if sw == 0 or sh == 0: continue
        
        # [V58] 呼叫扎實度檢查
        if use_strict_filter:
            if not is_valid_digit_shape(sub_roi):
                continue

        n_white_pix = cv2.countNonZero(sub_roi)
        box_area = sw * sh
        density = n_white_pix / float(box_area)

        if n_white_pix < min_area: continue
        if density < min_density: continue
        
        side = max(sw, sh)
        container = np.zeros((side+40, side+40), dtype=np.uint8)
        offset_y, offset_x_c = 20 + (side-sh)//2, 20 + (side-sw)//2
        container[offset_y:offset_y+sh, offset_x_c:offset_x_c+sw] = sub_roi
        
        final_roi = center_by_moments_cnn(cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA))
        final_roi_norm = np.expand_dims(final_roi.astype('float32') / 255.0, axis=-1)
        
        rois_to_pred.append(final_roi_norm)
        coords_to_draw.append((x, y, w, h))
        valid_boxes.append([x, y, w, h])

    final_ids = []
    if use_tracking:
        final_ids = update_tracker(valid_boxes)
    else:
        final_ids = list(range(1, len(valid_boxes) + 1))

    detected_info = []

    if len(rois_to_pred) > 0:
        predictions = cnn_model.predict(np.array(rois_to_pred), verbose=0)
        
        for i, pred_probs in enumerate(predictions):
            res_id = np.argmax(pred_probs)
            confidence = np.max(pred_probs)
            rx, ry, w, h = coords_to_draw[i]
            
            threshold = min_confidence
            if use_strict_filter:
                threshold = max(0.85, min_confidence)

            if confidence < threshold:
                continue

            current_id = final_ids[i] 

            roi_display = cv2.cvtColor(binary_proc[ry:ry+h, rx:rx+w], cv2.COLOR_GRAY2RGB)
            roi_display = cv2.bitwise_not(roi_display)

            detected_info.append({
                "id": current_id,
                "digit": str(res_id), 
                "confidence": float(confidence),
                "roi_img": roi_display
            })
            
            label = f"#{current_id}"
            pad = box_padding
            p_x1 = max(0, rx - pad)
            p_y1 = max(0, ry - pad)
            p_x2 = min(w_img_full, rx + w + pad)
            p_y2 = min(h_img_full, ry + h + pad)

            cv2.rectangle(result_img, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2)
            cv2.putText(result_img, label, (p_x1, p_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    detected_info.sort(key=lambda x: x['id'])
            
    return result_img, detected_info

# ==========================================
#              Streamlit UI 介面
# ==========================================
st.title("🔢 AI 手寫辨識 (V58 Solidity)")

st.sidebar.header("🔧 設定")
mode_option = st.sidebar.selectbox("輸入模式", ("✍️ 手寫板", "📷 拍照辨識", "📂 上傳圖片"))

if 'last_mode' not in st.session_state:
    st.session_state.last_mode = mode_option
if st.session_state.last_mode != mode_option:
    reset_tracker()
    st.session_state.last_mode = mode_option

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ 影像處理")
proc_mode_sel = st.sidebar.radio(
    "選擇演算法",
    ("otsu", "adaptive", "manual"),
    format_func=lambda x: {
        "otsu": "標準模式 (適合純黑手寫板)",
        "adaptive": "📄 拍照模式 (抗陰影)",
        "manual": "🎚️ 手動門檻"
    }[x],
    index=1 if mode_option != "✍️ 手寫板" else 0
)
if proc_mode_sel == "manual":
    manual_thresh = st.sidebar.slider("二值化門檻", 0, 255, 127)
else:
    manual_thresh = 127

box_padding = st.sidebar.slider("🖼️ 框框留白", 0, 30, 10)
dilation_iter = st.sidebar.slider("🐡 筆畫膨脹 (變粗)", 0, 3, 2)
use_morph_close = st.sidebar.checkbox("🩹 啟用斷筆修補", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🧲 進階修復")
enable_merge = st.sidebar.checkbox("啟用斷字合併", value=False)
merge_dist = 0
if enable_merge:
    merge_dist = st.sidebar.slider("合併距離 (像素)", 5, 50, 20)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 過濾設定")
use_strict_filter = st.sidebar.checkbox("🛡️ 嚴格過濾非數字", value=True, help="【強烈建議開啟】使用幾何扎實度與破洞檢查，專門過濾中文字與複雜背景。")

min_confidence = st.sidebar.slider("信心過濾器", 0.0, 1.0, 0.40) 

st.sidebar.subheader("🎛️ 靈敏度 (重要)")
min_area = st.sidebar.slider("最小面積 (數字不見調這裡)", 10, 500, 100) # [V58] 預設調高到 100，避免抓到梗圖裡的小雜點
min_density = st.sidebar.slider("最小密度", 0.05, 0.3, 0.05)
show_debug = st.sidebar.checkbox("👁️ 顯示 Debug 資訊", value=False)

def run_app(source_image, use_tracking=False):
    result_img, info_list = process_and_predict(
        source_image, min_area, min_density, min_confidence, box_padding, 
        proc_mode_sel, manual_thresh, dilation_iter, use_morph_close, merge_dist, 
        use_tracking, use_strict_filter, show_debug
    )
    
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.image(result_img, channels="BGR", use_container_width=True, caption="辨識結果")
    
    with c2:
        if info_list:
            st.success(f"✅ 找到 {len(info_list)} 個數字")
            if use_tracking:
                if st.button("🔄 清除編號記憶 (Reset ID)"):
                    reset_tracker()
                    st.rerun()

            st.markdown("### 詳細結果")
            with st.container(height=500):
                for item in info_list:
                    cols = st.columns([1, 1, 2])
                    with cols[0]:
                        st.caption(f"#{item['id']}")
                        st.image(item['roi_img'], width=50)
                    with cols[1]:
                        st.metric("數字", item['digit'])
                    with cols[2]:
                        conf = item['confidence']
                        st.caption(f"信心: {int(conf*100)}%")
                        st.progress(conf)
                    st.divider()
        else:
            if use_strict_filter:
                st.warning("⚠️ 未發現數字 (已過濾雜訊)")
                st.info("系統偵測到畫面太複雜（可能是中文或梗圖），已自動忽略。")
            else:
                st.warning("⚠️ 畫面中未發現數字！")

# 介面渲染
if mode_option == "✍️ 手寫板":
    st.info("💡 在手寫板模式下，系統會依照你寫的順序編號！")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=20, 
        stroke_color="#FFFFFF", 
        background_color="#000000", 
        height=300, 
        width=600, 
        drawing_mode="freedraw", 
        key="canvas",
        update_streamlit=True
    )
    
    if canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) > 0:
            img_data = canvas_result.image_data.astype(np.uint8)
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            run_app(img_bgr, use_tracking=True)
        else:
            reset_tracker()
            st.info("請在畫布上寫字...")

elif mode_option in ["📷 拍照辨識", "📂 上傳圖片"]:
    if mode_option == "📷 拍照辨識":
        file = st.camera_input("拍照")
    else:
        file = st.file_uploader("選擇圖片", type=["jpg", "png"])
        
    if file:
        bytes_data = file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if mode_option == "📂 上傳圖片": 
            st.image(cv2_img, caption="原始圖", width=200, channels="BGR")
        
        run_app(cv2_img, use_tracking=False)
