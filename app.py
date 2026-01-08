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
st.set_page_config(page_title="AI 手寫數字辨識 (V52 Merge)", page_icon="🔢", layout="wide")

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

# [V52] 新增：合併靠近的框框
def merge_nearby_boxes(boxes, threshold=20):
    if not boxes:
        return []
    
    # 根據 x 座標排序
    boxes.sort(key=lambda b: b[0])
    
    merged = []
    current_box = boxes[0] # [x, y, w, h]
    
    for next_box in boxes[1:]:
        cx, cy, cw, ch = current_box
        nx, ny, nw, nh = next_box
        
        # 計算水平距離 (右邊界 到 下一個的左邊界)
        distance = nx - (cx + cw)
        
        # 如果距離夠近，且垂直方向有重疊 (避免把上下兩行的字合併)
        # 簡單判定：下一字的中心點 y 座標，是否在當前字的 y 範圍內
        cy_center = cy + ch / 2
        ny_center = ny + nh / 2
        vertical_overlap = (ny < cy + ch) and (ny + nh > cy)

        if distance < threshold and vertical_overlap:
            # 執行合併：找出新的大框框邊界
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

def process_and_predict(image_bgr, min_area, min_density, min_confidence, box_padding, proc_mode, manual_thresh, dilation_iter, use_morph_close, merge_dist, show_debug):
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
        st.image(binary_proc, caption=f"【Debug】二值化影像 (處理後)", width=300)
    
    # 1. 先抓出所有框框
    nb, output, stats_cc, _ = cv2.connectedComponentsWithStats(binary_proc, connectivity=8)
    
    # 簡單過濾一下太小的雜訊 (這裡只濾極小的，主要過濾留到後面)
    raw_boxes = []
    for i in range(1, nb):
        x, y, w, h = stats_cc[i, :4]
        area = stats_cc[i, cv2.CC_STAT_AREA]
        # 放寬邊界檢查
        if x <= 1 or y <= 1 or (x + w) >= binary_proc.shape[1] - 1 or (y + h) >= binary_proc.shape[0] - 1: continue
        # 先不濾 area，等等合併完再濾
        raw_boxes.append([x, y, w, h])

    # 2. [V52] 執行「斷字合併」邏輯
    if merge_dist > 0:
        merged_boxes = merge_nearby_boxes(raw_boxes, threshold=merge_dist)
    else:
        merged_boxes = raw_boxes

    rois_to_pred = []
    coords_to_draw = []
    detected_info = []

    # 3. 對合併後的框框進行最後處理與辨識
    for box in merged_boxes:
        x, y, w, h = box
        
        # 這裡才切圖
        # 注意：因為合併後的框框可能包含多個不連通的區域，我們直接切那個方形範圍
        sub_roi = binary_proc[y:y+h, x:x+w]
        
        sh, sw = sub_roi.shape
        if sw == 0 or sh == 0: continue
        
        n_white_pix = cv2.countNonZero(sub_roi)
        box_area = sw * sh
        density = n_white_pix / float(box_area)

        # 最後過濾
        if n_white_pix < min_area: continue
        if density < min_density: continue
        
        side = max(sw, sh)
        container = np.zeros((side+40, side+40), dtype=np.uint8)
        offset_y, offset_x_c = 20 + (side-sh)//2, 20 + (side-sw)//2
        container[offset_y:offset_y+sh, offset_x_c:offset_x_c+sw] = sub_roi
        
        final_roi = center_by_moments_cnn(cv2.resize(container, (28, 28), interpolation=cv2.INTER_AREA))
        final_roi_norm = np.expand_dims(final_roi.astype('float32') / 255.0, axis=-1)
        
        rois_to_pred.append(final_roi_norm)
        coords_to_draw.append((x, y, w, h)) # 這裡不加 offset，因為我們是用 merge box 的座標

    if len(rois_to_pred) > 0:
        predictions = cnn_model.predict(np.array(rois_to_pred), verbose=0)
        
        for i, pred_probs in enumerate(predictions):
            res_id = np.argmax(pred_probs)
            confidence = np.max(pred_probs)
            rx, ry, w, h = coords_to_draw[i]
            
            if confidence < min_confidence:
                continue

            display_text = str(res_id)
            color = (0, 255, 0)
            
            roi_display = cv2.cvtColor(binary_proc[ry:ry+h, rx:rx+w], cv2.COLOR_GRAY2RGB)
            roi_display = cv2.bitwise_not(roi_display)

            current_id = len(detected_info) + 1

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

            cv2.rectangle(result_img, (p_x1, p_y1), (p_x2, p_y2), color, 2)
            cv2.putText(result_img, label, (p_x1, p_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return result_img, detected_info

# ==========================================
#              Streamlit UI 介面
# ==========================================
st.title("🔢 AI 手寫辨識 (V52 Merge)")

st.sidebar.header("🔧 設定")
mode_option = st.sidebar.selectbox("輸入模式", ("✍️ 手寫板", "📷 拍照辨識", "📂 上傳圖片"))

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

# [V52 新增] 斷字合併滑桿
st.sidebar.markdown("---")
merge_dist = st.sidebar.slider("🧲 斷字合併 (Merge)", 0, 50, 20, help="如果數字斷成兩半(如2斷成兩截)，調大這個數值可以把吸在一起")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 辨識設定")
min_confidence = st.sidebar.slider("信心過濾器", 0.0, 1.0, 0.40) 

st.sidebar.subheader("🎛️ 靈敏度 (重要)")
min_area = st.sidebar.slider("最小面積 (數字不見調這裡)", 10, 500, 50)
min_density = st.sidebar.slider("最小密度", 0.05, 0.3, 0.05)
show_debug = st.sidebar.checkbox("👁️ 顯示 Debug 資訊", value=False)

def run_app(source_image):
    result_img, info_list = process_and_predict(source_image, min_area, min_density, min_confidence, box_padding, proc_mode_sel, manual_thresh, dilation_iter, use_morph_close, merge_dist, show_debug)
    
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.image(result_img, channels="BGR", use_container_width=True, caption="辨識結果")
    
    with c2:
        if info_list:
            st.success(f"✅ 找到 {len(info_list)} 個數字")
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
            st.warning("⚠️ 畫面中未發現數字！")
            st.info("""
            **💡 小撇步：如何找回消失的字？**
            
            1. 📉 **調低「最小面積」** (試試看 20 或 30)
            2. 🧲 **調大「斷字合併」** (把斷掉的字吸在一起)
            3. 🐡 **調大「筆畫膨脹」**
            """)

# 介面渲染
if mode_option == "✍️ 手寫板":
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
            run_app(img_bgr)
        else:
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
        
        run_app(cv2_img)
