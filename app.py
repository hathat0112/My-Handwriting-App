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
st.set_page_config(page_title="AI 手寫數字辨識 (V50 Lite)", page_icon="🔢", layout="wide")

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

# [V50] 移除了 apply_temperature_scaling 函數

def process_and_predict(image_bgr, min_area, min_density, min_confidence, box_padding, proc_mode, manual_thresh, use_smart_logic, dilation_iter, use_morph_close, show_debug):
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
    
    nb, output, stats_cc, _ = cv2.connectedComponentsWithStats(binary_proc, connectivity=8)
    raw_boxes = sorted([stats_cc[i, :4] for i in range(1, nb)], key=lambda b: b[0])

    rois_to_pred = []
    coords_to_draw = []
    detected_info = []

    for box in raw_boxes:
        x, y, w, h = box
        if x <= 1 or y <= 1 or (x + w) >= binary_proc.shape[1] - 1 or (y + h) >= binary_proc.shape[0] - 1: continue
        if h < 10: continue 

        split_results = split_touching_digits(binary_proc[y:y+h, x:x+w])
        
        for offset_x, sub_roi in split_results:
            sh, sw = sub_roi.shape
            if sw == 0 or sh == 0: continue
            
            n_white_pix = cv2.countNonZero(sub_roi)
            box_area = sw * sh
            density = n_white_pix / float(box_area)

            if n_white_pix < min_area:
                continue
            if density < min_density:
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
            # [V50] 移除溫度校準，直接使用原始預測機率
            res_id = np.argmax(pred_probs)
            confidence = np.max(pred_probs)
            rx, ry, w, h, roi_original = coords_to_draw[i]
            
            if confidence < min_confidence:
                continue

            display_text = str(res_id)
            color = (0, 255, 0)
            
            is_corrected = False
            if use_smart_logic:
                num_holes, hole_y = analyze_hole_geometry(roi_original)
                if res_id == 6:
                    if hole_y is not None and hole_y < 0.58: res_id, display_text, color = 0, "0*", (0, 255, 255)
                elif res_id == 2:
                    h_r, w_r = roi_original.shape
                    pts = cv2.findNonZero(roi_original[int(h_r*0.7):, :])
                    if pts is not None and cv2.boundingRect(pts)[2] < w_r * 0.5:
                        res_id, display_text, color = 7, "7*", (0, 255, 255)
                elif res_id == 4 or res_id == 9:
                    has_hole = (num_holes > 0)
                    if res_id == 9 and not has_hole: res_id, display_text, color = 4, "4*", (0, 255, 255)
                    elif res_id == 4 and has_hole and confidence < 0.95: res_id, display_text, color = 9, "9*", (0, 255, 255)
                
                if "*" in display_text:
                    is_corrected = True
            
            roi_display = cv2.cvtColor(roi_original, cv2.COLOR_GRAY2RGB)
            roi_display = cv2.bitwise_not(roi_display)

            current_id = len(detected_info) + 1

            detected_info.append({
                "id": current_id,
                "digit": str(res_id), 
                "confidence": float(confidence),
                "is_corrected": is_corrected,
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
st.title("🔢 AI 手寫辨識 (V50 Lite)")

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

box_padding = st.sidebar.slider("🖼️ 框框留白", 0, 30, 10, help="如果覺得綠色框框太貼，可以調大這個")
dilation_iter = st.sidebar.slider("🐡 筆畫膨脹 (變粗)", 0, 3, 2, help="【重要】如果字寫太細或斷斷續續，請把這個調大！")
use_morph_close = st.sidebar.checkbox("🩹 啟用斷筆修補", value=True, help="自動把斷掉的筆劃連起來")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 辨識邏輯")
use_smart_logic = st.sidebar.checkbox("🧠 啟用規則修正", value=True)
# [V50 修改] 移除了信心溫度滑桿
min_confidence = st.sidebar.slider("信心過濾器", 0.0, 1.0, 0.40) 

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 靈敏度 (重要)")
min_area = st.sidebar.slider("最小面積 (數字不見調這裡)", 10, 500, 50, help="【最重要】如果你寫的字不見了，請把這個數值「往左拉」！如果雜訊太多，請「往右拉」。")
min_density = st.sidebar.slider("最小密度", 0.05, 0.3, 0.05)
show_debug = st.sidebar.checkbox("👁️ 顯示 Debug 資訊", value=False)

def run_app(source_image):
    result_img, info_list = process_and_predict(source_image, min_area, min_density, min_confidence, box_padding, proc_mode_sel, manual_thresh, use_smart_logic, dilation_iter, use_morph_close, show_debug)
    
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
            
            請嘗試調整左邊側邊欄的設定：
            1. 📉 **調低「最小面積」** (試試看 20 或 30)
            2. 🐡 **調大「筆畫膨脹」** (試試看 2 或 3)
            3. 🖼️ 檢查 **影像處理模式** 是否選對 (拍照請選「拍照模式」)
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
