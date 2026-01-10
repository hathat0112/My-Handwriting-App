# ==========================================
# 4. 模式 C: 上傳圖片專用邏輯 (Upload) - 強化準確版
# ==========================================
def run_upload_mode(erosion, dilation, min_conf):
    st.info("支援 JPG/PNG，系統會自動框選數字")
    
    file = st.file_uploader("選擇圖片", type=["jpg", "png", "jpeg"])
    
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_origin = cv2.imdecode(file_bytes, 1)
        h_orig, w_orig = img_origin.shape[:2]
        
        # 1. 影像增強 (Contrast Enhancement)
        # 增加對比度，讓文字更黑，背景更亮
        lab = cv2.cvtColor(img_origin, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # 2. 雙重二值化策略 (Dual Thresholding)
        # A計畫: 自適應閾值 (抓細節)
        thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
        # B計畫: Otsu 閾值 (抓主體)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 取交集：只有兩個方法都認為是黑字的地方才保留 (大幅減少雜訊)
        binary_combined = cv2.bitwise_and(thresh_adapt, thresh_otsu)
        
        # V65 形態學清理
        processed = v65_morphology(binary_combined, erosion, dilation)
        
        cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_count = 0
        display_img = img_origin.copy()
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 80: continue # 濾除太小的雜點
            x, y, w, h = cv2.boundingRect(c)
            
            # ==========================================
            # 🛑 準確度優化過濾器 (Accuracy Filters)
            # ==========================================
            
            # 1. 邊緣過濾：去除貼在圖片邊邊的雜訊
            if x < 5 or y < 5 or (x+w) > w_orig-5 or (y+h) > h_orig-5:
                continue

            # 2. 嚴格長寬比：數字很難寬於 1.1 倍 (除非是手寫很醜的 2 或 5)
            aspect_ratio = w / float(h)
            if aspect_ratio > 1.05: # 比之前更嚴格，直接濾掉大部分方塊中文字
                continue 
            if aspect_ratio < 0.2: # 太細長通常是雜訊線條
                continue

            # 3. 巨大物件過濾
            if w * h > (h_orig * w_orig * 0.08): 
                continue 

            # 4. 【關鍵】像素密度檢查 (Pixel Density)
            # 數字是線條組成的，所以黑色像素佔比應該在 20% ~ 55% 之間
            # 中文字筆畫多，通常會超過 55%；實心色塊會接近 100%
            roi_check = processed[y:y+h, x:x+w]
            density = cv2.countNonZero(roi_check) / (w * h)
            
            if density < 0.15: continue # 太空 (可能是雜訊圈圈)
            if density > 0.55: continue # 太滿 (通常是中文字或色塊)
            
            # ==========================================
            
            roi = processed[y:y+h, x:x+w]
            inp = preprocess_input(roi)
            pred = cnn_model.predict(inp, verbose=0)[0]
            
            # 對於容易誤判的數字 (1, 7)，提高門檻
            conf = np.max(pred)
            lbl = np.argmax(pred)
            
            final_conf_thresh = min_conf
            if lbl in [1, 7]: 
                final_conf_thresh += 0.15 # 對 1 和 7 要求更高信心
            
            if conf > final_conf_thresh:
                cv2.rectangle(display_img, (x,y), (x+w,y+h), (0,255,0), 2)
                # 加上底色讓文字更清楚
                label_text = f"{lbl}"
                (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(display_img, (x, y-lh-10), (x+lw, y), (0,255,0), -1)
                cv2.putText(display_img, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                detected_count += 1

        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # 顯示處理後的黑白圖 (Debug用)，讓使用者知道 AI 到底看到什麼
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(img_rgb, use_container_width=True, caption="辨識結果")
        with c2:
            st.image(processed, use_container_width=True, caption="[Debug] AI 視角 (二值化)")
            st.markdown(f"**共找到 {detected_count} 個數字**")
