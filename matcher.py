import re
import cv2
import json
import csv
import difflib
import numpy as np
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    pass

def clean_alphanumeric(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def get_center(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (sum(xs) / 4.0, sum(ys) / 4.0)

def merge_bboxes(bboxes):
    if not bboxes: return None
    min_x = min([min(pt[0] for pt in bbox) for bbox in bboxes])
    min_y = min([min(pt[1] for pt in bbox) for bbox in bboxes])
    max_x = max([max(pt[0] for pt in bbox) for bbox in bboxes])
    max_y = max([max(pt[1] for pt in bbox) for bbox in bboxes])
    return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

def filter_spatial_outliers(boxes):
    if len(boxes) <= 1: return boxes
    centers = [get_center(b['bbox']) for b in boxes]
    median_x = np.median([c[0] for c in centers])
    median_y = np.median([c[1] for c in centers])
    avg_h = np.mean([b['bbox'][2][1] - b['bbox'][0][1] for b in boxes])
    avg_w = np.mean([b['bbox'][2][0] - b['bbox'][0][0] for b in boxes])
    
    thresh_x = max(avg_w * 5, 200) 
    thresh_y = max(avg_h * 5, 100) 
    
    filtered = []
    for box, c in zip(boxes, centers):
        if abs(c[0] - median_x) < thresh_x and abs(c[1] - median_y) < thresh_y:
            filtered.append(box)
    return filtered if filtered else boxes

def extract_qwen_items(data, path=""):
    results = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_path = f"{path}.{k}" if path else k
            results.extend(extract_qwen_items(v, new_path))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]"
            results.extend(extract_qwen_items(item, new_path))
    else:
        val = str(data).strip()
        if val and val.lower() not in ["none", "-", "null", ""]:
            results.append({"field": path, "value": val, "clean": clean_alphanumeric(val), "claimed_boxes": []})
    return results

def get_match_weight(ocr_text, qwen_text):
    """
    Returns (True/False, Weight).
    Higher weight wins! This prevents "30 LYRICA" from stealing "30".
    """
    ocr_clean = clean_alphanumeric(ocr_text)
    q_clean = clean_alphanumeric(qwen_text)
    
    if not ocr_clean or not q_clean: return False, 0
    if ocr_clean == q_clean: return True, 4 # 🏆 EXACT MATCH wins over everything
    
    ocr_words = str(ocr_text).split()
    for w in ocr_words:
        w_clean = clean_alphanumeric(w)
        if w_clean and w_clean == q_clean: 
            return True, 3 # 🏅 EXACT TOKEN MATCH (OCR line contains exact Qwen word)
            
    if ocr_clean in q_clean and len(ocr_clean) >= 2: return True, 2 # Substring 
    if q_clean in ocr_clean and len(q_clean) >= 3: return True, 2 # Substring
    
    for w in ocr_words:
        w_clean = clean_alphanumeric(w)
        if w_clean and len(w_clean) >= 3 and w_clean in q_clean:
            return True, 1
            
    # Fuzzy matching for dates/typos (e.g. 08072025 vs 080725)
    if len(q_clean) >= 5 and len(ocr_clean) >= 4:
        ratio = difflib.SequenceMatcher(None, ocr_clean, q_clean).ratio()
        if ratio > 0.75: # ~80% similarity catches date formats perfectly
            return True, 0.5 
            
    return False, 0

def match_single_page(qwen_page_dict, ocr_page_list):
    qwen_items = extract_qwen_items(qwen_page_dict)
    
    for i, box in enumerate(ocr_page_list):
        box['id'] = i
        box['candidates_raw'] = []
        box['candidates'] = []
        
    # 2. Find Candidates with weights
    for box in ocr_page_list:
        for q in qwen_items:
            matched, weight = get_match_weight(box.get('text', ''), q['value'])
            if matched:
                box['candidates_raw'].append((q, weight))
                
        # ONLY KEEP THE TOP WEIGHT CANDIDATES (Prevents substring "hoovering")
        if box['candidates_raw']:
            max_w = max(c[1] for c in box['candidates_raw'])
            box['candidates'] = [c[0] for c in box['candidates_raw'] if c[1] == max_w]
                
    # 3. Anchor Unambiguous Boxes
    for box in ocr_page_list:
        if len(box['candidates']) == 1:
            q = box['candidates'][0]
            q['claimed_boxes'].append(box)
            box['assigned'] = q
        else:
            box['assigned'] = None
            
    # 4. Resolve Ambiguous Boxes Spatially
    for box in ocr_page_list:
        if len(box['candidates']) > 1 and box['assigned'] is None:
            # Hungry Round-Robin: If multiple exact matches exist (e.g. two $2,976.35),
            # only allow fields that haven't claimed one yet to bid!
            hungry_candidates = [q for q in box['candidates'] if not q['claimed_boxes']]
            search_pool = hungry_candidates if hungry_candidates else box['candidates']
            
            best_q = None
            min_dist = float('inf')
            b_center = get_center(box['bbox'])
            
            for q in search_pool:
                if q['claimed_boxes']:
                    dist = min(((b_center[0] - get_center(ab['bbox'])[0])**2 + (b_center[1] - get_center(ab['bbox'])[1])**2)**0.5 for ab in q['claimed_boxes'])
                    if dist < min_dist:
                        min_dist = dist
                        best_q = q
            
            if best_q is None:
                best_q = search_pool[0]
                    
            best_q['claimed_boxes'].append(box)
            box['assigned'] = best_q
            
    # 5. Compile Final Bounding Boxes
    final_output = []
    for q in qwen_items:
        if q['claimed_boxes']:
            sane_boxes = filter_spatial_outliers(q['claimed_boxes'])
            bboxes = [b['bbox'] for b in sane_boxes]
            final_bbox = merge_bboxes(bboxes)
            final_output.append({
                "field": q['field'],
                "qwen_value": q['value'],
                "bbox": final_bbox,
                "matched_ocr_text": " | ".join([b['text'] for b in sane_boxes])
            })
        else:
            final_output.append({"field": q['field'], "qwen_value": q['value'], "bbox": None, "matched_ocr_text": None})
            
    return final_output

def export_to_csv(all_matched_results, output_csv_path="matched_data.csv"):
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Page", "Field", "Qwen_Value", "OCR_Matched_Text", "Bounding_Box"])
        for res in all_matched_results:
            writer.writerow([
                res.get('page', 'Unknown'),
                res['field'],
                res['qwen_value'],
                res['matched_ocr_text'] if res['matched_ocr_text'] else "NO MATCH",
                str(res['bbox']) if res['bbox'] else "None"
            ])

def highlight_and_save_pdf(input_document_path, qwen_full_data, ocr_full_data, output_pdf_path="highlighted_output.pdf"):
    print(f"Loading document: {input_document_path}")
    if input_document_path.lower().endswith(".pdf"):
        pil_images = convert_from_path(input_document_path)
    else:
        pil_images = [Image.open(input_document_path).convert("RGB")]
        
    annotated_pil_images = []
    all_pages_results = []
    
    for page_index, pil_img in enumerate(pil_images):
        page_num = page_index + 1
        print(f"\n--- Processing Page {page_num} ---")
        
        qwen_page_dict = qwen_full_data.get(f"page_{page_num}", qwen_full_data) # Fallback if not paginated
        ocr_page_list = [box for box in ocr_full_data if box.get('page') == page_num]
        
        if not ocr_page_list: ocr_page_list = ocr_full_data 
            
        matched_results = match_single_page(qwen_page_dict, ocr_page_list)
        
        for res in matched_results:
            res['page'] = page_num
            status = "[MATCH]" if res['bbox'] else "[MISS ]"
            print(f"{status} | Field: {res['field']} | Qwen: '{res['qwen_value']}' -> OCR: '{res['matched_ocr_text']}'")
            
        all_pages_results.extend(matched_results)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        for res in matched_results:
            if res['bbox']:
                x1, y1 = int(res['bbox'][0][0]), int(res['bbox'][0][1])
                x2, y2 = int(res['bbox'][2][0]), int(res['bbox'][2][1])
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{res['field']}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                yd = max(20, y1)
                cv2.rectangle(cv_img, (x1, yd - 20), (x1 + w, yd), (0, 255, 0), -1)
                cv2.putText(cv_img, label, (x1, yd - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        annotated_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        annotated_pil_images.append(annotated_pil)

    csv_path = output_pdf_path.replace(".pdf", ".csv").replace(".jpg", ".csv")
    export_to_csv(all_pages_results, csv_path)

    if annotated_pil_images:
        annotated_pil_images[0].save(output_pdf_path, save_all=True, append_images=annotated_pil_images[1:])
        print(f"\n✅ Final visual document written to -> {output_pdf_path}")
