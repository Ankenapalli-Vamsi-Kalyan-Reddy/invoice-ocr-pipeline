import os
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output
from PIL import Image
import re
import re

import os
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output
from PIL import Image
import re
import xml.etree.ElementTree as ET
from datetime import datetime

# Optional: Tesseract path if not in environment
pytesseract.pytesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

sunoco_labels = {
    "ticket/bol": "Ticket/BOL"
}



    
def ocr_invoice_to_csv(input_path, output_csv_path):
    print("🔍 Running OCR on invoice...")

    images = []
    if input_path.lower().endswith(".pdf"):
        images = convert_from_path(input_path, poppler_path=r"C:\\poppler-24.08.0\\Library\\bin")
    else:
        images = [Image.open(input_path)]

    all_data = []

    for page_num, img in enumerate(images, start=1):
        ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(ocr_data['text'])

        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if text:
                all_data.append({
                    'text': text,
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i],
                    'page': page_num
                })

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ OCR complete. CSV saved to: {output_csv_path}")
    return df



def extract_bol_sunoco(df, keyword_text, y_gap_thresh=50, x_padding=50):
    keyword_rows = df[df['text'].str.lower().str.contains(keyword_text.lower())]
    for idx, row in keyword_rows.iterrows():
        print(f"  -> [{idx}] '{row['text']}' at (x={row['x']}, y={row['y']}, width={row['width']})")

    if keyword_rows.empty:
        return None

    for y_gap in [y_gap_thresh, 100, 150, 200]:
        for kw_idx, keyword in keyword_rows.iterrows():
            kx = keyword['x']
            ky = keyword['y']
            kw = keyword['width']
            

            # Same line
            same_line_candidates = df[
                (abs(df['y'] - ky) < 10) &
                (df['x'] >= kx + kw) &
                (df['x'] <= kx + kw + x_padding * 4)
            ]
            for c_idx, candidate in same_line_candidates.iterrows():
                text_clean = candidate['text'].replace(",", "").replace(".", "").replace("$", "").strip()
                if text_clean.isdigit():
                    return candidate['text']

            # Below the keyword, within y_gap
            below_candidates = df[
                (df['y'] > ky) &
                (df['y'] - ky < y_gap) &
                (df['x'] >= kx - x_padding) &
                (df['x'] <= kx + kw + x_padding)
            ]
            
            for c_idx, candidate in below_candidates.iterrows():
                text_clean = candidate['text'].replace(",", "").replace(".", "").replace("$", "").strip()
                if text_clean.isdigit():
                    return candidate['text']

    print("❌ No BOL value found after all y_gap thresholds.")
    return None

def extract_invoice_total_sunoco(df, y_gap_thresh=80, x_padding=30):
    df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)

    for i in range(len(df_sorted) - 1):
        row1 = df_sorted.loc[i]
        row2 = df_sorted.loc[i + 1]

        if 'invoice' in row1['text'].lower() and 'total' in row2['text'].lower():
            x_start = min(row1['x'], row2['x']) - x_padding
            x_end = max(row1['x'] + row1['width'], row2['x'] + row2['width']) + x_padding
            y_base = row2['y']

            candidates = df_sorted[
                (df_sorted['y'] > y_base) &
                (df_sorted['y'] - y_base < y_gap_thresh) &
                (df_sorted['x'] >= x_start) &
                (df_sorted['x'] <= x_end)
            ]

            for _, cand in candidates.iterrows():
                clean = cand['text'].replace(',', '').replace('$', '').replace('.', '')
                if clean.isdigit():
                    return cand['text']

    return None

def extract_from_sunoco_invoice(df):
    return {
        "company": "Sunoco, LLC",
        "ticket_bol": extract_bol_sunoco(df, sunoco_labels["ticket/bol"]),
        "invoice_total": extract_invoice_total_sunoco(df)
    }


# Metroplex extraction functions (assuming you have them defined already)
def extract_bol_metroplex(df, y_gap_thresh=200, x_margin=15):
    df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)
    bol_rows = df_sorted[df_sorted['text'].str.strip().str.lower() == 'bol']
    if bol_rows.empty:
        return None
    bol_row = bol_rows.iloc[0]
    bol_x = bol_row['x']
    bol_y = bol_row['y']
    candidates = df_sorted[
        (df_sorted['y'] > bol_y) &
        (df_sorted['y'] - bol_y < y_gap_thresh) &
        (abs(df_sorted['x'] - bol_x) <= x_margin)
    ]
    for _, row in candidates.iterrows():
        text = row['text'].strip().replace(",", "")
        if text.lower() in {"description", "gross", "amount", "price", "uom"}:
            continue
        if re.fullmatch(r"\d{4,}", text):
            return text
    return None

# def extract_document_total_metroplex(df):
#     df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)
#     for i in range(len(df_sorted) - 1):
#         if "document" in df_sorted.loc[i, 'text'].lower():
#             next_text = df_sorted.loc[i + 1, 'text'].lower()
#             if "total" in next_text:
#                 total_row = df_sorted.loc[i + 1]
#                 total_x_end = total_row['x'] + total_row['width']
#                 total_y = total_row['y']
#                 candidates = df_sorted[
#                     (df_sorted['page'] == total_row['page']) &
#                     (abs(df_sorted['y'] - total_y) < 5) &
#                     (df_sorted['x'] > total_x_end)
#                 ]
#                 min_dist = float('inf')
#                 value = None
#                 for _, c in candidates.iterrows():
#                     clean_text = c['text'].replace(',', '').replace('$', '').replace('.', '')
#                     if clean_text.isdigit():
#                         dist = c['x'] - total_x_end
#                         if dist < min_dist:
#                             min_dist = dist
#                             value = c['text']
#                 return value
#     return None


def extract_document_total_metroplex(df, y_tol=5):
    # --- Old method: adjacent "document" and "total" tokens ---
    df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)
    for i in range(len(df_sorted) - 1):
        if "document" in df_sorted.loc[i, 'text'].lower():
            next_text = df_sorted.loc[i + 1, 'text'].lower()
            if "total" in next_text:
                total_row = df_sorted.loc[i + 1]
                total_x_end = total_row['x'] + total_row['width']
                total_y = total_row['y']
                candidates = df_sorted[
                    (df_sorted['page'] == total_row['page']) &
                    (abs(df_sorted['y'] - total_y) < y_tol) &
                    (df_sorted['x'] > total_x_end)
                ]
                min_dist = float('inf')
                value = None
                for _, c in candidates.iterrows():
                    clean_text = c['text'].replace(',', '').replace('$', '').replace('.', '')
                    if clean_text.isdigit():
                        dist = c['x'] - total_x_end
                        if dist < min_dist:
                            min_dist = dist
                            value = c['text']
                if value:
                    return value

    # --- New method: "document" and "total" on the same line, not necessarily adjacent ---
    doc_rows = df_sorted[df_sorted['text'].str.lower().str.contains("document")]
    for _, doc_row in doc_rows.iterrows():
        doc_y = doc_row['y']
        doc_page = doc_row['page']
        total_candidates = df_sorted[
            (df_sorted['page'] == doc_page) &
            (abs(df_sorted['y'] - doc_y) <= y_tol) &
            (df_sorted['text'].str.lower().str.contains("total"))
        ]
        for _, total_row in total_candidates.iterrows():
            total_x_end = total_row['x'] + total_row['width']
            total_y = total_row['y']
            value_candidates = df_sorted[
                (df_sorted['page'] == doc_page) &
                (abs(df_sorted['y'] - total_y) <= y_tol) &
                (df_sorted['x'] > total_x_end)
            ]
            min_dist = float('inf')
            value = None
            for _, c in value_candidates.iterrows():
                clean_text = c['text'].replace(',', '').replace('$', '').replace('.', '')
                if clean_text.isdigit():
                    dist = c['x'] - total_x_end
                    if dist < min_dist:
                        min_dist = dist
                        value = c['text']
            if value:
                return value
    return None


def extract_from_metroplex_invoice(df):
    return {
        "company": "Metroplex",
        "ticket_bol": extract_bol_metroplex(df),
        "document_total": extract_document_total_metroplex(df)
    }


# Huguenot extraction functions
def extract_bol_huguenot(df, y_tol=5, y_gap_thresh=200, x_margin=20):
    df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)
    
    # Find all rows where text == 'bol'
    bol_rows = df_sorted[df_sorted['text'].str.strip().str.lower() == 'bol']
    
    for _, bol_row in bol_rows.iterrows():
        bol_x = bol_row['x']
        bol_y = bol_row['y']
        bol_page = bol_row['page']
        
        # Find "number" on the same line (y within tolerance), x greater than bol_x
        same_line_rows = df_sorted[
            (df_sorted['page'] == bol_page) &
            (abs(df_sorted['y'] - bol_y) <= y_tol) &
            (df_sorted['x'] > bol_x)
        ]
        
        number_rows = same_line_rows[same_line_rows['text'].str.lower().str.contains('number')]
        if number_rows.empty:
            continue
        
        # Use the first occurrence of "number"
        number_row = number_rows.iloc[0]
        number_x = number_row['x']
        number_y = number_row['y']
        
        # Look vertically below for numeric text aligned with BOL or Number (x close to bol_x or number_x)
        candidates = df_sorted[
            (df_sorted['page'] == bol_page) &
            (df_sorted['y'] > max(bol_y, number_y)) & 
            (df_sorted['y'] - max(bol_y, number_y) < y_gap_thresh) & 
            (
                (abs(df_sorted['x'] - bol_x) <= x_margin) | 
                (abs(df_sorted['x'] - number_x) <= x_margin)
            )
        ].sort_values(by='y')
        
        for _, candidate in candidates.iterrows():
            text = candidate['text'].strip().replace(',', '')
            if text.isdigit():
                return text

    return None

def extract_invoice_total_huguenot(df):
    df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)
    for i in range(len(df_sorted)):
        if 'invoice' in df_sorted.loc[i, 'text'].lower():
            if i + 1 < len(df_sorted) and 'total' in df_sorted.loc[i + 1, 'text'].lower():
                total_row = df_sorted.loc[i + 1]
                total_x_end = total_row['x'] + total_row['width']
                total_y = total_row['y']

                candidates = df_sorted[
                    (df_sorted['page'] == total_row['page']) &
                    (abs(df_sorted['y'] - total_y) < 10) &
                    (df_sorted['x'] > total_x_end)
                ]

                closest = None
                min_dist = float('inf')
                for _, cand in candidates.iterrows():
                    clean = cand['text'].replace(',', '').replace('$', '').replace('.', '')
                    if clean.isdigit():
                        dist = cand['x'] - total_x_end
                        if dist < min_dist:
                            min_dist = dist
                            closest = cand['text']
                return closest
    return None

def extract_from_huguenot_invoice(df):
    return {
        "company": "Huguenot Fuels",
        "ticket_bol": extract_bol_huguenot(df),
        "invoice_total": extract_invoice_total_huguenot(df)
    }


def extract_bol_lonewolf(df, y_tol=10):
    df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)

    # --- Case 1: "Ticket#:" as a single token (fuzzy) ---
    print("\n🔍 Checking for 'Ticket#:' as a single token (fuzzy match)...")
    pattern = re.compile(r"^ticket\s*#\s*:?", re.IGNORECASE)
    ticket_rows = df_sorted[df_sorted['text'].str.lower().str.match(pattern)]
    print(f"Found {len(ticket_rows)} rows matching 'Ticket#:' pattern.")

    for idx, row in ticket_rows.iterrows():
        y = row['y']
        page = row['page']
        x_end = row['x'] + row['width']
        print(f"➡️ 'Ticket#:'-like token at index {idx}, y={y}, page={page}, x_end={x_end}")
        same_line = df_sorted[
            (df_sorted['page'] == page) &
            (abs(df_sorted['y'] - y) <= y_tol) &
            (df_sorted['x'] > x_end)
        ].sort_values(by='x')
        for _, cand in same_line.iterrows():
            val = cand['text'].replace(",", "").replace("$", "").strip()
            print(f"   → Checking candidate: '{cand['text']}' cleaned: '{val}'")
            if val.isdigit():
                print(f"✅ Found BOL value: {val} (Case 1)")
                return val

    # --- Case 2: "Ticket" and "#:" as neighbors in any order ---
    print("\n🔍 Looking for 'Ticket' and '#:' neighbors in any order, then finding closest right-side value...")
    for i in range(len(df_sorted) - 1):
        a = df_sorted.loc[i, 'text'].strip().lower()
        b = df_sorted.loc[i + 1, 'text'].strip().lower()
        a_x = df_sorted.loc[i, 'x']
        a_w = df_sorted.loc[i, 'width']
        a_y = df_sorted.loc[i, 'y']
        a_page = df_sorted.loc[i, 'page']
        b_x = df_sorted.loc[i + 1, 'x']
        b_w = df_sorted.loc[i + 1, 'width']
        b_y = df_sorted.loc[i + 1, 'y']
        b_page = df_sorted.loc[i + 1, 'page']

        if ((a == 'ticket' and re.match(r'^\s*#\s*:?\s*$', b)) or
            (re.match(r'^\s*#\s*:?\s*$', a) and b == 'ticket')):
            print(f"➡️ Found neighbor pair at rows {i} and {i+1}: '{a}' / '{b}'")
            right_edge = max(a_x + a_w, b_x + b_w)
            print(f"   → Right edge for search: {right_edge}")
            candidates = df_sorted[
                (df_sorted['page'] == a_page) &
                (df_sorted['x'] > right_edge) &
                (
                    (abs(df_sorted['y'] - a_y) <= y_tol) |
                    (abs(df_sorted['y'] - b_y) <= y_tol)
                )
            ].copy()
            print(f"   → Candidates to the right: {len(candidates)}")
            candidates['x_dist'] = candidates['x'] - right_edge
            candidates = candidates.sort_values(by='x_dist')
            for _, cand in candidates.iterrows():
                val = cand['text'].replace(",", "").replace("$", "").strip()
                print(f"      → Checking right candidate: '{cand['text']}' cleaned: '{val}', x={cand['x']}, y={cand['y']}")
                if val.isdigit():
                    print(f"✅ Found BOL value: {val} (Case 2)")
                    return val
                
    print("\n🔍 Fallback: Looking for 'Ticket' immediately followed by '#:' in CSV order, then next value...")
    texts = df['text'].str.strip().str.lower().tolist()
    for i in range(len(texts) - 2):
        if texts[i] == "ticket" and texts[i+1].replace(" ", "") == "#:":
            next_val = df['text'].iloc[i + 2].replace(",", "").replace("$", "").strip()
            if next_val.isdigit():
                print(f"✅ Found BOL value: {next_val} (Case 3, sequential)")
                return next_val
    print("❌ No BOL value found for Lonewolf format.")
    return None



def extract_total_lonewolf(df, y_tol=10):
    df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)

    # Find "Total" text
    total_rows = df_sorted[df_sorted['text'].str.lower() == 'total']
    for _, row in total_rows.iterrows():
        page = row['page']
        y = row['y']
        x_end = row['x'] + row['width']

        # Find numeric candidate on same line to the right
        same_line = df_sorted[
            (df_sorted['page'] == page) &
            (abs(df_sorted['y'] - y) < y_tol) &
            (df_sorted['x'] > x_end)
        ].sort_values(by='x')

        for _, candidate in same_line.iterrows():
            text_clean = candidate['text'].replace(',', '').replace('$', '').strip()
            if text_clean.replace('.', '', 1).isdigit():
                return candidate['text']
    return None

def extract_from_lonewolf_invoice(df):
    return {
        "company": "Lonewolf",
        "ticket_bol": extract_bol_lonewolf(df),
        "invoice_total": extract_total_lonewolf(df)
    }
def extract_bol_marathon(df, y_gap_thresh=50, x_margin=30):
    """
    Extracts the BOL/Ticket number for Marathon:
    Finds the row with text "Ticket" and returns the first numeric value directly below it (in y), in the same x-column (within margin).
    """
    df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)
    ticket_rows = df_sorted[df_sorted['text'].str.strip().str.lower() == 'ticket']
    print(f"Marathon BOL extraction: Found {len(ticket_rows)} 'Ticket' rows.")
    for _, ticket_row in ticket_rows.iterrows():
        tx = ticket_row['x']
        ty = ticket_row['y']
        tpage = ticket_row['page']
        # Candidates: below, close in x, not too far down, same page
        candidates = df_sorted[
            (df_sorted['page'] == tpage) &
            (df_sorted['y'] > ty) &
            (df_sorted['y'] - ty < y_gap_thresh) &
            (abs(df_sorted['x'] - tx) <= x_margin)
        ].sort_values(by='y')
        for _, cand in candidates.iterrows():
            val = cand['text'].replace(",", "").replace("$", "").strip()
            print(f"   Checking candidate below 'Ticket': '{cand['text']}' cleaned: '{val}', x={cand['x']}, y={cand['y']}")
            if val.isdigit():
                print(f"✅ Marathon BOL found: {val}")
                return val
    print("❌ Marathon BOL not found.")
    return None


def extract_invoice_total_marathon(df, y_gap_thresh=120, x_padding=30, max_total_count=2):
    # First, try original Invoice+Total consecutive pattern
    df_sorted = df.sort_values(by=['page', 'y', 'x']).reset_index(drop=True)
    for i in range(len(df_sorted) - 1):
        row1 = df_sorted.loc[i]
        row2 = df_sorted.loc[i + 1]
        if 'invoice' in row1['text'].lower() and 'total' in row2['text'].lower():
            x_start = min(row1['x'], row2['x']) - x_padding
            x_end = max(row1['x'] + row1['width'], row2['x'] + row2['width']) + x_padding
            y_base = row2['y']
            print(f"Marathon Invoice Total: Found 'Invoice' at {i}, 'Total' at {i+1}, searching below y={y_base}, x=({x_start},{x_end})")
            candidates = df_sorted[
                (df_sorted['y'] > y_base) &
                (df_sorted['y'] - y_base < y_gap_thresh) &
                (df_sorted['x'] >= x_start) &
                (df_sorted['x'] <= x_end)
            ].sort_values(by='y')
            results = []
            for _, cand in candidates.iterrows():
                val = cand['text'].replace(",", "").replace("$", "").replace('"', '').strip()
                print(f"   Checking candidate below 'Invoice Total': '{cand['text']}' cleaned: '{val}', x={cand['x']}, y={cand['y']}")
                if val.replace('.', '', 1).isdigit():
                    results.append(cand['text'])
                    print(f"✅ Candidate numeric value found: {cand['text']}")
                    if len(results) == max_total_count:
                        break
            if results:
                return results
    # Fallback: Look for "Drafted for" pattern
    print("❌ Marathon Invoice Total not found. Falling back to 'Drafted for' pattern.")
    results = []
    text_list = df_sorted['text'].tolist()
    for i in range(len(text_list) - 2):
        # Look for the pattern: "Drafted", "for", <amount>
        if text_list[i].lower() == "drafted" and text_list[i+1].lower() == "for":
            candidate = text_list[i+2].replace(",", "").replace("$", "").replace('"', '').strip()
            if candidate.replace('.', '', 1).isdigit():
                print(f"✅ Found 'Drafted for' pattern with amount: {text_list[i+2]}")
                results.append(text_list[i+2])
                if len(results) == max_total_count:
                    break
    if results:
        return results
    print("❌ No 'Drafted for' pattern found.")
    return []


def extract_from_marathon_invoice(df):
    return {
        "company": "Marathon",
        "ticket_bol": extract_bol_marathon(df),
        "invoice_totals": extract_invoice_total_marathon(df)
    }

def detect_company(df):
    text_blob = " ".join(df['text'].str.lower())
    if "sunoco" in text_blob:
        return "Sunoco"
    elif "metroplex" in text_blob:
        return "Metroplex"
    elif "huguenot" in text_blob:
        return "Huguenot"
    elif "lonewolf" in text_blob:
        return "Lonewolf"
    elif "marathon" in text_blob:
        return "Marathon"
    return None

def process_invoice(input_path):
    output_dir = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/output"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ocr_output.csv")

    df = ocr_invoice_to_csv(input_path, csv_path)
    company = detect_company(df)

    if not company:
        filename = os.path.basename(input_path).lower()
        print("⚠️ Company not detected from OCR. Trying to detect from file name...")
        if "sunoco" in filename:
            company = "Sunoco"
        elif "metroplex" in filename:
            company = "Metroplex"
        elif "huguenot" in filename:
            company = "Huguenot"
        elif "lonewolf" in filename:
            company = "Lonewolf"
        elif "marathon" in filename:
            company = "Marathon"

    if not company:
        print("❌ Unable to detect known company in invoice from OCR or file name.")
        return

    print(f"🏷️ Detected invoice from: {company}")

    if company == "Sunoco":
        extracted = extract_from_sunoco_invoice(df)
    elif company == "Metroplex":
        extracted = extract_from_metroplex_invoice(df)
    elif company == "Huguenot":
        extracted = extract_from_huguenot_invoice(df)
    elif company == "Lonewolf":
        extracted = extract_from_lonewolf_invoice(df)
    elif company == "Marathon":
        extracted = extract_from_marathon_invoice(df)
    else:
        print("❌ No extraction logic defined for this company in this script.")
        return

    print("\n📄 Extracted Values:")
    for key, val in extracted.items():
        print(f"{key}: {val}")

CSV_PATH = r"C:\Users\ashri\OneDrive\Desktop\Csv_QB\input_files\SampleData(Sheet1).csv"  # Path to your CSV with all invoice rows

def load_invoice_csv(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        print(f"❌ Invoice CSV file not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Normalize column names for easier access
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def find_rows_by_bol(csv_df, bol_number):
    # Try both int and str match
    bol_col_candidates = [col for col in csv_df.columns if "invoice_nbr" in col or "bol" in col]
    if not bol_col_candidates:
        print("❌ No Invoice Nbr/BOL column found in CSV")
        return pd.DataFrame()
    bol_col = bol_col_candidates[0]
    mask = csv_df[bol_col].astype(str).str.strip() == str(bol_number).strip()
    rows = csv_df[mask]
    return rows

def round_amount(val):
    try:
        return round(float(val) + 1e-8, 2)
    except Exception:
        return None

def print_human_readable(rows, bol_number, invoice_total):
    if rows.empty:
        print(f"\n❌ No rows found in CSV for BOL/Invoice Nbr: {bol_number}")
        return

    print(f"\n🔎 Matched BOL/Invoice Nbr: {bol_number}")
    totals = []
    for idx, row in rows.iterrows():
        total = row.get('total', '')
        try:
            totals.append(float(str(total).replace(',', '')))
        except Exception:
            totals.append(None)

    sum_totals = sum([t for t in totals if t is not None])
    try:
        inv_total = float(str(invoice_total).replace(',', ''))
    except Exception:
        inv_total = None

    diff = round(inv_total - sum_totals, 2) if inv_total is not None else None

    # Print all but the last row
    for i, row in enumerate(rows.itertuples()):
        if i < len(rows) - 1:
            print(f"  Store: {row.store_name}, Date: {row.date}, Supplier: {row.supplier}, CSV Total: {totals[i]:,.2f}")
        else:
            # Adjust last row
            adjusted_total = totals[i] + diff if diff is not None else totals[i]
            print(f"  Store: {row.store_name}, Date: {row.date}, Supplier: {row.supplier}, CSV Total (adjusted): {adjusted_total:,.2f}")
            if diff is not None and abs(diff) >= 0.01:
                print(f"    (Adjusted by {diff:+.2f} to match invoice)")
    
    print(f"\nSum of CSV Totals (pre-adjustment): {sum_totals:,.2f}")
    print(f"Extracted Invoice Total: {invoice_total}")
    if diff is not None:
        print(f"Difference (Invoice - CSV summed total): {diff:+.2f}")
        if abs(diff) < 0.03:
            print("✅ Difference adjusted in final row.")
        else:
            print("⚠️ Difference is more than 2 cents!")
    else:
        print("⚠️ Could not parse invoice total for comparison.")

def rows_to_xml(rows, bol_number, invoice_total):
    root = ET.Element('InvoiceMatch')
    bol = ET.SubElement(root, 'BOL')
    bol.text = str(bol_number)
    extracted_total = ET.SubElement(root, 'ExtractedInvoiceTotal')
    extracted_total.text = str(invoice_total)
    totals = []
    for _, row in rows.iterrows():
        try:
            totals.append(float(str(row.get('total', '')).replace(',', '')))
        except Exception:
            totals.append(0.0)
    try:
        inv_total = float(str(invoice_total).replace(',', ''))
    except Exception:
        inv_total = None

    diff = round(inv_total - sum(totals), 2) if inv_total is not None else 0

    for i, (_, row) in enumerate(rows.iterrows()):
        entry = ET.SubElement(root, 'CSVEntry')
        for col in ['store_name', 'date', 'supplier']:
            val = row.get(col, '')
            child = ET.SubElement(entry, col.title().replace("_", ""))
            child.text = str(val)
        # Adjust total for last row
        if i == len(rows) - 1 and abs(diff) >= 0.01:
            adj_total = round(totals[i] + diff, 2)
            child = ET.SubElement(entry, 'Total')
            child.text = f"{adj_total:,.2f}"
        else:
            child = ET.SubElement(entry, 'Total')
            child.text = f"{totals[i]:,.2f}"
    return ET.tostring(root, encoding="unicode")

def watchdog(input_path, csv_path=CSV_PATH, processed_bols_path="./processed_bols.txt"):
    # 1. OCR and extract info as usual
    df = ocr_invoice_to_csv(input_path, "temp_ocr.csv")  # You can save to temp file
    company = detect_company(df)
    if not company:
        print("❌ Could not detect company.")
        return

    if company == "Sunoco":
        extracted = extract_from_sunoco_invoice(df)
    elif company == "Metroplex":
        extracted = extract_from_metroplex_invoice(df)
    elif company == "Huguenot":
        extracted = extract_from_huguenot_invoice(df)
    elif company == "Lonewolf":
        extracted = extract_from_lonewolf_invoice(df)
    elif company == "Marathon":
        extracted = extract_from_marathon_invoice(df)
    else:
        print("❌ No extraction logic for this company.")
        return

    bol_number = extracted.get("ticket_bol") or extracted.get("ticket/BOL")
    invoice_total = (
        extracted.get("invoice_total") or
        (extracted.get("invoice_totals")[0] if isinstance(extracted.get("invoice_totals"), list) else None) or
        extracted.get("document_total")
    )

    print("\n📄 Extracted Info from Invoice:")
    print(f"  Supplier: {company}")
    print(f"  BOL/Invoice Nbr: {bol_number}")
    print(f"  Invoice Total: {invoice_total}")

    # 2. Check if this BOL is already processed (avoid duplicate entry)
    processed_bols = set()
    if os.path.exists(processed_bols_path):
        with open(processed_bols_path, "r") as f:
            processed_bols = set(line.strip() for line in f.readlines())
    if bol_number and str(bol_number) in processed_bols:
        answer = input(f"⚠️ This invoice (BOL: {bol_number}) was already processed. Do you want to update it? (y/n): ").strip().lower()
        if answer != 'y':
            print("🛑 Skipping update as per user request.")
            return

    # 3. Load CSV and match by BOL/Invoice Nbr
    csv_df = load_invoice_csv(csv_path)
    if csv_df.empty:
        print("❌ Could not load invoice CSV.")
        return
    rows = find_rows_by_bol(csv_df, bol_number)
    print_human_readable(rows, bol_number, invoice_total)

    # >>> EXTRACT BILL DATE FROM CSV <<<
    if not rows.empty and 'date' in rows.columns:
        bill_date = str(rows.iloc[0]['date'])
        try:
            bill_date = pd.to_datetime(bill_date).strftime('%Y-%m-%d')
        except Exception:
            pass
    else:
        bill_date = None

    # 4. Print XML output
    print("\n--- XML Output ---")
    xml_str = rows_to_xml(rows, bol_number, invoice_total)
    print(xml_str)

    # 4b. Print and Save QBXML for QuickBooks
    print("\n--- QBXML for QuickBooks ---")
    if not rows.empty:
        qbxml = generate_bill_qbxml(
            vendor=company,
            ref_number=bol_number,
            invoice_total=invoice_total,
            rows=rows,
            bill_date=bill_date       # <<< pass the date!
        )
        print(qbxml)
        # Optional: Save QBXML to a file
        qbxml_file = f"qbxml_{bol_number}.xml"
        with open(qbxml_file, "w", encoding="utf-8") as qf:
            qf.write(qbxml)
            print(f"✅ QBXML saved to {qbxml_file}")
    else:
        print("❌ No rows found for QBXML generation.")

    # 5. Mark BOL as processed (add to processed_bols_path)
    if bol_number:
        with open(processed_bols_path, "a") as f:
            f.write(str(bol_number) + "\n")

def find_csv(folder):
    """Find the first .csv file in the given folder."""
    for file in os.listdir(folder):
        if file.lower().endswith('.csv'):
            return os.path.join(folder, file)
    return None

def batch_process_watchdog(folder):
    csv_path = find_csv(folder)
    if not csv_path:
        print("❌ No CSV file found in the folder!")
        return

    print(f"Using CSV: {csv_path}")

    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("❌ No PDF files found in the folder!")
        return

    for file in pdf_files:
        input_path = os.path.join(folder, file)
        print(f"\n--- Processing {input_path} ---")
        watchdog(input_path, csv_path=csv_path)

import xml.etree.ElementTree as ET

# Hierarchical mapping for AccountRef per FuelType
FUELTYPE_TO_ACCOUNT = {
    'REG': 'Cost of Goods Sold:Fuel:87-Regular cost',
    'ULT': 'Cost of Goods Sold:Fuel:93-ultimate cost',
    'PRE': 'Cost of Goods Sold:Fuel:93-premium cost',
    'DSL': 'Cost of Goods Sold:Fuel:Diesel cost',
}

def generate_bill_qbxml(vendor, ref_number, invoice_total, rows, bill_date=None):
    """
    vendor: str, e.g. "Sunoco"
    ref_number: str or int, e.g. "1314992"
    invoice_total: str or float, e.g. "22114.25"
    rows: pandas DataFrame with columns: store_name, fueltype, total, etc.
    bill_date: str, e.g. "2025-04-07" (YYYY-MM-DD) or None
    """
    # Safely convert amounts and compute adjustment
    amounts = []
    for _, row in rows.iterrows():
        try:
            amounts.append(float(str(row.get('total', 0)).replace(',', '')))
        except Exception:
            amounts.append(0.0)
    try:
        inv_total = float(str(invoice_total).replace(',', ''))
    except Exception:
        inv_total = sum(amounts)
    diff = round(inv_total - sum(amounts), 2)

    # QBXML root
    root = ET.Element('QBXML')
    msgs = ET.SubElement(root, 'QBXMLMsgsRq', onError="stopOnError")
    bill_add_rq = ET.SubElement(msgs, 'BillAddRq')
    bill_add = ET.SubElement(bill_add_rq, 'BillAdd')

    vendor_ref = ET.SubElement(bill_add, 'VendorRef')
    vendor_name = ET.SubElement(vendor_ref, 'FullName')
    vendor_name.text = str(vendor)

    ref = ET.SubElement(bill_add, 'RefNumber')
    ref.text = str(ref_number)

    # >>> Add date if present <<<
    if bill_date:
        txn_date = ET.SubElement(bill_add, 'TxnDate')
        txn_date.text = str(bill_date)

    #memo = ET.SubElement(bill_add, 'Memo')
    #memo.text = f"BOL {ref_number} | Supplier: {vendor}"

    # Add each invoice line as ExpenseLineAdd
    for i, row in enumerate(rows.itertuples()):
        fuel = str(getattr(row, 'fuel_type', '')).strip().upper()
        account_full = FUELTYPE_TO_ACCOUNT.get(fuel, 'Cost of Goods Sold')
        store = str(getattr(row, 'store_name', '')).strip()
        amount = amounts[i]
        # Adjust last row to match invoice total
        if i == len(rows) - 1:
            amount = round(amount + diff, 2)
        exp = ET.SubElement(bill_add, 'ExpenseLineAdd')
        acc = ET.SubElement(exp, 'AccountRef')
        acc_name = ET.SubElement(acc, 'FullName')
        acc_name.text = account_full
        amt = ET.SubElement(exp, 'Amount')
        amt.text = f"{amount:.2f}"
        class_ref = ET.SubElement(exp, 'ClassRef')
        class_name = ET.SubElement(class_ref, 'FullName')
        class_name.text = store
        #memo_exp = ET.SubElement(exp, 'Memo')
        #memo_exp.text = f"BOL {ref_number} | Fuel: {fuel} | Store: {store}"

    # Add XML declaration and QuickBooks version
    xml_str = '<?xml version="1.0" encoding="utf-8"?>\n<?qbxml version="13.0"?>\n' + ET.tostring(root, encoding="unicode")
    return xml_str


if __name__ == "__main__":
    folder = r"C:\Users\ashri\OneDrive\Desktop\Csv_QB\input_files"
    if not os.path.exists(folder):
        print("❌ Folder not found.")
    else:
        batch_process_watchdog(folder)


# if __name__ == "__main__":
#     input_path = "C:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/sunoco_1.pdf"
#     if not os.path.exists(input_path):
#         print("❌ File not found.")
#     else:
#         watchdog(input_path)


# if __name__ == "__main__":
#     #input_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/1106368-done.pdf"
#     #input_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/NC_Invoices/metroplex_2.pdf"
#     input_path = "C:/Users/ashri/OneDrive/Desktop/Csv_QB/NC_Invoices/NC Invoices/05. May/31/" \
#     "50563471_1116189-done.pdf"

#     if not os.path.exists(input_path):
#         print("❌ File not found.")
#     else:
#         process_invoice(input_path)

