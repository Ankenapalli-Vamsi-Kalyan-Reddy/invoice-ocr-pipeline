import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path  # Convert PDF pages to images
from pytesseract import Output        # To get OCR results with detailed info
import os                             # For file path operations

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\poppler-24.08.0\Library\bin"

#pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/lonewolf.pdf"
pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/metroplex.pdf"
#pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/huguenot.pdf"
#pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/marathon.pdf"
#pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/sunoco.pdf"

output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

pages = convert_from_path(pdf_path, poppler_path=poppler_path)

for i, page_image in enumerate(pages):
    image_filename = os.path.join(output_folder, f"page_{i+1}.png")
    page_image.save(image_filename)
    ocr_data = pytesseract.image_to_data(page_image, output_type=Output.DATAFRAME)
    ocr_data = ocr_data[ocr_data.text.notna() & (ocr_data.text.str.strip() != "")]
    print(f"\n--- OCR data for page {i+1} ---")
    for index, row in ocr_data.iterrows():
        x, y, w, h = row['left'], row['top'], row['width'], row['height']
        word = row['text']
        conf = row['conf']
        print(f"Word: '{word}', Confidence: {conf}, BBox: (x={x}, y={y}, w={w}, h={h})")
    csv_filename = os.path.join(output_folder, f"page_{i+1}_ocr.csv")
    ocr_data.to_csv(csv_filename, index=False)

pdf_file_name = os.path.splitext(os.path.basename(pdf_path))[0].lower()
ocr_output_folder = "c:/Users/ashri/OneDrive/Desktop/Csv_QB/output_images"

if "sunoco" in pdf_file_name:
    group1 = [
        {"left": 911, "top": 42, "width": 178, "height": 36},
        {"left": 1106, "top": 42, "width": 66, "height": 30},
    ]
    group2 = [
        {"left": 1999, "top": 327, "width": 110, "height": 21},
    ]
    # Added group3 (e.g., for total amount)
    group3 = [
        {"left": 2000, "top": 350, "width": 100, "height": 20},
    ]
    # Added group4 (e.g., for date)
    group4 = [
        {"left": 1792, "top": 283, "width": 118, "height": 21},
        
    ]
    # Added group5 (e.g., for BOL number)
    group5 = [
        {"left": 291, "top": 741, "width": 81, "height": 17},
    ] #291	741	81	17

    file1 = os.path.join(ocr_output_folder, "page_1_ocr.csv")
    file2 = os.path.join(ocr_output_folder, "page_2_ocr.csv")
    
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("❌ One or both OCR CSV files for Sunoco are missing!")
        exit()

    df_group1 = pd.read_csv(file1)
    df_group2 = pd.read_csv(file2)
    df_group3 = pd.read_csv(file2)  # Assuming same page as group2
    df_group4 = pd.read_csv(file1)
    df_group5 = pd.read_csv(file1)

elif "marathon" in pdf_file_name:
    group1 = [
        {"left": 82, "top": 109, "width": 183, "height": 31},
        {"left": 282, "top": 109, "width": 197, "height": 31},
        {"left": 494, "top": 110, "width": 181, "height": 37},
    ]
    group2 = [
        {"left": 405, "top": 1693, "width": 97, "height": 22},
    ]
    group3 = [
        {"left": 405, "top": 1720, "width": 78, "height": 21},
    ]
    group4 = [  # date
        {"left": 1217, "top": 301, "width": 112, "height": 20},
    ]
    group5 = [  # BOL number
        {"left": 291, "top": 937, "width": 71, "height": 15},
    ] # 291	937	71	15
    df_group1 = df_group2 = df_group3 = df_group4 = df_group5 = pd.read_csv(os.path.join(ocr_output_folder, "page_1_ocr.csv"))

elif "lonewolf" in pdf_file_name:
    group1 = [
        {"left": 178, "top": 406, "width": 143, "height": 25},
        {"left": 331, "top": 406, "width": 163, "height": 25},
        {"left": 505, "top": 406, "width": 53, "height": 25},
    ]
    group2 = [
        {"left": 1392, "top": 1636, "width": 130, "height": 26},
    ]
    group4 = [  # date
        {"left": 557, "top": 907, "width": 61, "height": 18},
        #557	907	61	18
    ]
    group5 = [  # BOL number
        {"left": 499, "top": 1218, "width": 44, "height": 17},
    ]#1275	451	86	16	499	1218	44	17
    df_group1 = df_group2 = df_group4 = df_group5 = pd.read_csv(os.path.join(ocr_output_folder, "page_1_ocr.csv"))

elif "metroplex" in pdf_file_name:
    group1 = [
        {"left": 153, "top": 539, "width": 161, "height": 22},
    ]
    group2 = [
        {"left": 860, "top": 2082, "width": 102, "height": 22},
    ]
    group4 = [
        {"left": 292, "top": 389, "width": 106, "height": 23},
    ]
    
    group5 = [
        {"left": 151, "top": 1373, "width": 80, "height": 22},
    ]
    df_group1 = df_group2 = df_group4 = df_group5 = pd.read_csv(os.path.join(ocr_output_folder, "page_1_ocr.csv"))

elif "huguenot" in pdf_file_name:
    group1 = [
        {"left": 52, "top": 250, "width": 83, "height": 18},
        {"left": 142, "top": 250, "width": 46, "height": 14},
        {"left": 195, "top": 250, "width": 29, "height": 14},
    ]
    group2 = [
        {"left": 1546, "top": 1964, "width": 84, "height": 17},
    ]
    group4 = [
        {"left": 1239, "top": 218, "width": 76, "height": 14},
    ]
    group5 = [
        {"left": 1333, "top": 663, "width": 53, "height": 14},
    ] #1333	663	53	14
    df_group1 = df_group2 = df_group4 = df_group5 = pd.read_csv(os.path.join(ocr_output_folder, "page_1_ocr.csv"))

else:
    print(f"No group definitions found for file: {pdf_file_name}")
    exit()

df_group1.columns = df_group1.columns.str.strip()
df_group2.columns = df_group2.columns.str.strip()
if 'df_group3' in locals():
    df_group3.columns = df_group3.columns.str.strip()
if 'df_group4' in locals():
    df_group4.columns = df_group4.columns.str.strip()
if 'df_group5' in locals():
    df_group5.columns = df_group5.columns.str.strip()

def extract_text_from_region(df, region):
    left = region["left"]
    top = region["top"]
    right = left + region["width"]
    bottom = top + region["height"]

    matched_words = df[
        (df["left"] < right) & (df["left"] + df["width"] > left) &
        (df["top"] < bottom) & (df["top"] + df["height"] > top)
    ]

    return " ".join(
        w for w in matched_words["text"].tolist()
        if isinstance(w, str) and w.strip()
    ).strip()

value1 = None
value2 = None

line1 = " ".join(extract_text_from_region(df_group1, r) for r in group1 if extract_text_from_region(df_group1, r))
print(line1)

for r in group2:
    line = extract_text_from_region(df_group2, r)
    if line:
        print(line)
        value1 = float(line.replace("$", "").replace(",", ""))
        print(value1)

if 'group3' in locals():
    for r in group3:
        line = extract_text_from_region(df_group3, r)
        if line:
            print(line)
            value2 = float(line.replace("$", "").replace(",", ""))

if value1 is not None and value2 is not None:
    print("the total amount is", (value1 + value2))

# Added extraction and print for group4 (date)
if 'group4' in locals():
    for r in group4:
        line = extract_text_from_region(df_group4, r)
        if line:
            print("Date:", line)

# Added extraction and print for group5 (BOL number)
if 'group5' in locals():
    for r in group5:
        line = extract_text_from_region(df_group5, r)
        if line:
            print("BOL Number:", line)


import pandas as pd
import math

# --------------------
# Step 1: Load CSV File
# --------------------
df_excel = pd.read_csv(r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/SampleData(Sheet1).csv")

invoice_dict = {}

# --------------------
# Step 2: Extracted from PDF (replace with your actual logic)
# --------------------
invoice_number = int("85718")  # Replace this with value extracted from PDF
actual_total = round(value1 + value2, 2) if value1 and value2 else round(value1 or value2, 2)

# --------------------
# Step 3: Match invoice number in CSV
# --------------------
matched_rows = df_excel[df_excel['Invoice Nbr'] == invoice_number].copy()

if not matched_rows.empty:
    deliveries = []
    for _, row in matched_rows.iterrows():
        deliveries.append({
            'Store': row['Store Name'],
            'Fuel': row['Fuel Type'],
            'Amount': row['Total']
        })

from decimal import Decimal, ROUND_HALF_UP

# Function to round to match the format in Excel (e.g., 2 decimal places)
def round_decimal(value, places=2):
    return float(Decimal(value).quantize(Decimal(f"1.{'0'*places}"), rounding=ROUND_HALF_UP))

# Get sum without rounding individual values
current_sum = sum(d['Amount'] for d in deliveries)

# Round only the final total and actual invoice
rounded_current_sum = round_decimal(current_sum)
rounded_actual_total = round_decimal(actual_total)

diff = round_decimal(rounded_actual_total - rounded_current_sum)

if abs(diff) >= 0.01:
    print(f"⚠️ Adjusting Invoice {invoice_number} by difference of {diff}")
    deliveries[-1]['Amount'] = round_decimal(deliveries[-1]['Amount'] + diff)

invoice_dict[invoice_number] = deliveries

# ✅ Print result
print(f"\n📦 Final delivery breakdown for Invoice {invoice_number}:")
for d in deliveries:
    print(d)






