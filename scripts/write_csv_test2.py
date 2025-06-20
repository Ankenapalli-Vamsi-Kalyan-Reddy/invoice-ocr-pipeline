import os
import time
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from pytesseract import Output
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from decimal import Decimal, ROUND_HALF_UP

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\poppler-24.08.0\Library\bin"

INPUT_FOLDER = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files"
OUTPUT_FOLDER = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/output_images"
CSV_PATH = os.path.join(INPUT_FOLDER, "SampleData(Sheet1).csv")

# === OCR Utilities ===
def convert_pdf_to_images(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    return convert_from_path(pdf_path, poppler_path=poppler_path)

def save_ocr_data(images, output_folder):
    for i, page_image in enumerate(images):
        image_filename = os.path.join(output_folder, f"page_{i+1}.png")
        page_image.save(image_filename)
        ocr_data = pytesseract.image_to_data(page_image, output_type=Output.DATAFRAME)
        ocr_data = ocr_data[ocr_data.text.notna() & (ocr_data.text.str.strip() != "")]
        csv_filename = os.path.join(output_folder, f"page_{i+1}_ocr.csv")
        ocr_data.to_csv(csv_filename, index=False)

def load_ocr_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"OCR CSV not found: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"OCR CSV has no data: {file_path}")
    return df[df["text"].notna() & (df["text"].str.strip() != "")].copy()

def extract_text_from_region(df, region):
    left = region["left"]
    top = region["top"]
    right = left + region["width"]
    bottom = top + region["height"]
    matched_words = df[(df["left"] < right) & (df["left"] + df["width"] > left) &
                       (df["top"] < bottom) & (df["top"] + df["height"] > top)]
    return " ".join(w for w in matched_words["text"].tolist() if isinstance(w, str) and w.strip()).strip()

def round_decimal(value, places=2):
    return float(Decimal(value).quantize(Decimal(f"1.{ '0'*places }"), rounding=ROUND_HALF_UP))

# === Business Logic ===
def extract_invoice_number(df):
    invoice_candidates = df[df['text'].str.contains("\\d{5,}", na=False, regex=True)]
    for text in invoice_candidates['text']:
        if text.isdigit():
            return int(text)
    raise ValueError("Invoice number not found.")

def get_group_definitions(pdf_file_name, output_folder):
    file1 = os.path.join(output_folder, "page_1_ocr.csv")
    file2 = os.path.join(output_folder, "page_2_ocr.csv")
    groups, dfs = {}, {}

    if "sunoco" in pdf_file_name:
        groups["group1"] = [{"left": 911, "top": 42, "width": 178, "height": 36}]
        groups["group2"] = [{"left": 1999, "top": 327, "width": 110, "height": 21}]
        groups["group4"] = [{"left": 1792, "top": 283, "width": 118, "height": 21}]
        groups["group5"] = [{"left": 291, "top": 741, "width": 81, "height": 17}]
        dfs = {
            "df_group1": load_ocr_csv(file1),
            "df_group2": load_ocr_csv(file2),
            "df_group4": load_ocr_csv(file1),
            "df_group5": load_ocr_csv(file1),
        }
    elif "metroplex" in pdf_file_name:
        group = [{"left": 153, "top": 539, "width": 161, "height": 22}]
        df = load_ocr_csv(file1)
        groups = {"group1": group, "group2": group, "group4": group, "group5": group}
        dfs = {"df_group1": df, "df_group2": df, "df_group4": df, "df_group5": df}
    else:
        raise ValueError(f"Unsupported vendor format: {pdf_file_name}")
    return groups, dfs

def extract_invoice_info(groups, dfs):
    value1 = value2 = None
    for r in groups["group2"]:
        text = extract_text_from_region(dfs["df_group2"], r)
        if text:
            value1 = float(text.replace("$", "").replace(",", ""))
    invoice_number = extract_invoice_number(dfs["df_group1"])
    return invoice_number, round_decimal(value1 or 0.0)

def process_csv_and_adjust(csv_path, invoice_number, actual_total):
    df = pd.read_csv(csv_path)
    matched = df[df['Invoice Nbr'] == invoice_number].copy()
    if matched.empty:
        raise ValueError(f"Invoice {invoice_number} not found in CSV")
    deliveries = [{
        'Store': row['Store Name'],
        'Fuel': row['Fuel Type'],
        'Amount': row['Total']
    } for _, row in matched.iterrows()]
    current_sum = sum(d['Amount'] for d in deliveries)
    rounded_diff = round_decimal(actual_total - current_sum)
    if abs(rounded_diff) >= 0.01:
        deliveries[-1]['Amount'] = round_decimal(deliveries[-1]['Amount'] + rounded_diff)
    return deliveries

def process_pdf_file(pdf_path):
    print(f"\n📄 Processing file: {pdf_path}")
    pdf_file_name = os.path.splitext(os.path.basename(pdf_path))[0].lower()
    pages = convert_pdf_to_images(pdf_path, OUTPUT_FOLDER)
    save_ocr_data(pages, OUTPUT_FOLDER)
    groups, dfs = get_group_definitions(pdf_file_name, OUTPUT_FOLDER)
    invoice_number, actual_total = extract_invoice_info(groups, dfs)
    deliveries = process_csv_and_adjust(CSV_PATH, invoice_number, actual_total)
    print(f"\n📦 Final delivery breakdown for Invoice {invoice_number}:")
    for d in deliveries:
        print(d)

# === Watchdog to Monitor Folder ===
class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.lower().endswith(".pdf"):
            try:
                process_pdf_file(event.src_path)
            except Exception as e:
                print(f"❌ Error processing {event.src_path}: {e}")

def start_watching():
    observer = Observer()
    observer.schedule(PDFHandler(), INPUT_FOLDER, recursive=False)
    observer.start()
    print(f"👀 Watching for new PDFs in: {INPUT_FOLDER}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watching()
