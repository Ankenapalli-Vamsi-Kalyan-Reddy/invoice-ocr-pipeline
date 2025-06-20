import os
import math
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from pytesseract import Output
from decimal import Decimal, ROUND_HALF_UP

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\poppler-24.08.0\Library\bin"

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
    df = pd.read_csv(file_path)
    return df[df["text"].notna() & (df["text"].str.strip() != "")].copy()

def extract_text_from_region(df, region):
    left = region["left"]
    top = region["top"]
    right = left + region["width"]
    bottom = top + region["height"]
    matched_words = df[(df["left"] < right) & (df["left"] + df["width"] > left) & (df["top"] < bottom) & (df["top"] + df["height"] > top)]
    return " ".join(w for w in matched_words["text"].tolist() if isinstance(w, str) and w.strip()).strip()

def round_decimal(value, places=2):
    return float(Decimal(value).quantize(Decimal(f"1.{'0'*places}"), rounding=ROUND_HALF_UP))

def get_group_definitions(pdf_file_name, output_folder):
    groups, dfs = {}, {}
    file1 = os.path.join(output_folder, "page_1_ocr.csv")
    file2 = os.path.join(output_folder, "page_2_ocr.csv")

    if "sunoco" in pdf_file_name:
        groups["group1"] = [
            {"left": 911, "top": 42, "width": 178, "height": 36},
            {"left": 1106, "top": 42, "width": 66, "height": 30},
        ]
        groups["group2"] = [{"left": 1999, "top": 327, "width": 110, "height": 21}]
        groups["group3"] = [{"left": 2000, "top": 350, "width": 100, "height": 20}]
        groups["group4"] = [{"left": 1792, "top": 283, "width": 118, "height": 21}]
        groups["group5"] = [{"left": 291, "top": 741, "width": 81, "height": 17}]
        dfs = {
            "df_group1": load_ocr_csv(file1),
            "df_group2": load_ocr_csv(file2),
            "df_group3": load_ocr_csv(file2),
            "df_group4": load_ocr_csv(file1),
            "df_group5": load_ocr_csv(file1),
        }
    elif "metroplex" in pdf_file_name:
        groups["group1"] = [{"left": 153, "top": 539, "width": 161, "height": 22}]
        groups["group2"] = [{"left": 860, "top": 2082, "width": 102, "height": 22}]
        groups["group4"] = [{"left": 292, "top": 389, "width": 106, "height": 23}]
        groups["group5"] = [{"left": 151, "top": 1373, "width": 80, "height": 22}]
        df = load_ocr_csv(file1)
        dfs = {
            "df_group1": df,
            "df_group2": df,
            "df_group4": df,
            "df_group5": df,
        }
    else:
        raise ValueError(f"Unsupported vendor PDF format: {pdf_file_name}")
    return groups, dfs

def extract_invoice_info(groups, dfs):
    line1 = " ".join(extract_text_from_region(dfs["df_group1"], r) for r in groups["group1"] if extract_text_from_region(dfs["df_group1"], r))
    print("Store Info:", line1)
    value1 = value2 = None
    for r in groups["group2"]:
        text = extract_text_from_region(dfs["df_group2"], r)
        if text:
            print("Amount 1:", text)
            value1 = float(text.replace("$", "").replace(",", ""))
    if "group3" in groups:
        for r in groups["group3"]:
            text = extract_text_from_region(dfs["df_group3"], r)
            if text:
                print("Amount 2:", text)
                value2 = float(text.replace("$", "").replace(",", ""))
    if "group4" in groups:
        for r in groups["group4"]:
            print("Date:", extract_text_from_region(dfs["df_group4"], r))
    if "group5" in groups:
        for r in groups["group5"]:
            print("BOL Number:", extract_text_from_region(dfs["df_group5"], r))
    if value1 is None and value2 is None:
        raise ValueError("No invoice amount extracted")
    return round_decimal(value1 + value2) if value1 and value2 else round_decimal(value1 or value2)

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
        print(f"\n⚠️ Adjusting Invoice {invoice_number} by difference of {rounded_diff}")
        deliveries[-1]['Amount'] = round_decimal(deliveries[-1]['Amount'] + rounded_diff)
    return deliveries

def main():
    #pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/sunoco.pdf"
    #pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/lonewolf.pdf"
    pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/metroplex.pdf"
    #pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/huguenot.pdf"
    #pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/marathon.pdf"
    #pdf_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/sunoco.pdf"
    output_folder = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/output_images"
    csv_path = r"c:/Users/ashri/OneDrive/Desktop/Csv_QB/input_files/SampleData(Sheet1).csv"
    invoice_number = 85718

    pdf_file_name = os.path.splitext(os.path.basename(pdf_path))[0].lower()
    pages = convert_pdf_to_images(pdf_path, output_folder)
    save_ocr_data(pages, output_folder)

    groups, dfs = get_group_definitions(pdf_file_name, output_folder)
    actual_total = extract_invoice_info(groups, dfs)

    deliveries = process_csv_and_adjust(csv_path, invoice_number, actual_total)

    print(f"\n📦 Final delivery breakdown for Invoice {invoice_number}:")
    for d in deliveries:
        print(d)

if __name__ == "__main__":
    main()
