import json
import time
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO
from PIL import Image

import google.generativeai as genai
from google.api_core import exceptions
from google.cloud import storage
from google.oauth2 import service_account

try:
    from pdf2image import convert_from_bytes

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789..")
CAPACITY_TOKENS = re.compile(r"(?i)\b(?:ml|ltrs?|ltr|gms?|gm|grams?|جم|غ|مل|لتر|كجم|kg|كيلو)\b")
MEMORY_PATH = "vendor_corrections.jsonl"
USER_PROMPT = r"""
**CRITICAL TASK: Parse the invoice image and return ONLY a valid JSON (UTF-8) matching EXACTLY this schema and keys.**

Top-level keys:
- "VNDR_NM": string|null
- "CSTMR_NM": string|null
- "DOC_NO": string|null
- "DOC_NO_TAX": string|null
- "ITEMS": [ { line objects as below } ]

For each line in ITEMS:
- "ITM_CODE": string|null
- "ITM_L_NM": string|null
- "ITM_F_NM": string|null
- "ITM_UNT": string|null
- "ITM_QTY": number|null
- "ITM_PRICE": number|null
- "TOTAL_BFR_TAX": number|null
- "ITM_DSCNT": number
- "TOTAL_AFTR_TAX": number|null

STRICT RULES:
1) Read Arabic invoices from RIGHT to LEFT. The Item Code column is often the rightmost column.
2) Do NOT use size/capacity tokens from description as codes (e.g., 230ML).
3) All numeric values must be JSON numbers (not strings).
4) If a field is not present, return null.
5) Self-check: if (ITM_PRICE * ITM_QTY) does not approx equal TOTAL_BFR_TAX, but (ITM_PRICE * ITM_CODE_as_number) does, swap QTY and CODE.

Return ONLY the JSON object, nothing else.
"""



def get_gcs_credentials() -> Optional[dict]:
    gcs_creds_str = os.getenv("GCS_CREDENTIALS_JSON")
    if not gcs_creds_str:
        print("Warning: GCS_CREDENTIALS_JSON environment variable not set. Cloud upload will be disabled.")
        return None
    try:
        return json.loads(gcs_creds_str)
    except json.JSONDecodeError:
        print("Error: GCS_CREDENTIALS_JSON is not valid JSON.")
        return None



def process_invoice_extraction(
        image_bytes: bytes, original_filename: str, use_memory: bool = False
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Orchestrates the entire invoice extraction process.
    Returns (result_data, error_message)
    """
    raw1, err1 = call_gemini_api(image_bytes, USER_PROMPT, "gemini-1.5-flash-latest")
    if err1:
        return None, f"Initial extraction failed: {err1}"

    fixed_data = validate_and_fix_schema(raw1)
    vendor_name = fixed_data.get("VNDR_NM")

    if use_memory and vendor_name:
        hint = build_vendor_hint(vendor_name)
        if hint:
            prompt2 = USER_PROMPT + "\n" + hint
            raw2, err2 = call_gemini_api(image_bytes, prompt2, "gemini-1.5-flash-latest")
            if raw2:
                fixed_data = validate_and_fix_schema(raw2)  # Overwrite with improved data
            else:
                print(f"Warning: Re-extraction for vendor '{vendor_name}' failed: {err2}")

    # 3. Upload artifacts to Google Cloud Storage
    upload_to_gcs(
        image_bytes=image_bytes,
        json_data=fixed_data,
        base_filename=original_filename
    )

    return fixed_data, None




_model_cache = {}


def get_model(model_name: str = "gemini-1.5-flash-latest", enforce_json: bool = True):
    if model_name in _model_cache:
        return _model_cache[model_name]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment.")

    genai.configure(api_key=api_key)
    generation_config = {"response_mime_type": "application/json"} if enforce_json else {}
    model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
    _model_cache[model_name] = model
    return model


def call_gemini_api(image_bytes: bytes, prompt: str, model_name: str) -> Tuple[Optional[dict], Optional[str]]:
    if not image_bytes:
        return None, "No image bytes provided"

    try:
        model = get_model(model_name)
    except Exception as e:
        return None, f"Model initialization failed: {e}"

    last_err = "Unknown failure"
    delay = 2
    for attempt in range(3):  # Reduced retries for faster API response
        try:
            img_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
            resp = model.generate_content([prompt, img_pil])
            text = getattr(resp, "text", "")

            # Simple JSON cleanup
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise ValueError("No valid JSON object found in the response.")

            json_str = match.group(0)
            data = json.loads(json_str)
            return data, None

        except (exceptions.ResourceExhausted, exceptions.DeadlineExceeded) as e:
            last_err = f"API rate limit or timeout: {e}"
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            last_err = f"Error processing Gemini response: {e}"
            break  # Don't retry on other errors like bad JSON

    return None, last_err



def image_to_jpeg_bytes(img: Image.Image, max_side: int = 2400, quality: int = 92) -> bytes:
    if img.mode != 'RGB':
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", optimize=True, quality=quality)
    return buf.getvalue()


def pdf_page_to_jpeg_bytes(pdf_bytes: bytes, dpi: int = 200) -> Optional[bytes]:
    if not PDF_SUPPORT:
        raise RuntimeError("PDF processing is not supported. Please install 'pdf2image' and its dependencies.")

    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
    if pages:
        return image_to_jpeg_bytes(pages[0])
    return None


def normalize_number(val: Any) -> Optional[float]:
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip().translate(ARABIC_INDIC).replace("\u00A0", " ")
    s = re.sub(r"[\s\$£€¥ر.سج.د]*", "", s).replace(",", "")
    s = re.sub(r"\.(?=.*\.)", "", s)
    try:
        return float(s)
    except:
        return None


def coerce_nulls(x: Any) -> Any:
    if isinstance(x, dict): return {k: coerce_nulls(v) for k, v in x.items()}
    if isinstance(x, list): return [coerce_nulls(v) for v in x]
    return None if isinstance(x, str) and x.strip() == "" else x


def normalize_item_code(code: Optional[str]) -> Optional[str]:
    if not code: return code
    s = re.sub(r"\s+", " ", str(code)).strip()
    s = re.sub(r"^-+", "", s)
    s = " ".join(p for p in s.split() if not CAPACITY_TOKENS.search(p)).strip()
    return s or None


def as_float_if_numeric_str(val: Any) -> Optional[float]:
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        try:
            return float(s)
        except:
            return None
    return None


def validate_and_fix_schema(data: dict) -> dict:
    vndr = data.get("VNDR_NM") or data.get("اسم_المورد")
    cstm = data.get("CSTMR_NM") or data.get("اسم_العميل")
    doc = data.get("DOC_NO") or data.get("رقم_الفاتورة")
    doct = data.get("DOC_NO_TAX") or data.get("رقم_الفاتورة_الضريبية")
    items = data.get("ITEMS") or data.get("الأصناف")
    if not isinstance(items, list): items = [] if items is None else [items]
    fixed_items: List[Dict[str, Any]] = []
    for item in items:
        item = item or {}
        code = item.get("ITM_CODE") or item.get("رقم_الصنف")
        name_ar = item.get("ITM_L_NM") or item.get("اسم_الصنف")
        name_en = item.get("ITM_F_NM") or item.get("اسم_الصنف_انجليزي")
        unit = item.get("ITM_UNT") or item.get("الوحدة")
        qty = item.get("ITM_QTY") or item.get("الكمية")
        price = item.get("ITM_PRICE") or item.get("سعر_الوحدة")
        total_before = item.get("TOTAL_BFR_TAX") or item.get("الاجمالي_قبل_الضريبة")
        disc = item.get("ITM_DSCNT") or item.get("الخصم")
        total_after = item.get("TOTAL_AFTR_TAX") or item.get("الاجمالي_بعد_الضريبة")
        fixed = {
            "ITM_CODE": normalize_item_code(code), "ITM_L_NM": name_ar, "ITM_F_NM": name_en,
            "ITM_UNT": unit, "ITM_QTY": normalize_number(qty), "ITM_PRICE": normalize_number(price),
            "TOTAL_BFR_TAX": normalize_number(total_before), "ITM_DSCNT": normalize_number(disc) or 0.0,
            "TOTAL_AFTR_TAX": normalize_number(total_after),
        }
        qty_v, code_str, code_num, unit_price, line_total = (
            fixed["ITM_QTY"], fixed["ITM_CODE"], as_float_if_numeric_str(fixed["ITM_CODE"]),
            fixed["ITM_PRICE"], fixed["TOTAL_BFR_TAX"],
        )
        if unit_price and line_total and unit_price != 0:
            expected_qty = line_total / unit_price
            eps = max(0.02, abs(expected_qty) * 0.01)

            def close(a, b):
                return a is not None and b is not None and abs(a - b) <= eps

            if not close(qty_v, expected_qty) and close(code_num, expected_qty):
                prev_qty, fixed["ITM_QTY"] = qty_v, code_num
                if isinstance(prev_qty, (int, float)) and 0 < prev_qty <= 1_000_000:
                    fixed["ITM_CODE"] = str(int(round(prev_qty)))
                else:
                    fixed["ITM_CODE"] = None
        fixed_items.append(fixed)
    out = {
        "VNDR_NM": vndr, "CSTMR_NM": cstm, "DOC_NO": doc,
        "DOC_NO_TAX": doct, "ITEMS": fixed_items,
    }
    return coerce_nulls(out)



def load_memory() -> List[dict]:
    if not os.path.exists(MEMORY_PATH): return []
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except:
        return []


def save_memory(record: dict) -> None:
    with open(MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_vendor_examples(vendor: str, max_n: int = 3) -> List[dict]:
    mem = load_memory()
    examples = [r for r in mem if (r.get("VNDR_NM") == vendor)]
    return examples[:max_n]


def build_vendor_hint(vendor: str) -> str:
    ex = get_vendor_examples(vendor)
    if not ex: return ""
    lines = ["\n\n📌 Vendor-specific guidance (learned from previous corrections):", f"- Vendor: {vendor}"]
    for i, r in enumerate(ex, start=1):
        items = r.get("ITEMS") or []
        if not items: continue
        s = items[0]
        code, qty = s.get("ITM_CODE"), s.get("ITM_QTY")
        lines.append(f"  • Example {i}: item_code='{code}', qty={qty} → use this as a pattern.")
    lines.append(
        "- Never use size tokens (e.g., 230ML) as codes. If unit_price*qty mismatches but unit_price*code matches total, swap them.")
    return "\n".join(lines)



_gcs_bucket = None


def _get_gcs_bucket():
    global _gcs_bucket
    if _gcs_bucket:
        return _gcs_bucket

    creds_info = get_gcs_credentials()
    bucket_name = os.getenv("GCS_BUCKET_NAME")

    if not creds_info or not bucket_name:
        return None

    try:
        creds = service_account.Credentials.from_service_account_info(creds_info)
        client = storage.Client(credentials=creds)
        _gcs_bucket = client.get_bucket(bucket_name)
        print(f"Successfully connected to GCS bucket: {bucket_name}")
        return _gcs_bucket
    except Exception as e:
        print(f"Error connecting to GCS: {e}")
        return None


def upload_to_gcs(image_bytes: bytes, json_data: dict, base_filename: str):
    bucket = _get_gcs_bucket()
    if not bucket:
        return

    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', base_filename)
        base_name = f"{timestamp}_{safe_filename}"

        image_blob = bucket.blob(f"invoices/{base_name}.jpg")
        image_blob.upload_from_string(image_bytes, content_type="image/jpeg")

        json_blob = bucket.blob(f"results/{base_name}.json")
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        json_blob.upload_from_string(json_str, content_type="application/json")
        print(f"Successfully uploaded {base_name} to GCS.")
    except Exception as e:
        print(f"Warning: Could not upload to GCS. Error: {e}")