import os
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image
from io import BytesIO

import logic
import schemas

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Invoice Extractor API",
    description="An API to extract structured data from invoice images using Google Gemini, with vendor-specific memory.",
    version="2.0.0"
)



@app.post("/extract/invoice", response_model=schemas.InvoiceData)
async def extract_from_invoice(
        use_memory: bool = Query(False, description="Use vendor memory to improve extraction accuracy."),
        file: UploadFile = File(..., description="The invoice file (PDF, JPG, PNG).")
):
    """
    Receives an invoice file, processes it, and returns the extracted JSON data.
    """
    if not file.content_type in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPG, PNG, or PDF.")

    contents = await file.read()
    image_bytes: Optional[bytes] = None

    try:
        if file.content_type == "application/pdf":
            image_bytes = logic.pdf_page_to_jpeg_bytes(contents)
        else:
            img = Image.open(BytesIO(contents))
            image_bytes = logic.image_to_jpeg_bytes(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image file: {e}")

    if not image_bytes:
        raise HTTPException(status_code=500, detail="Could not extract an image from the provided file.")

    try:
        result_data, error_message = logic.process_invoice_extraction(
            image_bytes=image_bytes,
            original_filename=file.filename,
            use_memory=use_memory
        )

        if error_message:
            raise HTTPException(status_code=500, detail=error_message)

        if not result_data:
            raise HTTPException(status_code=500, detail="Extraction returned no data.")

        return result_data

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/corrections", status_code=201)
async def save_user_correction(correction: schemas.CorrectionRequest):
    """
    Receives a corrected invoice JSON and saves it to the vendor memory file.
    """
    try:
        logic.save_memory(correction.dict())
        return {"message": "Correction saved to vendor memory successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save correction: {e}")