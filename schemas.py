from pydantic import BaseModel
from typing import List, Optional, Any

class LineItem(BaseModel):
    ITM_CODE: Optional[str] = None
    ITM_L_NM: Optional[str] = None
    ITM_F_NM: Optional[str] = None
    ITM_UNT: Optional[str] = None
    ITM_QTY: Optional[float] = None
    ITM_PRICE: Optional[float] = None
    TOTAL_BFR_TAX: Optional[float] = None
    ITM_DSCNT: Optional[float] = 0.0
    TOTAL_AFTR_TAX: Optional[float] = None

class InvoiceData(BaseModel):
    VNDR_NM: Optional[str] = None
    CSTMR_NM: Optional[str] = None
    DOC_NO: Optional[str] = None
    DOC_NO_TAX: Optional[str] = None
    ITEMS: List[LineItem] = []

class CorrectionRequest(BaseModel):
    VNDR_NM: Optional[str] = None
    ITEMS: List[LineItem] = []
    CSTMR_NM: Optional[str] = None
    DOC_NO: Optional[str] = None
    DOC_NO_TAX: Optional[str] = None