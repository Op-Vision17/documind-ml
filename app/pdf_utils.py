# ml-service/app/pdf_utils.py
from pypdf import PdfReader

def extract_pdf_text(path: str) -> str:
    """
    Extract text from a PDF file. Returns concatenated text of pages.
    """
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")
    return "\n".join(text_parts)
