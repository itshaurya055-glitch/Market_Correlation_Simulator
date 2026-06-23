"""
EPC Intelligence Core — Document Ingestion Pipeline

Processes PDF documents into text chunks suitable for vector embedding:
  1. PyMuPDF (fitz) for text extraction
  2. pdfplumber for table extraction
  3. pytesseract OCR fallback for scanned pages
  4. LangChain RecursiveCharacterTextSplitter for chunking
  5. Metadata tagging (source doc, page number, doc_type)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings

logger = logging.getLogger("epc_intelligence.ingestion")


def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page using PyMuPDF (fitz).
    Returns list of {page_number, text} dicts.
    """
    pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({"page_number": page_num + 1, "text": text.strip()})
    doc.close()
    return pages


def extract_tables_pdfplumber(pdf_path: str) -> list[dict]:
    """
    Extract tables from each page using pdfplumber.
    Returns list of {page_number, text} dicts where text is the
    flattened table content.
    """
    table_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                table_text_parts = []
                for table in tables:
                    for row in table:
                        # Filter None values and join cells
                        cells = [str(cell).strip() for cell in row if cell]
                        if cells:
                            table_text_parts.append(" | ".join(cells))
                if table_text_parts:
                    table_pages.append({
                        "page_number": page_num + 1,
                        "text": "\n".join(table_text_parts),
                    })
    return table_pages


def extract_text_ocr(pdf_path: str) -> list[dict]:
    """
    OCR fallback for scanned pages using pytesseract.
    Only processes pages where PyMuPDF found no text.
    """
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        logger.warning("pytesseract or Pillow not available. Skipping OCR.")
        return []

    ocr_pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Check if page has extractable text
        if page.get_text("text").strip():
            continue  # Skip — already extracted by PyMuPDF

        # Render page to image for OCR
        try:
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image)
            if text.strip():
                ocr_pages.append({
                    "page_number": page_num + 1,
                    "text": f"[OCR] {text.strip()}",
                })
        except Exception as e:
            logger.warning(f"OCR failed on page {page_num + 1}: {e}")

    doc.close()
    return ocr_pages


def extract_all_text(pdf_path: str) -> list[dict]:
    """
    Combined extraction: PyMuPDF text + pdfplumber tables + OCR fallback.
    Merges content per page.
    """
    # Primary: PyMuPDF text extraction
    text_pages = extract_text_pymupdf(pdf_path)
    logger.info(f"PyMuPDF extracted text from {len(text_pages)} pages.")

    # Secondary: Table extraction
    table_pages = extract_tables_pdfplumber(pdf_path)
    logger.info(f"pdfplumber extracted tables from {len(table_pages)} pages.")

    # Tertiary: OCR for scanned pages
    ocr_pages = extract_text_ocr(pdf_path)
    if ocr_pages:
        logger.info(f"OCR extracted text from {len(ocr_pages)} pages.")

    # Merge all content by page number
    page_content: dict[int, list[str]] = {}
    for source in [text_pages, table_pages, ocr_pages]:
        for item in source:
            pn = item["page_number"]
            if pn not in page_content:
                page_content[pn] = []
            page_content[pn].append(item["text"])

    # Combine into final list
    merged = []
    for page_num in sorted(page_content.keys()):
        merged.append({
            "page_number": page_num,
            "text": "\n\n".join(page_content[page_num]),
        })

    return merged


def chunk_document(
    pages: list[dict],
    filename: str,
    doc_type: str,
    project_id: Optional[int] = None,
    original_filename: Optional[str] = None,
) -> list[dict]:
    """
    Split extracted pages into overlapping chunks with metadata.

    Returns list of dicts with keys:
      - text: the chunk content
      - metadata: {source, page_number, doc_type, project_id, chunk_index}
    """
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    # Use original_filename for metadata so ChromaDB 'source' matches the
    # filename stored in the SQL Document record (which has no proj prefix).
    source_name = original_filename if original_filename else filename

    chunks = []
    chunk_index = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "page_number": page["page_number"],
                    "doc_type": doc_type,
                    "project_id": str(project_id) if project_id else "global",
                    "chunk_index": chunk_index,
                },
            })
            chunk_index += 1

    logger.info(
        f"Document '{filename}' split into {len(chunks)} chunks "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})."
    )
    return chunks


def ingest_pdf(
    file_path: str,
    doc_type: str,
    project_id: Optional[int] = None,
    original_filename: Optional[str] = None,
) -> list[dict]:
    """
    Main entry point: Extract text from PDF and split into chunks.

    Args:
        file_path: Path to the PDF file (may have a storage prefix like proj2_)
        doc_type: One of 'submittal', 'spec', 'standard', 'rfi_log', 'schedule'
        project_id: Optional project ID for metadata tagging
        original_filename: Original upload filename to use as the ChromaDB
            'source' metadata key. If omitted, path.name is used. Pass this
            when the saved path has been prefixed (e.g. 'proj2_file.pdf') so
            that the 'source' in ChromaDB matches the filename stored in the
            SQL Document record.

    Returns:
        List of chunk dicts ready for vector store insertion
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    filename = path.name
    source_label = original_filename or filename
    logger.info(f"Ingesting '{filename}' as '{source_label}' (type={doc_type}, project={project_id})...")

    # Extract text from all sources
    pages = extract_all_text(str(path))
    if not pages:
        logger.warning(f"No text extracted from '{filename}'.")
        return []

    # Chunk the document — use original_filename so ChromaDB 'source' matches DB
    chunks = chunk_document(pages, filename, doc_type, project_id, original_filename=source_label)
    return chunks
