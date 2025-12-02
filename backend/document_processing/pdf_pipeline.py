from __future__ import annotations

import concurrent.futures
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

try:  # Optional heavy dependencies
    import docling

    HAS_DOCLING = True
except Exception:  # pragma: no cover - optional dependency
    HAS_DOCLING = False

try:
    from wordfreq import zipf_frequency

    HAS_WORD_FREQ = True
except Exception:  # pragma: no cover
    HAS_WORD_FREQ = False

try:
    from langdetect import detect_langs

    HAS_LANGDETECT = True
except Exception:  # pragma: no cover
    HAS_LANGDETECT = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_TRANSFORMERS = False

RE_BAD_CHAR = re.compile(r"[^\x00-\x7F\u00A0-\u017F]")
RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]{2,}")


@dataclass
class PageResult:
    pageno: int
    final_text: str
    extraction_method: str
    scores: Dict[str, float]
    text_ops: int
    image_ops: int


def analyze_pdf_ops(filepath: str) -> List[Tuple[int, int, int]]:
    """Count text/image operators per page to guide extraction strategy."""
    reader = PdfReader(filepath)
    summary: List[Tuple[int, int, int]] = []
    for idx, page in enumerate(reader.pages):
        contents = page.get_contents()
        stream_data = b""
        if contents:
            if isinstance(contents, list):
                for c in contents:
                    try:
                        stream_data += c.get_data()
                    except Exception:
                        pass
            else:
                try:
                    stream_data = contents.get_data()
                except Exception:
                    stream_data = b""
        text_ops = (
            stream_data.count(b"Tj")
            + stream_data.count(b"TJ")
            + stream_data.count(b"Tf")
            + stream_data.count(b"Td")
            + stream_data.count(b"TD")
            + stream_data.count(b"Tm")
            + stream_data.count(b"T*")
        )
        image_ops = stream_data.count(b"Do")
        summary.append((idx + 1, text_ops, image_ops))
    return summary


def extract_text_pypdf(filepath: str, page_index: int) -> str:
    reader = PdfReader(filepath)
    text = reader.pages[page_index].extract_text() or ""
    return text.strip()


def ocr_tesseract_image(image) -> str:
    """Run pytesseract on a PIL Image instance."""
    return pytesseract.image_to_string(image)


def ocr_docling_image(image) -> str:
    if not HAS_DOCLING:  # pragma: no cover - optional dependency
        raise RuntimeError("Docling is not installed in this environment.")
    return docling.ocr_image_to_text(image)


def garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    total = len(text)
    bad = len(RE_BAD_CHAR.findall(text))
    return bad / max(1, total)


def word_dictionary_fraction(text: str) -> float:
    words = RE_WORD.findall(text)
    if not words:
        return 0.0

    if HAS_WORD_FREQ:
        valid = sum(1 for w in words if zipf_frequency(w.lower(), "en") > 1.5)
        return valid / len(words)

    def looks_real(token: str) -> bool:
        return len(token) > 2 and re.search(r"[aeiouyAEIOUYàèéùôî]", token) is not None

    valid = sum(1 for w in words if looks_real(w))
    return valid / len(words)


def detect_primary_language_confidence(text: str) -> float:
    if not text or not HAS_LANGDETECT:
        return 0.0
    try:
        langs = detect_langs(text)
        return float(langs[0].prob) if langs else 0.0
    except Exception:
        return 0.0


def perplexity_score(text: str, model_name: str = "gpt2") -> float:
    if not HAS_TRANSFORMERS or not text:
        return 0.5
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():  # pragma: no cover - requires GPU
            model.to("cuda")
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss.item()
            ppl = math.exp(loss)
            return 1.0 / (1.0 + math.log(1.0 + ppl))
    except Exception:
        return 0.5


def evaluate_text_quality(text: str) -> Dict[str, float]:
    if not text:
        return {
            "garbage_ratio": 1.0,
            "dictionary_fraction": 0.0,
            "lang_conf": 0.0,
            "perplexity_score": 0.5,
            "score": 0.0,
        }

    g = garbage_ratio(text)
    d = word_dictionary_fraction(text)
    l = detect_primary_language_confidence(text)
    p = perplexity_score(text)
    garbage_ok = max(0.0, 1.0 - g)
    score = (0.35 * d) + (0.25 * garbage_ok) + (0.25 * p) + (0.15 * l)
    score = max(0.0, min(1.0, score))
    return {
        "garbage_ratio": g,
        "dictionary_fraction": d,
        "lang_conf": l,
        "perplexity_score": p,
        "score": score,
    }


def evaluate_text_quality_fast(text: str) -> Dict[str, float]:
    if not text:
        return {"score": 0.0}
    total_chars = len(text)
    non_ascii = len([c for c in text if ord(c) > 127])
    garbage_ratio_val = non_ascii / max(1, total_chars)
    words = len(text.split())
    garbage_score = max(0.0, 1.0 - garbage_ratio_val)
    word_score = min(1.0, words / 100)
    score = (0.6 * word_score) + (0.4 * garbage_score)
    return {"score": score, "garbage_ratio": garbage_ratio_val, "word_count": words}


def create_fallback_result(page_index: int) -> PageResult:
    return PageResult(
        pageno=page_index + 1,
        final_text="",
        extraction_method="error",
        scores={"score": 0.0},
        text_ops=0,
        image_ops=0,
    )


def process_page(
    filepath: str,
    page_index: int,
    dpi: int = 300,
    min_text_ops_for_born_digital: int = 10,
    tesseract_threshold: float = 0.45,
    docling_threshold: float = 0.30,
) -> PageResult:
    reader = PdfReader(filepath)
    page = reader.pages[page_index]
    contents = page.get_contents()
    stream_data = b""
    if contents:
        if isinstance(contents, list):
            for c in contents:
                try:
                    stream_data += c.get_data()
                except Exception:
                    pass
        else:
            try:
                stream_data = contents.get_data()
            except Exception:
                stream_data = b""
    text_ops = (
        stream_data.count(b"Tj")
        + stream_data.count(b"TJ")
        + stream_data.count(b"Tf")
        + stream_data.count(b"Td")
        + stream_data.count(b"TD")
        + stream_data.count(b"Tm")
        + stream_data.count(b"T*")
    )
    image_ops = stream_data.count(b"Do")

    if text_ops >= min_text_ops_for_born_digital and text_ops > image_ops:
        extracted = page.extract_text() or ""
        scores = evaluate_text_quality(extracted)
        return PageResult(
            pageno=page_index + 1,
            final_text=extracted,
            extraction_method="pypdf-born-digital",
            scores=scores,
            text_ops=text_ops,
            image_ops=image_ops,
        )

    images = convert_from_path(
        filepath,
        first_page=page_index + 1,
        last_page=page_index + 1,
        dpi=dpi,
    )
    if not images:
        return PageResult(
            pageno=page_index + 1,
            final_text="",
            extraction_method="empty-page",
            scores={"score": 0.0},
            text_ops=text_ops,
            image_ops=image_ops,
        )
    img = images[0]
    pypdf_text = page.extract_text() or ""
    tesseract_text = ocr_tesseract_image(img)
    t_scores = evaluate_text_quality(tesseract_text)

    if t_scores["score"] >= tesseract_threshold:
        combined = (pypdf_text + "\n" + tesseract_text).strip()
        return PageResult(
            pageno=page_index + 1,
            final_text=combined,
            extraction_method="tesseract",
            scores=t_scores,
            text_ops=text_ops,
            image_ops=image_ops,
        )

    slow_text = None
    if HAS_DOCLING:
        try:
            slow_text = ocr_docling_image(img)
        except Exception:
            slow_text = None

    if slow_text is None and HAS_TRANSFORMERS:  # pragma: no cover - heavy path
        try:
            slow_text = ocr_tesseract_image(img)
        except Exception:
            slow_text = None

    if slow_text:
        s_scores = evaluate_text_quality(slow_text)
        if s_scores["score"] >= t_scores["score"] or s_scores["score"] >= docling_threshold:
            combined = (pypdf_text + "\n" + slow_text).strip()
            return PageResult(
                pageno=page_index + 1,
                final_text=combined,
                extraction_method="docling" if HAS_DOCLING else "transformer_slow",
                scores=s_scores,
                text_ops=text_ops,
                image_ops=image_ops,
            )

    return PageResult(
        pageno=page_index + 1,
        final_text=(pypdf_text + "\n" + tesseract_text).strip(),
        extraction_method="tesseract_fallback",
        scores=t_scores,
        text_ops=text_ops,
        image_ops=image_ops,
    )


def analyze_pdf_needs_ocr(filepath: str, n_pages: int) -> List[bool]:
    reader = PdfReader(filepath)
    needs = []
    for i in range(min(n_pages, 1000)):
        try:
            page = reader.pages[i]
            text = page.extract_text() or ""
            needs.append(len(text.strip()) < 100)
        except Exception:
            needs.append(True)
    return needs


def analyze_pdf_ops_fast(filepath: str) -> List[Tuple[int, int, int]]:
    reader = PdfReader(filepath)
    ops_summary: List[Tuple[int, int, int]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            text_ops = 1 if len(text.strip()) > 50 else 0
            contents = page.get_contents()
            image_ops = 0
            if contents:
                stream_data = b""
                if isinstance(contents, list):
                    for c in contents[:2]:
                        try:
                            stream_data += c.get_data()[:1000]
                        except Exception:
                            pass
                else:
                    try:
                        stream_data = contents.get_data()[:1000]
                    except Exception:
                        pass
                image_ops = stream_data.count(b"Do")
            ops_summary.append((i + 1, text_ops, image_ops))
        except Exception:
            ops_summary.append((i + 1, 0, 1))
    return ops_summary


def process_page_fast(
    filepath: str,
    page_index: int,
    operator_info: Tuple[int, int, int],
    pre_converted_image=None,
) -> PageResult:
    reader = PdfReader(filepath)
    page = reader.pages[page_index]
    _, text_ops, image_ops = operator_info
    pypdf_text = page.extract_text() or ""
    if len(pypdf_text.strip()) > 100 or text_ops > 5:
        return PageResult(
            pageno=page_index + 1,
            final_text=pypdf_text,
            extraction_method="pypdf-born-digital",
            scores={"score": 0.8},
            text_ops=text_ops,
            image_ops=image_ops,
        )
    if pre_converted_image is not None:
        tesseract_text = ocr_tesseract_image(pre_converted_image)
        combined = (pypdf_text + "\n" + tesseract_text).strip()
        return PageResult(
            pageno=page_index + 1,
            final_text=combined,
            extraction_method="tesseract_fast",
            scores={"score": 0.5},
            text_ops=text_ops,
            image_ops=image_ops,
        )
    return PageResult(
        pageno=page_index + 1,
        final_text=pypdf_text,
        extraction_method="pypdf_fallback",
        scores={"score": 0.3},
        text_ops=text_ops,
        image_ops=image_ops,
    )


def process_page_optimized(
    filepath: str,
    page_index: int,
    dpi: int = 150,
    pre_converted_image=None,
    fast_mode: bool = True,
) -> PageResult:
    reader = PdfReader(filepath)
    page = reader.pages[page_index]
    pypdf_text = page.extract_text() or ""
    if len(pypdf_text.strip()) > 200:
        scores = evaluate_text_quality_fast(pypdf_text) if fast_mode else evaluate_text_quality(pypdf_text)
        return PageResult(
            pageno=page_index + 1,
            final_text=pypdf_text,
            extraction_method="pypdf-born-digital",
            scores=scores,
            text_ops=1,
            image_ops=0,
        )
    if pre_converted_image is not None:
        img = pre_converted_image
    else:
        images = convert_from_path(
            filepath,
            first_page=page_index + 1,
            last_page=page_index + 1,
            dpi=dpi,
            grayscale=True,
        )
        img = images[0] if images else None
    if img is None:
        return create_fallback_result(page_index)
    tesseract_text = ocr_tesseract_image(img)
    combined = (pypdf_text + "\n" + tesseract_text).strip()
    scores = evaluate_text_quality_fast(combined) if fast_mode else evaluate_text_quality(combined)
    return PageResult(
        pageno=page_index + 1,
        final_text=combined,
        extraction_method="tesseract_fast" if fast_mode else "tesseract",
        scores=scores,
        text_ops=0,
        image_ops=1,
    )


def process_large_pdf_fast(
    filepath: str,
    dpi: int = 150,
    max_pages: Optional[int] = None,
    max_workers: int = 4,
) -> Dict:
    reader = PdfReader(filepath)
    n_pages = len(reader.pages)
    pages_to_process = range(n_pages) if max_pages is None else range(min(n_pages, max_pages))
    operator_summary = analyze_pdf_ops_fast(filepath)
    all_images = convert_from_path(filepath, dpi=dpi, grayscale=True, thread_count=4)
    results: List[PageResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_page_fast,
                filepath,
                idx,
                operator_summary[idx] if idx < len(operator_summary) else (idx + 1, 0, 0),
                all_images[idx] if idx < len(all_images) else None,
            ): idx
            for idx in pages_to_process
        }
        for future in concurrent.futures.as_completed(futures):
            page_index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Page {page_index + 1} failed: {exc}")
                result = create_fallback_result(page_index)
            results.append(result)
    results.sort(key=lambda x: x.pageno)
    summary = {
        "total_pages": n_pages,
        "processed_pages": len(results),
        "born_digital_pages": sum(1 for p in results if "pypdf" in p.extraction_method),
        "tesseract_pages": sum(1 for p in results if "tesseract" in p.extraction_method),
        "mode": "fast_large_pdf",
    }
    return {"pages": [r.__dict__ for r in results], "summary": summary}


def process_pdf(
    filepath: str,
    dpi: int = 150,
    max_pages: Optional[int] = None,
    max_workers: int = 4,
    fast_mode: bool = True,
) -> Dict:
    reader = PdfReader(filepath)
    n_pages = len(reader.pages)
    if n_pages > 50 and fast_mode:
        return process_large_pdf_fast(filepath, dpi=dpi, max_pages=max_pages, max_workers=max_workers)
    pages_to_process = range(n_pages) if max_pages is None else range(min(n_pages, max_pages))
    needs_ocr_flags = analyze_pdf_needs_ocr(filepath, n_pages)
    ocr_indices = [i for i, needs_ocr in enumerate(needs_ocr_flags) if needs_ocr]
    all_images = []
    if ocr_indices:
        all_images = convert_from_path(
            filepath,
            dpi=dpi,
            grayscale=True,
            thread_count=4,
            first_page=1,
            last_page=n_pages,
        )
    image_map = {idx: all_images[idx] for idx in ocr_indices if idx < len(all_images)}
    results: List[PageResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_page_optimized,
                filepath,
                idx,
                dpi,
                image_map.get(idx),
                fast_mode,
            ): idx
            for idx in pages_to_process
        }
        for future in concurrent.futures.as_completed(futures):
            page_index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Page {page_index + 1} failed: {exc}")
                result = create_fallback_result(page_index)
            results.append(result)
    results.sort(key=lambda x: x.pageno)
    summary = {
        "total_pages": n_pages,
        "processed_pages": len(results),
        "born_digital_pages": sum(1 for p in results if p.extraction_method.startswith("pypdf")),
        "tesseract_pages": sum(1 for p in results if p.extraction_method.startswith("tesseract")),
        "docling_pages": sum(1 for p in results if p.extraction_method.startswith("docling")),
        "error_pages": sum(1 for p in results if p.extraction_method == "error"),
    }
    return {"pages": [r.__dict__ for r in results], "summary": summary}


__all__ = [
    "PageResult",
    "process_pdf",
    "process_page",
    "process_page_fast",
    "process_page_optimized",
    "process_large_pdf_fast",
    "analyze_pdf_ops",
    "analyze_pdf_ops_fast",
    "analyze_pdf_needs_ocr",
    "extract_text_pypdf",
    "evaluate_text_quality",
    "evaluate_text_quality_fast",
]
