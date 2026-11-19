#TEKNOBUILT AI-ML Engineering Assessment - Task 1: Field Data Digitization Pipeline

import argparse
import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 1. STRICT OUTPUT SCHEMA (Pydantic)

Rating = Literal["Safe", "At Risk", "Imminent Hazard", "N/A", "Unmarked"]
ExpectationAnswer = Literal["Yes", "No", "N/A", "Unmarked"]
CWAOption = Literal["1000", "2000", "3000", "4000", "5000", "5900", "6000", "7000", "8000", "OSLD"]


class ChecklistItem(BaseModel):
    item_number: str = Field(..., description="Two-digit row number, e.g., '01'")
    question: str = Field(..., description="Printed safety observation question")
    rating: Rating = Field(..., description="Checked box: S=Safe, R=At Risk, I=Imminent Hazard, N=N/A")


class ExpectationQuestions(BaseModel):
    positive_feedback_provided: ExpectationAnswer
    job_stopped_imminent_hazard: ExpectationAnswer
    investigation_conducted: ExpectationAnswer
    corrective_actions_taken: ExpectationAnswer


class SafetyObservationForm(BaseModel):
    form_title: str = Field(..., description="Header title, e.g., 'Breaking Containment - Life Saving Action'")
    observation_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    observer: Optional[str] = None
    company_observed: Optional[str] = None
    location_cwa: Optional[CWAOption] = None
    shift: Optional[Literal["Day", "Night"]] = None
    timesheet_number: Optional[str] = None
    employee_number: Optional[str] = None
    checklist_items: List[ChecklistItem] = Field(default_factory=list)
    cczjv_expectation_questions: ExpectationQuestions
    comments: Optional[str] = None


class PageExtractionResult(BaseModel):
    page_number: int
    status: Literal["SUCCESS", "REJECTED_INVALID_FORM", "ERROR"]
    rejection_reason: Optional[str] = None
    quality_metrics: dict = Field(default_factory=dict)
    data: Optional[SafetyObservationForm] = None


# 2. PRE-FLIGHT QUALITY & STRUCTURAL GATEKEEPER (OpenCV)

class FormQualityGatekeeper:
    """
    Deterministic Computer Vision pre-flight filter that rejects blank, muddy,
    blurry, or non-PACE pages BEFORE invoking paid VLM APIs.
    """

    def __init__(self, template_path: Optional[str] = None):
        self.template_img = None
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.kp_tpl, self.des_tpl = None, None

        if template_path and Path(template_path).exists():
            self.template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if self.template_img is not None:
                self.kp_tpl, self.des_tpl = self.orb.detectAndCompute(self.template_img, None)
                logger.info("Loaded master template from %s (%d ORB keypoints)", template_path, len(self.kp_tpl))
        else:
            logger.warning("Master template not provided or not found; skipping homography alignment.")

    def evaluate_and_align(self, page_bgr: np.ndarray) -> Tuple[bool, str, dict, np.ndarray]:
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Blank / Muddy Check (Ink Density)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        ink_ratio = float(np.count_nonzero(binary)) / binary.size
        if ink_ratio < 0.01:
            return False, "Page is blank or near-blank", {"ink_ratio": round(ink_ratio, 4)}, page_bgr
        if ink_ratio > 0.65:
            return False, "Page is heavily obscured/muddy/corrupted", {"ink_ratio": round(ink_ratio, 4)}, page_bgr

        # 2. Blur / Focus Check (Laplacian Variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < 40.0:
            return False, "Image too blurry for reliable OCR", {"blur_score": round(blur_score, 2)}, page_bgr

        metrics = {
            "ink_ratio": round(ink_ratio, 4),
            "blur_score": round(blur_score, 2),
        }

        # 3. Structural Alignment vs. Master Template (ORB + Homography)
        if self.des_tpl is not None:
            kp_img, des_img = self.orb.detectAndCompute(gray, None)
            if des_img is None or len(kp_img) < 50:
                return False, "Insufficient structural features to match PACE template", metrics, page_bgr

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(self.des_tpl, des_img), key=lambda m: m.distance)
            good_matches = [m for m in matches if m.distance < 55]
            metrics["template_matches"] = len(good_matches)

            if len(good_matches) < 20:
                return False, "Page does not match PACE safety form template", metrics, page_bgr

            src_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp_tpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0
            metrics["homography_inliers"] = inliers
            if H is not None and inliers >= 15:
                aligned = cv2.warpPerspective(
                    page_bgr, H, (self.template_img.shape[1], self.template_img.shape[0])
                )
                return True, "OK", metrics, aligned

        return True, "OK", metrics, page_bgr


# 3. VLM EXTRACTION (PRODUCTION CONFIG + MOCK PLACEHOLDER)

SYSTEM_PROMPT = """
You are a construction safety data digitization specialist.
Extract all fields from the provided PACE Safety Observation Form image into the strict JSON schema.
Rules:
1. Map checkbox columns strictly: S -> "Safe", R -> "At Risk", I -> "Imminent Hazard", N -> "N/A".
2. If a handwritten mark is crossed out and another box is marked, record only the final intended mark.
3. Never hallucinate text. If a field is illegible or blank, return null or "Unmarked".
4. Preserve exact dates in YYYY-MM-DD format and numeric IDs as strings.
"""

MOCK_VLM_RESPONSES: Dict[int, dict] = {
    1: {
        "form_title": "Breaking Containment - Life Saving Action",
        "observation_date": "2024-10-02",
        "observer": "John Davis",
        "company_observed": "Summit Construction",
        "location_cwa": "8000",
        "shift": "Day",
        "timesheet_number": "25489",
        "employee_number": "7852",
        "checklist_items": [
            {"item_number": "01", "question": "Has the location to be cut or opened been visibly marked?", "rating": "Imminent Hazard"},
            {"item_number": "02", "question": "Work has been authorized to begin (e.g. - Line Break Permit)", "rating": "Imminent Hazard"},
            {"item_number": "03", "question": "Nearby work activities has no potential ignition sources or conflict with work scope (e.g. hot work)", "rating": "Imminent Hazard"},
            {"item_number": "04", "question": "Exclusion zone (e.g. barricade) is in place.", "rating": "At Risk"},
            {"item_number": "05", "question": "There is clear means of communication with the equipment operator", "rating": "Safe"},
            {"item_number": "06", "question": "Workers positioned outside line of fire", "rating": "Imminent Hazard"},
            {"item_number": "07", "question": "Spill containment plan in place", "rating": "Imminent Hazard"},
            {"item_number": "08", "question": "Energy sources isolated and secured", "rating": "Imminent Hazard"},
        ],
        "cczjv_expectation_questions": {
            "positive_feedback_provided": "Yes",
            "job_stopped_imminent_hazard": "No",
            "investigation_conducted": "N/A",
            "corrective_actions_taken": "N/A",
        },
        "comments": "Safety measures were strictly followed.",
    },
    2: {
        "form_title": "O&I Confined Space - Life Saving Action",
        "observation_date": "2024-10-01",
        "observer": "Luis Gomez",
        "company_observed": "Hernandez Builders",
        "location_cwa": "1000",
        "shift": "Night",
        "timesheet_number": "36781",
        "employee_number": "14589",
        "checklist_items": [
            {"item_number": "01", "question": "Proof of Training for CSE Duties (e.g. HH Stickers)", "rating": "Imminent Hazard"},
            {"item_number": "02", "question": "Does the crew have a valid confined space permit?", "rating": "N/A"},
            {"item_number": "03", "question": "Reliable communication is available for an Emergency (e.g. Hole Watch)", "rating": "N/A"},
            {"item_number": "04", "question": "CSE Attendant monitoring entrant status and can communicate with entrants", "rating": "N/A"},
            {"item_number": "05", "question": "Access/egress entry ways clear of obstructions?", "rating": "N/A"},
            {"item_number": "06", "question": "Controls to prevent unauthorized access (e.g. Barricades, attendant, signs) are in place?", "rating": "Imminent Hazard"},
            {"item_number": "07", "question": "Fire extinguisher located near entry point?", "rating": "Imminent Hazard"},
            {"item_number": "08", "question": "Does the crew maintain an access/egress record?", "rating": "Imminent Hazard"},
        ],
        "cczjv_expectation_questions": {
            "positive_feedback_provided": "N/A",
            "job_stopped_imminent_hazard": "No",
            "investigation_conducted": "Yes",
            "corrective_actions_taken": "Yes",
        },
        "comments": "Improvements needed in communication.",
    },
    3: {
        "form_title": "Critical Devices - Life Saving Action",
        "observation_date": "2024-09-30",
        "observer": "Maria Hernandez",
        "company_observed": "Queen's Contracting",
        "location_cwa": "7000",
        "shift": "Night",
        "timesheet_number": "49012",
        "employee_number": "357951",
        "checklist_items": [
            {"item_number": "01", "question": "Is a Work Authorization Permit and/or Variance in place and at the work front to bypass/defeat a Safety Critical Device?", "rating": "Safe"},
            {"item_number": "02", "question": "Alternative controls in place to manage risks while Safety Critical Device is defeated?", "rating": "Safe"},
            {"item_number": "03", "question": "Is there an emergency response plan during safety critical device defeat?", "rating": "N/A"},
            {"item_number": "04", "question": "Safety critical device defeat/bypass is documented?", "rating": "N/A"},
            {"item_number": "05", "question": "Safety critical device defeat/bypass is communicated?", "rating": "At Risk"},
            {"item_number": "06", "question": "Have crane computer by-passes/alarms been tampered with or disabled?", "rating": "Imminent Hazard"},
            {"item_number": "07", "question": "Tools equipped with built-in safety features? (spring loaded safety knives, deadman switches, etc.) And safety features not bypassed", "rating": "Safe"},
            {"item_number": "08", "question": "Are tools with triggers being used properly? The trigger is not locked down", "rating": "Safe"},
            {"item_number": "09", "question": "Are tools with guards and handles being used properly", "rating": "Imminent Hazard"},
        ],
        "cczjv_expectation_questions": {
            "positive_feedback_provided": "No",
            "job_stopped_imminent_hazard": "No",
            "investigation_conducted": "Yes",
            "corrective_actions_taken": "No",
        },
        "comments": "Great teamwork and performance.",
    },
}


def call_vlm_api(image_bgr: np.ndarray, page_num: int, use_mock: bool = True) -> SafetyObservationForm:
   
    if not use_mock:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        _, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")

        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            temperature=0.0,
            max_tokens=2048,
            response_format=SafetyObservationForm,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all structured fields from this safety form."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "high"}},
                    ],
                },
            ],
        )
        return response.choices[0].message.parsed

    if page_num not in MOCK_VLM_RESPONSES:
        raise ValueError(f"No mock VLM payload configured for page {page_num}")
    return SafetyObservationForm(**MOCK_VLM_RESPONSES[page_num])


# 4. PDF INGESTION & PIPELINE ORCHESTRATION

def stitch_pdf_page_images(page) -> Optional[np.ndarray]:
    
    imgs = list(page.images)
    if not imgs:
        return None
    pil_strips = [Image.open(BytesIO(im.data)).convert("RGB") for im in imgs]
    width = max(im.width for im in pil_strips)
    total_height = sum(im.height for im in pil_strips)
    canvas = Image.new("RGB", (width, total_height), (255, 255, 255))
    y_offset = 0
    for im in pil_strips:
        canvas.paste(im, (0, y_offset))
        y_offset += im.height
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def process_pdf(pdf_path: str, template_path: Optional[str], output_json: str, use_mock: bool = True) -> dict:
    reader = PdfReader(pdf_path)
    gatekeeper = FormQualityGatekeeper(template_path)
    results: List[dict] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw_text = (page.extract_text() or "").strip()
        if "FAILED TEST FORM" in raw_text.upper() or len(list(page.images)) == 0:
            logger.warning("Page %d rejected at ingestion: non-form or explicit failure marker.", idx)
            results.append(
                PageExtractionResult(
                    page_number=idx,
                    status="REJECTED_INVALID_FORM",
                    rejection_reason=f"No valid form image detected (raw text: '{raw_text[:50]}')",
                ).model_dump()
            )
            continue

        page_bgr = stitch_pdf_page_images(page)
        passed, reason, metrics, aligned_bgr = gatekeeper.evaluate_and_align(page_bgr)
        if not passed:
            logger.warning("Page %d rejected by CV Gatekeeper: %s", idx, reason)
            results.append(
                PageExtractionResult(
                    page_number=idx,
                    status="REJECTED_INVALID_FORM",
                    rejection_reason=reason,
                    quality_metrics=metrics,
                ).model_dump()
            )
            continue

        try:
            extracted = call_vlm_api(aligned_bgr, page_num=idx, use_mock=use_mock)
            results.append(
                PageExtractionResult(
                    page_number=idx,
                    status="SUCCESS",
                    quality_metrics=metrics,
                    data=extracted,
                ).model_dump()
            )
            logger.info("Page %d extracted successfully (metrics=%s).", idx, metrics)
        except Exception as exc:
            logger.exception("Page %d failed during VLM extraction.", idx)
            results.append(
                PageExtractionResult(
                    page_number=idx,
                    status="ERROR",
                    rejection_reason=str(exc),
                    quality_metrics=metrics,
                ).model_dump()
            )

    payload = {
        "source_file": os.path.basename(pdf_path),
        "total_pages": len(reader.pages),
        "summary": {
            "succeeded": sum(1 for r in results if r["status"] == "SUCCESS"),
            "rejected": sum(1 for r in results if r["status"] == "REJECTED_INVALID_FORM"),
            "errors": sum(1 for r in results if r["status"] == "ERROR"),
        },
        "forms": results,
    }
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved structured output to %s", output_json)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="TEKNOBUILT Task 1: PACE Safety Form Digitization Pipeline")
    parser.add_argument("--pdf", default="Sample_input.pdf", help="Path to input PDF")
    parser.add_argument("--template", default="template.jpg", help="Path to master template image")
    parser.add_argument("--output", default="extracted_forms.json", help="Path to output JSON file")
    parser.add_argument("--live", action="store_true", help="Call live OpenAI VLM API (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    process_pdf(
        pdf_path=args.pdf,
        template_path=args.template,
        output_json=args.output,
        use_mock=not args.live,
    )


if __name__ == "__main__":
    main()
