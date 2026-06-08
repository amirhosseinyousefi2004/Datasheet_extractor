import json
import re
import sys
import os
from pathlib import Path

import fitz
# import pytesseract
from PIL import Image
from openpyxl import Workbook

# ========================================================
import tqdm

class NoOpTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        if self.iterable is None:
            return iter([])
        return iter(self.iterable)

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

tqdm.tqdm = NoOpTqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer, util

import traceback


FIELDS = [
    "Reference_DataSheet","Tag_number","Equipment","Position",
    "MDMT_Design","Temperature_Design","Temperature_Operating",
    "External_Pressure_Design","Internal_Pressure_Design",
    "Internal_Pressure_Operating","Internal_Diameter","Length",
    "Body_Material","Internal_Material","Insulation"
]

FIELD_DEFINITIONS = {
    "Tag_number":["tag number","equipment tag","tag","document no"],
    "Equipment":["equipment","separator","vessel","reactor","tank","drum","tower","silencer","filter","package","heat exchanger","deaerator"],
    "Position":["orientation","equipment orientation","mounting orientation"],
    "MDMT_Design":["mdmt","mdt","minimum design temperature"],
    "Temperature_Design":["design temperature","design tmp"],
    "Temperature_Operating":["operating temperature","normal temperature","operating tmp"],
    "External_Pressure_Design":["external pressure"],
    "Internal_Pressure_Design":["design pressure","internal pressure"],
    "Internal_Pressure_Operating":["operating pressure","working pressure"],
    "Internal_Diameter":["inside diameter","internal diameter"],
    "Length":["length","overall length","tan to tan","t.l.","total length"],
    "Body_Material":["shell material","body material","material of construction"],
    "Internal_Material":["lining material","internal material"],
    "Insulation":["insulation","cladding"]
}

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "bge-large-en-v1.5"

class DatasheetExtractor:

    def __init__(self):
        try:
            self.model = SentenceTransformer(str(model_path))
        except Exception:
            with open("error.log", "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)

                import sys
                f.write(f"\nstdout={sys.stdout}\n")
                f.write(f"stderr={sys.stderr}\n")

                if hasattr(sys.stdout, "encoding"):
                    f.write(f"stdout encoding={sys.stdout.encoding}\n")

                if hasattr(sys.stderr, "encoding"):
                    f.write(f"stderr encoding={sys.stderr.encoding}\n")

            raise
        self.results = []
        # self.model = SentenceTransformer(str(model_path))

        self.field_embeddings = {
            k: self.model.encode(v, convert_to_tensor=True)
            for k, v in FIELD_DEFINITIONS.items()
        }

    def process_pdf(self, pdf_path):
        blocks = self._extract_blocks(pdf_path)

        result = {
            f: {"value":"", "confidence":0.0, "source":""}
            for f in FIELDS
        }

        result["Reference_DataSheet"] = {
            "value": Path(pdf_path).name,
            "confidence": 1.0,
            "source": "filename"
        }

        field_candidates = {
            field: [] for field in FIELD_DEFINITIONS
        }

        for block in blocks:
            emb = self.model.encode(
                block["text"],
                convert_to_tensor=True
            )

            for field, field_emb in self.field_embeddings.items():

                similarity = util.cos_sim(
                    emb,
                    field_emb
                ).max().item()

                if similarity < 0.55:
                    continue

                neighbors = self._find_neighbors(block, blocks)

                for candidate_text, start_idx in self._generate_candidates(neighbors):
                    validation = self._validate(field, candidate_text)
                    compactness = self._compactness(candidate_text)
                    position_score = 1.0 / (1 + start_idx)

                    candidate_score = (
                        validation * 0.60 +
                        compactness * 0.25 +
                        position_score * 0.15
                    )

                    confidence = (
                        similarity * 0.40 +
                        candidate_score * 0.60
                    )

                    field_candidates[field].append({
                        "value": candidate_text,
                        "confidence": round(confidence, 3),
                        "source": f'label="{block["text"]}"'
                    })

        for field, candidates in field_candidates.items():
            if candidates:
                best = max(
                    candidates,
                    key=lambda x: x["confidence"]
                )

                if field == "Position":
                    best["value"] = self._normalize_orientation(
                        best["value"]
                    )

                result[field] = best

        self.results.append(result)
        return result

    def _extract_blocks(self, pdf_path):
        doc = fitz.open(pdf_path)
        blocks = []

        for page_num, page in enumerate(doc):
            text_dict = page.get_text("dict")

            page_has_text = False

            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            page_has_text = True
                            blocks.append({
                                "text": text,
                                "bbox": span["bbox"],
                                "page": page_num
                            })

            # if not page_has_text:
            #     pix = page.get_pixmap(dpi=300)
            #     img_path = "_ocr.png"
            #     pix.save(img_path)

            #     data = pytesseract.image_to_data(
            #         Image.open(img_path),
            #         output_type=pytesseract.Output.DICT
            #     )

            #     for i, txt in enumerate(data["text"]):
            #         txt = txt.strip()

            #         if txt:
            #             blocks.append({
            #                 "text": txt,
            #                 "bbox": (
            #                     data["left"][i],
            #                     data["top"][i],
            #                     data["left"][i] + data["width"][i],
            #                     data["top"][i] + data["height"][i]
            #                 ),
            #                 "page": page_num
            #             })

        return blocks

    def _find_neighbors(self, label_block, blocks):
        x0, y0, x1, y1 = label_block["bbox"]
        y_avg = (y0 + y1) / 2
        candidates = []

        for block in blocks:
            if block == label_block or block["page"] != label_block["page"]:
                continue

            bx0, by0, bx1, by1 = block["bbox"]
            by_avg = (by0 + by1) / 2

            dx = bx0 - x1
            dy = y_avg - by_avg

            same_row = (
                0 <= dx <= 500 and -7 <= dy <= 10
            )

            below = (
                0 <= by0 - y1 <= 40 and
                0 <= bx0 - x0 <= 200
            )

            if same_row:
                priority = 0
            elif below:
                priority = 1
            else:
                continue

            distance = max(dx, 0) + 12 * abs(dy)
            candidates.append((priority, distance, block))

        candidates.sort(key=lambda x: (x[0], x[1]))
        return [h[2] for h in candidates[:5]]

    def _generate_candidates(self, neighbors):
        texts = [n["text"] for n in neighbors]
        out = []

        for start in range(len(texts)):
            for end in range(start + 1, len(texts) + 1):
                out.append((" ".join(texts[start:end]).strip(), start))

        return out

    def _compactness(self, text):
        n = max(1, len(text.split()))
        return 1.0 / (1 + 0.25 * (n - 1))

    def _validate(self, field, value):
        if not value:
            return 0.0

        if "Pressure" in field:
            if (re.search(r'[-+]?\d+(\.\d+)?\s*(bar|barg|psi|psig|kpa|mpa)', value, re.I) or
                re.search(r'(bar|barg|psi|psig|kpa|mpa)\s*[-+]?\d+(\.\d+)?', value, re.I)):
                return 1.0

        elif "Temperature" in field or "MDMT" in field:
            if (re.search(r'[-+]?\d+(\.\d+)?\s*(°c|°f|°C|°F)', value, re.I) or
                re.search(r'(°c|°f|°C|°F)\s*[-+]?\d+(\.\d+)?', value, re.I)):
                return 1.0

        elif "Material" in field:
            if re.search(r'(sa[- ]?\d+|ss\d+|aisi|astm|tp\d+)', value, re.I):
                return 1.0

        elif field == "Position":
            if re.search(r'\b(v|h|vertical|horizontal)\b', value, re.I):
                return 1.0

        elif field in ["Length", "Internal_Diameter"]:
            if (re.search(r'\d+.*\s*(mm|cm|m|in|inch|ft)', value, re.I) or
            re.search(r'\d+(\.\d+)?\s*(mm|cm|m|in|inch|ft)', value, re.I)):
                return 1.0

        return 0.4

    def _normalize_orientation(self, text):
        t = text.strip().lower()

        if t == "v":
            return "Vertical"
        if t == "h":
            return "Horizontal"
        if "vertical" in t:
            return "Vertical"
        if "horizontal" in t:
            return "Horizontal"

        return text

    def export_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

    def export_excel(self, path):
        wb = Workbook()
        ws = wb.active
        ws.title = "Extracted_Data"

        ws.append(FIELDS)

        for row in self.results:
            ws.append([
                row.get(field, {}).get("value", "")
                for field in FIELDS
            ])

        wb.save(path)
