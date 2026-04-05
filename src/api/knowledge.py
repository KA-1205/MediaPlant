"""
src/api/knowledge.py
=====================
RAG-based medicinal plant knowledge retrieval using Gemini 2.5 Flash.

Pipeline:
    1. RETRIEVE  — look up plant in src/data/plant_kb.json (local KB)
    2. AUGMENT   — inject retrieved facts into Gemini prompt
    3. GENERATE  — Gemini 2.5 Flash at temperature=0.1
    4. VALIDATE  — cross-check response against KB

Why RAG instead of plain Gemini:
    Plain Gemini generates from parametric memory → hallucination risk,
    especially for rare Indian medicinal plants.
    With RAG: verified facts injected as grounding context → reduced
    hallucination, especially for safety-critical information.
"""

import os
import json
import argparse
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

KB_PATH = Path(__file__).parent.parent / "data" / "plant_kb.json"


def load_kb() -> dict:
    if not KB_PATH.exists():
        return {}
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


def retrieve(plant_name: str, kb: dict) -> dict:
    """Case-insensitive, underscore-normalized lookup."""
    if plant_name in kb:
        return kb[plant_name]
    norm = plant_name.lower().replace("_", " ").strip()
    for name, data in kb.items():
        if name.lower().replace("_", " ").strip() == norm:
            return data
    return {}


def build_prompt(plant_name: str, confidence: float, context: dict) -> str:
    has_ctx = bool(context)
    if has_ctx:
        ctx = ("VERIFIED KNOWLEDGE BASE CONTEXT (ground truth — do not contradict):\n"
               + "\n".join([
                   f"Scientific: {context.get('scientific_name','')}",
                   f"Family: {context.get('family','')}",
                   f"Parts: {', '.join(context.get('parts_used',[]))}",
                   f"Compounds: {', '.join(context.get('active_compounds',[]))}",
                   f"Reference: {context.get('classical_reference','')}",
                   "Benefits: " + "; ".join(context.get("known_benefits", [])),
                   "Warnings: " + "; ".join(context.get("safety_warnings", [])),
                   "Interactions: " + "; ".join(context.get("drug_interactions", [])),
               ]))
        inst = "Expand on verified context with additional research. Never contradict it."
    else:
        ctx  = "Plant not in local KB. Be conservative — only well-established facts."
        inst = "Write 'Limited research available' for anything uncertain."

    return f"""You are a medicinal plant expert specializing in Indian Ayurvedic medicine.
A computer vision model identified a leaf as "{plant_name}" with {confidence:.1f}% confidence.

{ctx}

{inst}

Respond ONLY with valid JSON, no markdown:
{{
  "plant_name": "{plant_name}",
  "scientific_name": "binomial nomenclature",
  "common_names": ["name1", "name2"],
  "family": "botanical family",
  "parts_used": ["part1", "part2"],
  "medicinal_benefits": ["benefit with mechanism", "benefit", "benefit", "benefit"],
  "traditional_uses": ["use with preparation", "use", "use"],
  "active_compounds": ["compound1", "compound2"],
  "dosage_guidelines": {{
    "leaf_juice": "dosage or N/A",
    "powder": "dosage or N/A",
    "decoction": "dosage or N/A",
    "oil": "dosage or N/A"
  }},
  "safety_warnings": ["specific warning with who/why", "warning", "warning"],
  "drug_interactions": ["interaction with mechanism"],
  "ayush_reference": "classical text and chapter if known",
  "rag_grounded": {"true" if has_ctx else "false"},
  "confidence_note": "brief reliability note"
}}"""


def call_gemini(prompt: str) -> dict:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not in .env — get free key at aistudio.google.com")
    client = genai.Client(api_key=key)
    resp   = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = prompt,
        config   = types.GenerateContentConfig(temperature=0.1, top_p=0.9),
    )
    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())


def validate(data: dict, plant_name: str, context: dict) -> dict:
    issues = []
    if data.get("plant_name", "").lower() != plant_name.lower():
        issues.append(f"Plant name mismatch: expected '{plant_name}'")
    if context:
        kb_sci = context.get("scientific_name", "")
        g_sci  = data.get("scientific_name", "")
        if kb_sci and g_sci and kb_sci.lower() not in g_sci.lower():
            issues.append(f"Scientific name differs: KB='{kb_sci}' Gemini='{g_sci}'")
    if not data.get("safety_warnings"):
        issues.append("No safety warnings — consult Ayurvedic practitioner")
    if not context:
        issues.append("Not in local KB — verify with qualified practitioner")
    if issues:
        data["validation_warnings"] = issues
    return data


def get_plant_knowledge(plant_name: str, confidence: float = 100.0) -> dict:
    """
    Main RAG pipeline: retrieve → augment → generate → validate.

    Args:
        plant_name:  Plant class name from ML model
        confidence:  Ensemble confidence score (%)

    Returns:
        Structured dict with medicinal knowledge and rag_grounded flag
    """
    kb      = load_kb()
    context = retrieve(plant_name, kb)
    print(f"  [RAG] {plant_name} | KB: {'found' if context else 'not found'} | {confidence:.1f}%")
    data = call_gemini(build_prompt(plant_name, confidence, context))
    return validate(data, plant_name, context)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plant",      required=True)
    p.add_argument("--confidence", type=float, default=100.0)
    p.add_argument("--save",       action="store_true")
    a = p.parse_args()
    result = get_plant_knowledge(a.plant, a.confidence)
    print(json.dumps(result, indent=2))
    if a.save:
        import os; os.makedirs("outputs/inference", exist_ok=True)
        with open(f"outputs/inference/{a.plant}_knowledge.json", "w") as f:
            json.dump(result, f, indent=2)
