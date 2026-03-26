"""
Phase 5 — AI Knowledge Integration
=====================================
Takes identified plant name → returns structured medicinal knowledge
using Gemini 2.5 Flash API with hallucination safeguards.

Usage (standalone):
    python src/api/knowledge.py --plant "Neem"
    python src/api/knowledge.py --plant "Tulsi" --save

Usage (as module):
    from src.api.knowledge import get_plant_knowledge
    info = get_plant_knowledge("Neem", confidence=98.5)
"""

import os
import json
import argparse
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Trusted plant database for RAG-style grounding ───────────────────────────

TRUSTED_PLANT_DB = {
    "Neem": {
        "scientific_name":  "Azadirachta indica",
        "family":           "Meliaceae",
        "ayush_category":   "Ayurvedic",
    },
    "Tulsi": {
        "scientific_name":  "Ocimum tenuiflorum",
        "family":           "Lamiaceae",
        "ayush_category":   "Ayurvedic",
    },
    "Aloevera": {
        "scientific_name":  "Aloe vera",
        "family":           "Asphodelaceae",
        "ayush_category":   "Ayurvedic",
    },
    "Amla": {
        "scientific_name":  "Phyllanthus emblica",
        "family":           "Phyllanthaceae",
        "ayush_category":   "Ayurvedic",
    },
    "Ashwagandha": {
        "scientific_name":  "Withania somnifera",
        "family":           "Solanaceae",
        "ayush_category":   "Ayurvedic",
    },
}


# ── Build Prompt ──────────────────────────────────────────────────────────────

def build_prompt(plant_name: str, confidence: float, db_context: dict) -> str:
    context_str = ""
    if db_context:
        context_str = f"""
Known verified information about this plant:
- Scientific name : {db_context.get('scientific_name', 'Unknown')}
- Family          : {db_context.get('family', 'Unknown')}
- AYUSH category  : {db_context.get('ayush_category', 'Unknown')}
"""

    return f"""You are a medicinal plant expert specializing in Indian Ayurvedic and traditional medicine.
A computer vision model has identified a plant leaf image as "{plant_name}" with {confidence:.1f}% confidence.
{context_str}
Provide accurate, research-backed medicinal information about this plant.

IMPORTANT RULES:
- Only include information you are highly confident about
- If unsure about any detail, write "Limited research available" instead of guessing
- Do not invent dosages or medical claims
- Always include safety warnings
- Base your response on traditional Ayurvedic texts and modern phytochemical research

Respond ONLY with a valid JSON object in exactly this format.
No preamble, no explanation, no markdown backticks — pure JSON only:
{{
  "plant_name": "{plant_name}",
  "scientific_name": "scientific name here",
  "common_names": ["name1", "name2"],
  "family": "plant family",
  "parts_used": ["leaf", "bark", "root"],
  "medicinal_benefits": [
    "benefit 1 with brief explanation",
    "benefit 2 with brief explanation",
    "benefit 3 with brief explanation"
  ],
  "traditional_uses": [
    "traditional use 1",
    "traditional use 2",
    "traditional use 3"
  ],
  "active_compounds": ["compound1", "compound2"],
  "dosage_guidelines": {{
    "leaf_juice": "dosage if applicable",
    "powder": "dosage if applicable",
    "decoction": "dosage if applicable"
  }},
  "safety_warnings": [
    "warning 1",
    "warning 2"
  ],
  "drug_interactions": ["interaction 1 if known"],
  "ayush_reference": "AYUSH or classical text reference if known",
  "confidence_note": "note about information reliability"
}}"""


# ── Call Gemini 2.5 Flash API ─────────────────────────────────────────────────

def call_gemini_api(prompt: str) -> dict:
    """Call Gemini 2.5 Flash and parse structured JSON response."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

    client   = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = prompt,
        config   = types.GenerateContentConfig(
            temperature = 0.2,   # low temp for factual/structured output
            top_p       = 0.95,
        ),
    )

    raw_text = response.text.strip()

    # Strip markdown backticks if model includes them
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    return json.loads(raw_text)


# ── Hallucination Safeguard ───────────────────────────────────────────────────

def validate_response(data: dict, plant_name: str) -> dict:
    """Basic validation to catch obvious issues. Adds warning flags."""
    issues = []

    if data.get("plant_name", "").lower() != plant_name.lower():
        issues.append(f"Plant name mismatch: expected {plant_name}")

    if not data.get("scientific_name") or data["scientific_name"] == "Unknown":
        issues.append("Scientific name missing — verify manually")

    if not data.get("medicinal_benefits"):
        issues.append("No medicinal benefits returned")

    if not data.get("safety_warnings"):
        issues.append("No safety warnings — always verify before use")

    if issues:
        data["validation_warnings"] = issues

    return data


# ── Main entry point (used by Streamlit app + CLI) ───────────────────────────

def get_plant_knowledge(plant_name: str, confidence: float = 100.0) -> dict:
    """
    Main entry point for Phase 5.
    Returns structured medicinal knowledge for the identified plant.

    Args:
        plant_name  : Plant class name from ML model
        confidence  : Model confidence (%) from inference

    Returns:
        dict with full structured medicinal information
    """
    db_context = TRUSTED_PLANT_DB.get(plant_name, {})
    prompt     = build_prompt(plant_name, confidence, db_context)

    print(f"  Fetching medicinal knowledge: {plant_name} ({confidence:.1f}% confidence)...")
    data = call_gemini_api(prompt)
    data = validate_response(data, plant_name)

    return data


# ── Format for CLI display ────────────────────────────────────────────────────

def format_display(data: dict) -> str:
    lines = [
        f"\n{'='*60}",
        f"  {data.get('plant_name')}  ({data.get('scientific_name', '')})",
        f"  Family: {data.get('family', 'Unknown')}",
        f"  Common Names: {', '.join(data.get('common_names', []))}",
        f"{'='*60}",
        "\n  MEDICINAL BENEFITS:",
    ]
    for b in data.get("medicinal_benefits", []):
        lines.append(f"    • {b}")

    lines.append("\n  TRADITIONAL USES:")
    for u in data.get("traditional_uses", []):
        lines.append(f"    • {u}")

    lines.append(f"\n  PARTS USED: {', '.join(data.get('parts_used', []))}")
    lines.append(f"  ACTIVE COMPOUNDS: {', '.join(data.get('active_compounds', []))}")

    lines.append("\n  DOSAGE GUIDELINES:")
    for form, dose in data.get("dosage_guidelines", {}).items():
        if dose and dose.lower() not in ["n/a", "not applicable", ""]:
            lines.append(f"    {form}: {dose}")

    lines.append("\n  SAFETY WARNINGS:")
    for w in data.get("safety_warnings", []):
        lines.append(f"    ⚠️  {w}")

    if data.get("drug_interactions"):
        lines.append("\n  DRUG INTERACTIONS:")
        for d in data["drug_interactions"]:
            lines.append(f"    • {d}")

    if data.get("validation_warnings"):
        lines.append("\n  VALIDATION NOTES:")
        for v in data["validation_warnings"]:
            lines.append(f"    ⚠️  {v}")

    lines.append(f"\n  Reference: {data.get('ayush_reference', 'Not specified')}")
    lines.append(f"  Note: {data.get('confidence_note', '')}")
    lines.append(f"{'='*60}\n")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant",      type=str,   required=True)
    parser.add_argument("--confidence", type=float, default=100.0)
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    data = get_plant_knowledge(args.plant, args.confidence)
    print(format_display(data))

    if args.save:
        os.makedirs("outputs/inference", exist_ok=True)
        path = f"outputs/inference/{args.plant}_knowledge.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved -> {path}")