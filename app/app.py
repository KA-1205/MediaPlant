"""
app/app.py — MediPlant AI Streamlit Application

Tabs:
    1. Plant Identification  — upload → classify → RAG knowledge
    2. Strategy Comparison   — all strategies on same image
    3. About                 — architecture, k-fold results, how it works

Final Evaluation (April 2026):
    Production Default: Simple Average = 98.16%
    Research Baseline: Learned Stacking (OOF) = 97.87% (Leakage resolved)
    Failed Stacking: Learned Stacking (Soup) = 94.40% (Distribution mismatch)

Run:
    streamlit run app/app.py
"""

import os, sys, json, yaml
import torch, torch.nn.functional as F
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

from src.data.transforms import get_val_transforms
from src.models.swin_model import get_swin_model
from src.models.convnext_model import get_convnext_model
from src.models.ensemble import (LearnedStackingEnsemble,
                                  simple_average_ensemble, weighted_average_ensemble)
from src.api.knowledge import get_plant_knowledge

st.set_page_config(page_title="MediPlant AI", page_icon="🌿",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid #e5e7eb;}
.stTabs [data-baseweb="tab"]{font-size:.9rem;font-weight:500;color:#6b7280;padding:.5rem 1.2rem;border:none;background:transparent;}
.stTabs [aria-selected="true"]{color:#111827!important;border-bottom:2px solid #111827!important;background:transparent!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.5rem;}
[data-testid="stMetric"]{background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:.9rem;}
[data-testid="stMetricValue"]{font-size:1.3rem!important;color:#111827;font-weight:600;}
[data-testid="stMetricLabel"]{font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;}
.stProgress>div>div{background-color:#374151!important;border-radius:4px;}
[data-testid="stFileUploader"]{border:1.5px dashed #d1d5db;border-radius:10px;}
.stButton>button{background:#111827;color:#fff;border:none;border-radius:7px;font-size:.9rem;font-weight:500;}
hr{border-color:#f3f4f6;margin:1.2rem 0;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-thumb{background:#e5e7eb;}
</style>
""", unsafe_allow_html=True)

REQUIRED = ["configs/config.yaml",
            "outputs/checkpoints/best_swin.pth",
            "outputs/checkpoints/best_convnext.pth",
            "outputs/checkpoints/weighted_avg_config.json",
            "data/splits/class_map.csv"]

@st.cache_resource(show_spinner=False)
def load_models():
    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(f"  {f}" for f in missing))

    with open("configs/config.yaml") as f: cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc     = cfg["data"]["num_classes"]

    swin = get_swin_model(nc, False)
    swin.load_state_dict(torch.load("outputs/checkpoints/best_swin.pth", map_location=device))
    swin = swin.to(device).eval()

    conv = get_convnext_model(nc, False)
    conv.load_state_dict(torch.load("outputs/checkpoints/best_convnext.pth", map_location=device))
    conv = conv.to(device).eval()

    meta = None
    if os.path.exists("outputs/checkpoints/best_ensemble.pth"):
        meta = LearnedStackingEnsemble(nc, cfg["ensemble"]["meta_hidden_dim"])
        meta.load_state_dict(torch.load("outputs/checkpoints/best_ensemble.pth", map_location=device))
        meta.eval()

    with open("outputs/checkpoints/weighted_avg_config.json") as f: wavg = json.load(f)
    names = pd.read_csv("data/splits/class_map.csv").sort_values("label")["class_name"].tolist()

    return {"swin":swin,"conv":conv,"meta":meta,"wavg":wavg,
            "class_names":names,"device":device,"cfg":cfg}


@torch.no_grad()
def get_logits(img, M):
    t = get_val_transforms(M["cfg"]["data"]["image_size"])
    x = t(img).unsqueeze(0).to(M["device"])
    return M["swin"](x).cpu(), M["conv"](x).cpu()


def apply_strategy(ls, lc, strategy, M):
    if strategy == "simple_avg":
        return simple_average_ensemble(ls, lc).squeeze(0)
    elif strategy == "weighted_avg":
        return weighted_average_ensemble(ls,lc,M["wavg"]["w_swin"],M["wavg"]["w_convnext"]).squeeze(0)
    elif strategy == "learned_stacking" and M["meta"] is not None:
        return F.softmax(M["meta"](ls,lc),dim=1).squeeze(0)
    elif strategy == "swin":
        return F.softmax(ls,dim=1).squeeze(0)
    elif strategy == "convnext":
        return F.softmax(lc,dim=1).squeeze(0)
    return simple_average_ensemble(ls, lc).squeeze(0)


def top3(probs, names):
    v,i = torch.topk(probs,3)
    return [{"rank":r+1,"class":names[idx],"label_idx":idx.item(),"confidence":round(val.item()*100,2)}
            for r,(val,idx) in enumerate(zip(v,i))]

# ── Load models ────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading models..."): M = load_models()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 MediPlant AI")
    st.divider()
    st.markdown("**Architecture**")
    st.markdown("- Swin Transformer Tiny\n- ConvNeXt Tiny\n- Simple Average Ensemble")
    st.divider()
    st.markdown("**Dataset**")
    st.markdown("- Indian Medicinal Leaves\n- 80 species · 6,900 images")
    st.divider()
    rp = "outputs/results/final_test_results.json"
    if os.path.exists(rp):
        with open(rp) as f: r = json.load(f)
        best = max(r, key=lambda k: r[k]["accuracy"])
        st.markdown("**Best Test Accuracy**")
        st.markdown(f"## {r[best]['accuracy']*100:.2f}%")
        st.caption(f"{best}")
    else:
        st.markdown("**Best Test Accuracy**")
        st.markdown("## 98.16%")
        st.caption("Simple Average (Production Default)")
    st.divider()
    st.markdown("**Knowledge API**")
    st.markdown("Gemini 2.5 Flash + RAG")

st.markdown("# 🌿 MediPlant AI")
st.markdown("Identify Indian medicinal plant leaves using **Swin + ConvNeXt Simple Average ensemble** "
            "(98.16% test accuracy), then retrieve Ayurvedic knowledge via **Gemini 2.5 Flash RAG pipeline**.")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍  Plant Identification","📊  Strategy Comparison","ℹ️  About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PLANT IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    cl_, cr_ = st.columns([1,1.6], gap="large")

    with cl_:
        st.markdown("#### Upload Leaf Image")
        up = st.file_uploader("upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if up:
            img = Image.open(up).convert("RGB")
            st.image(img, use_container_width=True, caption="Uploaded leaf")
            st.caption("Best results: single leaf, plain background, good lighting.")

    with cr_:
        st.markdown("#### Stage 1 — Classification")
        if up:
            with st.spinner("Running Simple Average ensemble..."):
                ls, lc = get_logits(img, M)
                probs  = apply_strategy(ls, lc, "simple_avg", M)
                t3     = top3(probs, M["class_names"])

            st.success(f"**Predicted Plant:  {t3[0]['class']}**")
            st.markdown("**Ensemble Confidence**")
            st.progress(min(int(t3[0]["confidence"]),100),
                        text=f"{t3[0]['confidence']:.1f}%  ·  Simple Average (98.16% test accuracy)")
            st.markdown("**Top-3 Predictions**")
            df_t3 = pd.DataFrame(t3)[["rank","class","confidence"]]
            df_t3.columns = ["Rank","Plant","Confidence (%)"]
            st.dataframe(df_t3, hide_index=True, use_container_width=True)

            with st.expander("🔬 Model Diagnostics"):
                ps = F.softmax(ls,dim=1).squeeze(0)
                pc_ = F.softmax(lc,dim=1).squeeze(0)
                ti  = t3[0]["label_idx"]
                d1,d2 = st.columns(2)
                d1.metric("Swin Transformer", f"{ps[ti].item()*100:.1f}%",
                          help="Global leaf structure — shape, venation, spatial layout")
                d2.metric("ConvNeXt",         f"{pc_[ti].item()*100:.1f}%",
                          help="Local texture — surface detail, fine veins, edges")
                c1,c2 = st.columns(2)
                def t5df(probs_):
                    v_,i_ = torch.topk(probs_,5)
                    return pd.DataFrame({"Plant":[M["class_names"][x] for x in i_],
                                         "%":[round(x.item()*100,2) for x in v_]})
                with c1: st.caption("Swin Top-5"); st.dataframe(t5df(ps), hide_index=True, use_container_width=True)
                with c2: st.caption("ConvNeXt Top-5"); st.dataframe(t5df(pc_), hide_index=True, use_container_width=True)
        else:
            st.markdown("<div style='text-align:center;padding:4rem;color:#d1d5db;'>"
                        "<div style='font-size:3rem'>🌱</div>"
                        "<p style='color:#9ca3af;margin-top:.8rem'>Upload a leaf image to begin</p></div>",
                        unsafe_allow_html=True)

    if up and "t3" in dir():
        st.divider()
        st.markdown("#### Stage 2 — Medicinal Knowledge")
        st.caption(f"Retrieving for **{t3[0]['class']}** · RAG: local plant_kb.json → Gemini 2.5 Flash")
        with st.spinner("Fetching from Gemini 2.5 Flash..."):
            try:
                info = get_plant_knowledge(t3[0]["class"], t3[0]["confidence"])
            except Exception as e:
                st.error(f"Knowledge retrieval failed: {e}")
                st.info("Set GEMINI_API_KEY in .env file."); st.stop()

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Scientific Name", info.get("scientific_name","—"))
        k2.metric("Family",          info.get("family","—"))
        k3.metric("Parts Used",      ", ".join(info.get("parts_used",[])[:2]))
        k4.metric("Active Compounds",f"{len(info.get('active_compounds',[]))} identified")

        grounded = str(info.get("rag_grounded","false")).lower() == "true"
        (st.success if grounded else st.warning)(
            ("✓ RAG grounded — verified knowledge base context used"
             if grounded else "⚠ Not in local KB — Gemini only. Verify with practitioner."),
            icon=("🔒" if grounded else "⚠️"))

        if info.get("common_names"): st.caption("Also known as: " + ", ".join(info["common_names"]))

        cl2,cr2 = st.columns(2, gap="large")
        with cl2:
            st.markdown("**✅ Medicinal Benefits**")
            st.info("\n".join(f"- {b}" for b in info.get("medicinal_benefits",[])))
            st.markdown("**🌿 Traditional Uses**")
            st.info("\n".join(f"- {u}" for u in info.get("traditional_uses",[])))
        with cr2:
            st.markdown("**💊 Dosage Guidelines**")
            d = {k:v for k,v in info.get("dosage_guidelines",{}).items()
                 if v and v.upper() not in ["N/A",""]}
            (st.warning("\n".join(f"**{k.replace('_',' ').title()}:** {v}" for k,v in d.items()))
             if d else st.warning("Consult qualified Ayurvedic practitioner."))
            st.markdown("**⚠️ Safety Warnings**")
            st.error("\n".join(f"- {w}" for w in info.get("safety_warnings",[])))
            if info.get("drug_interactions"):
                with st.expander("🔗 Drug Interactions"):
                    for d_ in info["drug_interactions"]: st.markdown(f"- {d_}")

        if info.get("active_compounds"):
            st.markdown("**Active Compounds:** " + ", ".join(info["active_compounds"]))
        if info.get("ayush_reference"):
            st.caption(f"📚 {info['ayush_reference']}")
        if info.get("validation_warnings"):
            with st.expander("⚠ Validation Notes"):
                for v in info["validation_warnings"]: st.markdown(f"- {v}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STRATEGY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Strategy Comparison")
    st.caption("Local evaluation: Simple Average = **98.16%** (Production Default)")

    if not up:
        st.info("Upload an image in **Plant Identification** tab first.")
    else:
        with st.spinner("Running all strategies..."):
            ls2, lc2 = get_logits(img, M)
            strat_map = {"Swin Transformer":"swin","ConvNeXt":"convnext",
                         "Simple Average":"simple_avg","Weighted Average":"weighted_avg",
                         "Learned Stacking":"learned_stacking"}
            results = {n: top3(apply_strategy(ls2,lc2,k,M), M["class_names"])
                       for n,k in strat_map.items()}

        st.markdown("### Top-1 Prediction Per Strategy")
        badges = {"Swin Transformer":"98.07% (Base)","ConvNeXt":"96.91% (Base)",
                  "Simple Average":"98.16% ⭐ BEST","Weighted Average":"98.16% ⭐ BEST",
                  "Learned Stacking":"97.87% (OOF Research Fix)"}
        cols = st.columns(5)
        for col,(name,res) in zip(cols,results.items()):
            col.metric(f"{name}", res[0]["class"],
                       f"{res[0]['confidence']:.1f}% | {badges[name]}")

        st.divider()
        st.markdown("### Confidence Comparison")
        chart = pd.DataFrame([{"Strategy":n,"Plant":r[0]["class"],
                                "Confidence (%)":r[0]["confidence"]}
                               for n,r in results.items()])
        fig = px.bar(chart, x="Strategy", y="Confidence (%)", color="Plant",
                     text=chart["Confidence (%)"].apply(lambda v:f"{v:.1f}%"),
                     height=350, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="outside")
        fig.update_layout(plot_bgcolor="#fff",paper_bgcolor="#fff",font_family="Inter",
                          yaxis=dict(range=[0,110]),margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Detailed Top-3 Per Strategy")
        for name, res in results.items():
            with st.expander(f"**{name}** — {res[0]['class']} ({res[0]['confidence']:.1f}%) | {badges[name]}"):
                d = pd.DataFrame(res)[["rank","class","confidence"]]
                d.columns = ["Rank","Plant","Confidence (%)"]
                st.dataframe(d, hide_index=True, use_container_width=True)

        rp = "outputs/results/final_test_results.json"
        if os.path.exists(rp):
            st.divider(); st.markdown("### Test Set Accuracy")
            with open(rp) as f: tr = json.load(f)
            pf = [{"Strategy":n,"Test Accuracy (%)":round(tr.get(n,{}).get("accuracy",0)*100,2),
                   "F1 Macro (%)":round(tr.get(n,{}).get("f1_macro",0)*100,2)}
                  for n in strat_map.keys() if n in tr]
            if pf:
                dfp = pd.DataFrame(pf); pc = ["Test Accuracy (%)","F1 Macro (%)"]
                st.dataframe(dfp.style.highlight_max(subset=pc,color="#dcfce7")
                             .format("{:.2f}",subset=pc), hide_index=True, use_container_width=True)
        else:
            st.info("Run `python scripts/evaluate.py` to generate test accuracy table.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("ℹ️ About MediPlant AI"); st.divider()

    st.markdown("### What is it?")
    st.markdown("An end-to-end system identifying **80 Indian medicinal plant species** "
                "from leaf images using a Swin Transformer + ConvNeXt ensemble, with "
                "Ayurvedic knowledge retrieval via a RAG pipeline.")

    c1,c2 = st.columns(2)
    c1.info("**Dataset**\n\n- Indian Medicinal Leaves (Kaggle: aryashah2k)\n"
            "- 80 species · 6,900 images\n- 70 / 15 / 15 split\n- CC-BY-NC-SA-4.0")
    c2.info("**Task**\n\n- Fine-grained leaf classification\n"
            "- Input: single leaf image (any size)\n"
            "- Output: species + confidence\n- Metric: Top-1 Accuracy + F1 Macro")

    st.divider(); st.markdown("### Architecture")
    a1,a2,a3 = st.columns(3)
    a1.info("**Swin Transformer Tiny**\n\n"
            "Hierarchical attention → global leaf structure\n\n"
            "Pretrained: ImageNet-1K\n\n"
            "Val accuracy: 98.07%\n\n"
            "Captures shape, venation, spatial layout")
    a2.info("**ConvNeXt Tiny**\n\n"
            "Modernized CNN → local texture\n\n"
            "Pretrained: ImageNet-1K\n\n"
            "Val accuracy: 96.91%\n\n"
            "Captures surface detail, fine veins, edges")
    a3.info("**Simple Average Ensemble**\n\n"
            "Equal weight average (0.5 / 0.5)\n\n"
            "Test accuracy: **98.16%**\n\n"
            "Production Default — robust and reliable")

    st.divider(); st.markdown("### The K-Fold Stacking Experiment")
    st.markdown("A deep dive into model soups and logit distribution mismatch (April 2026).")
    kfold_data = {
        "Strategy":        ["Swin (Base)","ConvNeXt (Base)",
                            "Simple Average (Base)","Learned Stacking (OOF)", "Learned Stacking (Soup)"],
        "Test Accuracy (%)": [98.07, 96.91, 98.16, 97.87, 94.40],
        "Notes": [
            "Original single-run",
            "Original single-run",
            "BEST — Production Default",
            "Leak-free K-Fold meta-classifier",
            "Negative Result: Distribution mismatch",
        ]
    }
    df_kf = pd.DataFrame(kfold_data)
    st.dataframe(
        df_kf.style.highlight_max(subset=["Test Accuracy (%)"], color="#dcfce7"),
        hide_index=True, use_container_width=True
    )

    st.info(
        "**The Distribution Mismatch Finding:**\n\n"
        "Applying a meta-classifier to 'Model Soups' (averaged K-Fold weights) crashed the accuracy to 94.40%. "
        "Why? Averaging the weights shifted the underlying probability distributions the meta-classifier was trained on. "
        "By re-engineering the pipeline to extract strict Out-Of-Fold (OOF) logits, the data leakage was resolved and "
        "performance rebounded to a scientifically rigorous 97.87%.",
        icon="🔬"
    )

    st.divider(); st.markdown("### RAG Pipeline")
    r1,r2,r3 = st.columns(3)
    r1.info("**1. Retrieve**\n\nLook up plant in plant_kb.json — "
            "verified data for 35+ Indian plants: scientific names, "
            "active compounds, classical Ayurvedic references, safety warnings.")
    r2.info("**2. Augment**\n\nInject KB facts into Gemini prompt "
            "as VERIFIED CONTEXT BLOCK. Gemini treats them as ground "
            "truth — cannot contradict them.")
    r3.info("**3. Generate**\n\nGemini 2.5 Flash at temperature=0.1 "
            "for factual accuracy. Response validated for scientific "
            "name consistency and hallucination flags.")

    st.divider(); st.markdown("### How to Run")
    st.code("pip install -r requirements.txt\n"
            "echo GEMINI_API_KEY=your_key >> .env\n"
            "streamlit run app/app.py", language="bash")