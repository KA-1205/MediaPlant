# MediPlant AI

A research prototype that classifies leaf images with Swin Transformer and ConvNeXt, compares ensemble strategies, and requests structured plant information from Gemini. The interface is built with Streamlit.

**Status:** source and training scripts are available. Training images, trained checkpoints, and measured evaluation artifacts are not included in this checkout. No accuracy, clinical reliability, or deployment result is claimed here. Generated plant information must not be used to choose remedies or dosages.

## Implementation

| Component | Source |
| --- | --- |
| Image preprocessing and dataset loading | [src/data](src/data/) |
| Swin, ConvNeXt, and ensemble strategies | [src/models](src/models/) |
| Training and held-out evaluation scripts | [scripts](scripts/) |
| Streamlit classification and comparison interface | [app/app.py](app/app.py) |
| Gemini prompt, local plant metadata, and response checks | [src/api/knowledge.py](src/api/knowledge.py) |

The knowledge step uses a small, hardcoded metadata dictionary as prompt context. It is not retrieval from a validated medical literature corpus, and its basic response checks do not establish factual correctness.

## Prepare and run

Use a Python environment compatible with PyTorch and the packages in [requirements.txt](requirements.txt). Training is configured for CUDA in [configs/config.yaml](configs/config.yaml); review the device, class count, and Weights & Biases settings before running.

```bash
git clone https://github.com/KA-1205/MediaPlant.git
cd MediaPlant
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Supply images in `data/processed/<class_name>/`, with only class directories at that level. Set `num_classes` to the dataset's class count. Then, from the repository root:

```bash
python scripts/prepare_data.py --data_dir data/processed --output_dir data/splits
python scripts/train.py --model swin
python scripts/train.py --model convnext
python scripts/train_ensemble.py
python scripts/evaluate.py
streamlit run app/app.py
```

The preparation script creates stratified 70/15/15 splits with seed 42. Before claiming generalization, also check whether near-duplicate images or the same plant specimen cross those splits.

The app requires `best_swin.pth`, `best_convnext.pth`, `best_ensemble.pth`, and `weighted_avg_config.json` under `outputs/checkpoints/`, plus `data/splits/class_map.csv`. It stops with a missing-files message when these are absent. Use checkpoints you trust.

For the optional Gemini knowledge request, configure `GEMINI_API_KEY` locally using [.env.example](.env.example). The code imports the [Google Gen AI SDK](https://googleapis.github.io/python-genai/). Model availability, credentials, and provider charges are separate from local classification.

## Evaluation and open evidence

[scripts/evaluate.py](scripts/evaluate.py) compares individual models, simple averaging, weighted averaging, and learned stacking. It writes metrics and figures to `outputs/results/`. Treat these as outputs to reproduce, not results already supplied by this repository. Do not select a final method using the test set and then report that selection as an independent validation.

TODO(Kartik): add the dataset source and usage rights, exact environment, checkpoint download or reproduction details, raw held-out metrics, and a recorded demo.

## My role

TODO(Kartik): confirm your implementation responsibilities and collaborators, with links to the relevant commits. Repository ownership alone is not a contribution statement.
