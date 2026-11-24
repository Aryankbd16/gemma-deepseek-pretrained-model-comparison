# Pretrained Model Lab CA - G19 (Gemma) and G20 (DeepSeek-R1) + Comparative Study

| Group | Checkpoint | Why |
|---|---|---|
| G19 | `google/gemma-2b-it` (Gemma 1.0, 2B, instruction-tuned) | Original Gemma family, chat-capable, ~5 GB in bf16 so it fits a free Colab T4. Gated: accept the licence on Hugging Face and create an access token. |
| G20 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (exact, no substitution) | Public, MIT. 15.2 GB of weights, so the notebook defaults to 4-bit NF4 on a T4. |

> **HONEST STATUS - read first.** This package was authored in a sandbox with **no GPU and no access to the gated checkpoint**. The notebooks were compiled and executed end-to-end only against *stub* models (to test the code paths), and 67 evaluator unit tests pass. **No real model output or measurement exists in this package.** Every measured cell in the PDFs/XLSX says **NOT RUN** until you run the notebooks in Colab and rebuild the reports (steps below). Literature numbers are labelled as such and kept separate from ours.
> Also note: the faculty brief `Lab_CA_Pretrained_Models.docx.pdf` could not be re-verified in text form; the structure here follows the project execution prompt. Please compare against the brief before submission.

## What to run (about 1-2 hours on a T4)
1. Open `G19_Gemma/notebook/G19_Gemma_Implementation.ipynb` in Colab -> Runtime > GPU. Add your HF token (Colab secret `HF_TOKEN`) after accepting the Gemma licence. Run all. (There is a `QUICK_RUN` switch near the top.)
2. Open `G20_DeepSeek/notebook/G20_DeepSeek_R1_Implementation.ipynb`, run all (no token needed). Expect long generations (reasoning traces).
3. Each notebook writes `G19_results.json` / `G20_results.json`, prediction files, failure cases, resource results, charts and a zip. Download them into `G19_Gemma/results/` and `G20_DeepSeek/results/`.
4. Open `Comparative_Study/notebook/G19_G20_Comparison.ipynb` (upload both results files when prompted or place them as the first cell says). It writes `Experimental_Results.xlsx`, comparison CSVs and charts (copy charts into `Comparative_Study/charts/`).
5. Rebuild reports with real numbers: `pip install reportlab && python Shared/build_reports.py` (run from the package root). PDFs in `Reports/` and the per-folder copies are regenerated.

## Folder map
- `G19_Gemma/`, `G20_DeepSeek/`: `research/` (sources + research analysis PDF), `notebook/`, `dataset/` (30 samples JSON/CSV + card), `results/`, `documentation/` (implementation report PDF).
- `Comparative_Study/`: comparison notebook, three comparative PDFs, `results/Experimental_Results.xlsx` (template until you run), `charts/`.
- `Reports/`: all PDFs together, incl. `Paper_to_Implementation_Mapping.pdf`. `Viva/Viva_Preparation.pdf` (48 Q&A).
- `Shared/`: `ca_common.py` (evaluator + Colab helpers, embedded in the notebooks), `dataset_def.py`, report builder, notebook generator (`build_notebooks.py`), tests (`test_common.py`, `test_notebooks.py`).

## Dataset and method (summary)
30 hand-authored prompts (10 categories x 3): 20 dev, 10 held-out test (never used for tuning). Evaluation only - no training or fine-tuning of any model. Metrics: accuracy (rule-based), exact match, token F1, ROUGE, reasoning/math correctness, instruction compliance, structured validity, Wilson 95% CI. Robustness: 5 variations / 26 prompts. Failure analysis: only observed failures. Resource experiments: precision, input length, generation length (time, allocated/reserved/peak memory with `cuda.synchronize`).

## Fairness caveats (also in the reports)
2B vs 7B, bf16/fp16 vs 4-bit, greedy vs sampling (T=0.6, top-p 0.95 per DeepSeek README), 512 vs 1536 token caps, different specialisation. No universal winner is claimed.

## Verified facts and sources
Gemma paper arXiv:2403.08295; HF card google/gemma-2b-it; github.com/google-deepmind/gemma (Gemma-1 code at commit 59ffafe; main covers newer models); DeepSeek-R1 paper arXiv:2501.12948; github.com/deepseek-ai/DeepSeek-R1 (README only, no code); HF listing of DeepSeek-R1-Distill-Qwen-7B. Key distinction: the distilled checkpoint was trained with SFT on ~800k R1 samples, **not** RL.

## Model weights
Not included (multi-GB). They download automatically from Hugging Face in the notebooks.
