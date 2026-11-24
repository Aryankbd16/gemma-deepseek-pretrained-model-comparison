"""Generates the three Colab notebooks (plain nbformat-4 JSON; nbformat package not required)."""
import json, textwrap, uuid, os
from dataset_def import dataset_json

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON_SRC = open(os.path.join(HERE, "ca_common.py"), encoding="utf-8").read()
DATASET_SRC = dataset_json()


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": textwrap.dedent(text).strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip("\n").splitlines(True)}


def notebook(cells, title):
    return {"cells": cells, "metadata": {
        "colab": {"provenance": [], "gpuType": "T4", "name": title},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
        "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}


def raw_file_cell(filename, content):
    """A code cell that writes `content` verbatim to `filename` (no %%writefile quoting problems)."""
    marker = "EOF_" + uuid.uuid4().hex[:6].upper()
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {"cellView": "form"}, "execution_count": None, "outputs": [],
            "source": ("#@title Write %s (embedded copy - run this cell, no need to read it)\n" % filename).splitlines(True)
            + ("_content = r'''" + content.replace("'''", "\\'\\'\\'") + "'''\n").splitlines(True)
            + ["open(%r, 'w', encoding='utf-8').write(_content)\n" % filename, "print('wrote', %r, len(_content), 'chars')\n" % filename]}


# ---------------------------------------------------------------------------------------------
# Individual model notebook
# ---------------------------------------------------------------------------------------------
def model_notebook(cfg):
    K = cfg["key"]
    c = []
    c.append(md(cfg["intro_md"]))
    c.append(md("""
    ## How to run (Google Colab)
    1. **Runtime -> Change runtime type -> T4 GPU** (or better). 2. Run all cells top-to-bottom (`Runtime -> Run all`).
    3. %s
    4. At the end the notebook saves `%s_results.json`, prediction files, CSVs and charts into `/content/Pretrained_Model_CA/%s/results/`
       and downloads a ZIP. Put those files into the matching `results/` folder of the project; the comparison notebook reads them.

    > **Honesty note.** Nothing in this notebook is pre-filled. Every number is produced by *your* run. Where an experiment cannot run on your hardware
    > the notebook records `SKIPPED`/`FAILED` with the reason instead of inventing a value.
    > These are **pretrained** models: no weights are trained here. The dataset is an *evaluation* set only.
    """ % (cfg["auth_step"], K, cfg["folder"])))
    # 2 installation
    c.append(md("## 1. Installation and dependency setup"))
    c.append(code("""
    # Colab already ships torch / numpy / pandas / matplotlib. We only add / upgrade what we need.
    # If pip reports a dependency conflict with a Colab preinstalled package, it is usually harmless; if imports fail, Runtime -> Restart session and re-run.
    !pip install -q -U transformers accelerate bitsandbytes huggingface_hub psutil packaging
    """))
    c.append(raw_file_cell("ca_common.py", COMMON_SRC))
    c.append(raw_file_cell("common_eval_dataset.json", DATASET_SRC))
    # 3 versions + hardware
    c.append(md("## 2. Environment, GPU and RAM detection (Python / PyTorch / Transformers / CUDA)"))
    c.append(code("""
    import os, json, time, shutil, importlib, statistics, platform
    import ca_common as ca
    import torch
    ENV = ca.env_report()
    for k, v in ENV.items():
        print(f"{k:>18}: {v}")
    ALLOW_CPU = False      # set True only for a *very* slow smoke test; results would not be comparable
    if not torch.cuda.is_available() and not ALLOW_CPU:
        raise SystemExit("No GPU detected. Runtime -> Change runtime type -> T4 GPU, then re-run. (Nothing was fabricated.)")
    """))
    # 4 config
    c.append(md("## 3. Configuration"))
    c.append(code(cfg["config_code"]))
    # 5 auth
    c.append(md(cfg["auth_md"]))
    c.append(code(cfg["auth_code"]))
    # 6 load
    c.append(md("## 5. Exact checkpoint loading + tokenizer initialisation (timed)\n" + cfg["load_md"]))
    c.append(code(cfg["precision_code"] + """

    tried, model, tok = [], None, None
    chain = [PRIMARY_PRECISION] + [p for p in FALLBACKS if p != PRIMARY_PRECISION]
    for p in chain:
        try:
            model, tok, LOAD_INFO = ca.load_hf_model(MODEL_ID, p, token=HF_TOKEN)
            PRECISION = p
            break
        except torch.cuda.OutOfMemoryError:
            tried.append((p, "CUDA out of memory")); print(f"[{p}] out of memory -> trying next fallback (same checkpoint, lower precision)")
            ca.release()
        except Exception as e:
            tried.append((p, repr(e)[:300])); print(f"[{p}] failed: {e!r}")
            ca.release()
            if any(t in repr(e) for t in ("Gated", "401", "403", "RepositoryNotFound")):
                break   # auth problem: lower precision will not help
    if model is None:
        raise RuntimeError(f"Could not load {MODEL_ID}. Attempts: {tried}. The assigned checkpoint is never replaced by another model.")
    print("\\nLoaded", MODEL_ID, "as", PRECISION, "| attempts before success:", tried)
    print("Parameters loaded:", f"{LOAD_INFO['n_parameters_loaded']/1e9:.3f} B", "| max_position_embeddings:", LOAD_INFO["max_position_embeddings"], "| vocab:", LOAD_INFO["vocab_size"])
    print(json.dumps({k: v for k, v in LOAD_INFO.items() if k not in ("model_id",)}, indent=1, default=str))
    """))
    # 7 tokenizer / prompt formatting
    c.append(md("## 6. Tokenizer and prompt formatting\n" + cfg["prompt_md"]))
    c.append(code(cfg["tokenizer_code"]))
    # 8 inference
    c.append(md("## 7. Inference implementation (reusable, configurable generation parameters)\n" + cfg["infer_md"]))
    c.append(code("""
    gen = ca.make_generate_fn(model, tok, GEN_KW, force_think_prefix=FORCE_THINK_PREFIX)
    ca.warmup(gen)            # first CUDA call includes kernel compilation; excluded from all timings
    demo = gen("Explain in two sentences what a transformer language model is.")
    print(demo["text"]); print({k: demo[k] for k in ("input_tokens", "new_tokens", "seconds", "hit_max_tokens")}); print(demo["mem"])
    """))
    c.append(md("### Interactive prompt testing\nEdit `MY_PROMPTS` or set `INTERACTIVE = True` to type prompts in a loop (blank line to stop). Override generation settings per call, e.g. `gen(p, temperature=0.2, max_new_tokens=100)`."))
    c.append(code(cfg["interactive_code"]))
    # 9 dataset
    c.append(md("""
    ## 8. Common evaluation dataset (30 samples, 20 dev / 10 held-out test)
    Identical for G19 and G20. Written by us for this assignment. **Evaluation only** - the pretrained weights are never trained on it.
    The dev split was used to verify prompts, answer extraction and evaluators; the **10 held-out test prompts are run once and never used to tune anything**.
    """))
    c.append(code("""
    import pandas as pd
    DS = ca.load_json("common_eval_dataset.json")
    ALL = DS["samples"]; ROB = DS["robustness"]
    ET = {s["id"]: s["eval_type"] for s in ALL + ROB}
    DEV = [s for s in ALL if s["split"] == "dev"]; TEST = [s for s in ALL if s["split"] == "test"]
    if QUICK_RUN:
        DEV, TEST, ROB = DEV[:3], TEST[:2], ROB[:4]; print("QUICK_RUN: reduced sample sets -> results are NOT comparable and are flagged in the JSON")
    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(CHART_DIR, exist_ok=True)
    pd.set_option("display.max_colwidth", 70, "display.width", 200)
    pd.DataFrame(ALL)[["id", "split", "category", "difficulty", "eval_type", "criteria"]]
    """))
    # 10 dev run
    c.append(md("## 9. Predefined experiments - development split (20)"))
    c.append(code("""
    dev_recs = ca.run_samples(DEV, gen, expects_think=EXPECTS_THINK, tag="dev")
    dev_metrics = ca.aggregate(dev_recs)
    ca.save_json(f"{RESULTS_DIR}/{MODEL_KEY}_predictions_dev.json", dev_recs)
    pd.DataFrame([{ "id": r["id"], "cat": r["category"], "pass": r["eval"]["correct"], "sec": round(r["seconds"], 2), "new_tok": r["new_tokens"], "extracted": r["eval"]["extracted"][:50]} for r in dev_recs])
    """))
    c.append(md("## 10. Held-out test split (10) - run once, identical prompts for both models"))
    c.append(code("""
    test_recs = ca.run_samples(TEST, gen, expects_think=EXPECTS_THINK, tag="test")
    test_metrics = ca.aggregate(test_recs)
    ca.save_json(f"{RESULTS_DIR}/{MODEL_KEY}_predictions_test.json", test_recs)
    pd.DataFrame(test_recs)[["id", "category", "seconds", "new_tokens", "tokens_per_sec", "hit_max_tokens"]].assign(passed=[r["eval"]["correct"] for r in test_recs])
    """))
    # 11 metrics
    c.append(md("""
    ## 11. Evaluation metrics (what they measure, and their limits)
    | Metric | Applied to | Measures | Limitation |
    |---|---|---|---|
    | Accuracy | all samples (task-specific automatic checker) | fraction judged correct | checkers are rule-based; unusual but valid phrasings can be marked wrong |
    | Exact Match (EM) | short-answer items | normalised answer equals reference | very strict; verbose answers score 0 |
    | Token F1 | short-answer items | word overlap with reference | rewards overlap, not truth |
    | ROUGE-1/2/L | summarisation | n-gram/LCS overlap with a single reference | one reference, no semantics (own implementation, no stemming) |
    | Reasoning correctness | items flagged `reasoning` (logic, maths, multi-step, analytical, trick questions) | final conclusion correct | does not grade the reasoning chain itself |
    | Mathematical answer correctness | numeric items | extracted number within tolerance | extractor uses `Final answer:` > `\\\\boxed{}` > last number |
    | Instruction compliance | IF, summary length, structured-output cleanliness | explicit constraints satisfied | only constraints we can check mechanically |
    | Structured-output validity | JSON / table items | parses as the requested format | validity != correctness (checked separately) |
    | Response relevance | **not applied** | - | no reliable automatic metric; use the rubric columns for manual review |
    The 10-sample test split gives very wide confidence intervals (see `accuracy_ci95`); differences of 1-2 items are **not** statistically meaningful.
    """))
    c.append(code("""
    def show(m, title):
        rows = {k: m[k] for k in ("n", "n_correct", "accuracy", "exact_match", "token_f1", "reasoning_correctness", "instruction_compliance",
                                  "structured_validity", "rouge1", "rouge2", "rougeL", "time_mean_s", "time_median_s", "tokens_per_sec_aggregate",
                                  "peak_gpu_mem_gb_max", "n_hit_max_tokens", "n_unfinished_thinking")}
        rows["accuracy_ci95"] = [round(x, 3) for x in m["accuracy_ci95"]]
        print(title); display(pd.Series(rows, name=title).to_frame())
    show(dev_metrics, "DEV (n=%d)" % dev_metrics["n"]); show(test_metrics, "HELD-OUT TEST (n=%d)" % test_metrics["n"])
    pd.DataFrame({"dev": {k: v["accuracy"] for k, v in dev_metrics["per_category"].items()}, "test": {k: v["accuracy"] for k, v in test_metrics["per_category"].items()}})
    """))
    c.append(code("""
    import matplotlib; import matplotlib.pyplot as plt
    both = dev_recs + test_recs
    cats = sorted({r["category"] for r in both})
    acc = [sum(r["eval"]["correct"] for r in both if r["category"] == c_) / sum(1 for r in both if r["category"] == c_) for c_ in cats]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].barh(cats, acc, color="#3b6ea5"); ax[0].set_xlim(0, 1); ax[0].set_title(f"{MODEL_KEY}: accuracy by category (dev+test, 3 items each)"); ax[0].set_xlabel("accuracy")
    ax[1].bar([r["id"] for r in test_recs], [r["seconds"] for r in test_recs], color="#a55b3b"); ax[1].set_title(f"{MODEL_KEY}: inference time per held-out test sample"); ax[1].set_ylabel("seconds"); ax[1].tick_params(axis="x", rotation=60)
    plt.tight_layout(); plt.savefig(f"{CHART_DIR}/{MODEL_KEY}_category_and_time.png", dpi=150); plt.show()
    """))
    # robustness
    c.append(md("""
    ## 12. Input-variation and robustness experiments (5 variations)
    V1 prompt wording - V2 increasing input complexity - V3 ambiguous instructions - V4 increasing mathematical complexity - V5 different structured-output formats.
    For every item the record stores the input, expected behaviour, actual output, evaluation result, inference time and an automatic observation.
    """))
    c.append(code("""
    rob_recs = ca.run_samples(ROB, gen, expects_think=EXPECTS_THINK, tag="robust")
    ca.save_json(f"{RESULTS_DIR}/{MODEL_KEY}_robustness.json", rob_recs)
    by_var = {}
    for r in rob_recs:
        d = by_var.setdefault(r["variation"], {"n": 0, "correct": 0, "sec": []}); d["n"] += 1; d["correct"] += int(r["eval"]["correct"]); d["sec"].append(r["seconds"])
    ROB_SUMMARY = {k: {"n": v["n"], "correct": v["correct"], "accuracy": v["correct"] / v["n"], "mean_seconds": sum(v["sec"]) / len(v["sec"])} for k, v in by_var.items()}
    display(pd.DataFrame(ROB_SUMMARY).T)
    pd.DataFrame([{"id": r["id"], "variation": r["variation"], "pass": r["eval"]["correct"], "sec": round(r["seconds"], 1), "observation": r["observation"][:90]} for r in rob_recs])
    """))
    c.append(code("""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(list(ROB_SUMMARY), [v["accuracy"] for v in ROB_SUMMARY.values()], color="#5b8c5a"); ax.set_ylim(0, 1); ax.set_ylabel("accuracy"); ax.set_title(f"{MODEL_KEY}: robustness accuracy by variation"); ax.tick_params(axis="x", rotation=25)
    plt.tight_layout(); plt.savefig(f"{CHART_DIR}/{MODEL_KEY}_robustness.png", dpi=150); plt.show()
    """))
    # consistency
    c.append(md("### Consistency check (same prompt, 3 seeds)\n" + cfg["consistency_md"]))
    c.append(code("""
    CONS = [ca.consistency_test(gen, s, seeds=(0, 1, 2), expects_think=EXPECTS_THINK) for s in DEV if s["id"] in ("MR01", "LR02", "MS02")]
    ca.save_json(f"{RESULTS_DIR}/{MODEL_KEY}_consistency.json", CONS)
    for r in CONS: print(r["id"], "distinct answers:", r["n_distinct_answers"], "| all correct:", r["all_correct"], [o["extracted"] for o in r["runs"]])
    """))
    # failure
    c.append(md("""
    ## 13. Failure and limitation analysis
    **Part A - OBSERVED failures** (automatically collected from this run; nothing is invented. If fewer than five failures occurred, the table is simply shorter.)
    **Part B - THEORETICAL limitations** (from the paper/model card; *not* observed in this run) is a separate list below.
    """))
    c.append(code("""
    FAILS = ca.build_failure_table([(r, ET[r["id"]]) for r in dev_recs + test_recs + rob_recs])
    # context-length limitation evidence is appended after the resource experiments (section 14)
    print(len(FAILS), "observed failures so far")
    fdf = pd.DataFrame(FAILS)
    fdf[["id", "category", "failure_type", "input", "actual_output"]].head(40) if len(FAILS) else print("No failures observed in dev/test/robustness runs.")
    """))
    c.append(md(cfg["theoretical_md"]))
    # resource
    c.append(md("## 14. Resource experiments (same checkpoint, controlled variables)\n" + cfg["resource_md"]))
    c.append(code(cfg["resource_code"]))
    c.append(code("""
    # Append observed context-length failures to the failure table
    for row in RES_INPUT:
        if row.get("status") == "OK" and not row["found_needle"]:
            FAILS.append({"id": "LEN%d" % row["target_tokens"], "category": "context_length", "input": f"~{row['actual_input_tokens']}-token log with one secret code in the middle",
                          "expected_behaviour": "retrieve the code K7-4921", "actual_output": row["answer_head"], "failure_type": "Context-length limitation (needle not retrieved)",
                          "explanation": "Observed in the input-length experiment at the stated prompt length.", "observed": True})
    ca.save_json(f"{RESULTS_DIR}/{MODEL_KEY}_failure_cases.json", FAILS)
    print(len(FAILS), "observed failure cases saved")
    """))
    c.append(code("""
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ok = [r for r in RES_PRECISION if r["status"] == "OK"]
    if ok:
        ax[0].bar([r["precision"] for r in ok], [r["load"]["model_load_s"] for r in ok], color="#3b6ea5"); ax[0].set_title("load time (s, cached weights)")
        ax[1].bar([r["precision"] for r in ok], [r["peak_mem_gb"] or 0 for r in ok], color="#a55b3b"); ax[1].set_title("peak GPU memory (GB)")
        ax[2].bar([r["precision"] for r in ok], [r["metrics"]["accuracy"] for r in ok], color="#5b8c5a"); ax[2].set_ylim(0, 1); ax[2].set_title("accuracy on resource subset")
    plt.tight_layout(); plt.savefig(f"{CHART_DIR}/{MODEL_KEY}_resource_precision.png", dpi=150); plt.show()
    il = [r for r in RES_INPUT if r.get("status") == "OK"]
    if il:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot([r["actual_input_tokens"] for r in il], [r["seconds"] for r in il], "o-"); ax[0].set_xlabel("input tokens"); ax[0].set_ylabel("seconds"); ax[0].set_title("latency vs input length")
        ax[1].plot([r["actual_input_tokens"] for r in il], [r["peak_mem_gb"] for r in il], "o-", color="#a55b3b"); ax[1].set_xlabel("input tokens"); ax[1].set_ylabel("peak GB"); ax[1].set_title("peak memory vs input length")
        plt.tight_layout(); plt.savefig(f"{CHART_DIR}/{MODEL_KEY}_resource_input_length.png", dpi=150); plt.show()
    """))
    # save
    c.append(md("## 15. Save everything (consumed by `G19_G20_Comparison.ipynb`)"))
    c.append(code("""
    import datetime
    RESULTS = {
        "meta": {"model_key": MODEL_KEY, "checkpoint": MODEL_ID, "display_name": PRIMARY_NAME, "precision": PRECISION, "quick_run": QUICK_RUN,
                 "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "env": ENV, "generation_config": GEN_KW, "expects_think": EXPECTS_THINK,
                 "force_think_prefix": FORCE_THINK_PREFIX, "load": LOAD_INFO, "fallback_attempts": tried, "dataset_version": DS["version"],
                 "n_dev": len(DEV), "n_test": len(TEST), "n_robustness": len(ROB)},
        "dev": {"metrics": dev_metrics, "records": dev_recs}, "test": {"metrics": test_metrics, "records": test_recs},
        "robustness": {"summary": ROB_SUMMARY, "records": rob_recs}, "consistency": CONS, "failures": FAILS,
        "resource": {"precision": RES_PRECISION, "input_length": RES_INPUT, "generation_length": RES_GEN},
    }
    ca.save_json(f"{RESULTS_DIR}/{MODEL_KEY}_results.json", RESULTS)
    for name, recs in (("dev", dev_recs), ("test", test_recs), ("robustness", rob_recs)):
        pd.DataFrame([{**{k: r[k] for k in ("id", "category", "prompt", "final_output", "seconds", "new_tokens", "tokens_per_sec", "hit_max_tokens")},
                       "correct": r["eval"]["correct"], "extracted": r["eval"]["extracted"], "observation": r["observation"]} for r in recs]).to_csv(f"{RESULTS_DIR}/{MODEL_KEY}_{name}_predictions.csv", index=False)
    with open(f"{RESULTS_DIR}/{MODEL_KEY}_execution_log.txt", "w") as f:
        f.write(json.dumps(RESULTS["meta"], indent=1, default=str))
    print("saved ->", sorted(os.listdir(RESULTS_DIR)))
    ZIP = shutil.make_archive(f"/content/{MODEL_KEY}_results", "zip", RESULTS_DIR)
    try:
        from google.colab import files; files.download(ZIP)
    except Exception as e:
        print("download manually:", ZIP)
    """))
    c.append(md("## 16. Final observations and conclusion"))
    c.append(code("""
    from IPython.display import Markdown
    t = test_metrics; d = dev_metrics
    Markdown(f\"\"\"**Auto-generated summary of THIS run (verify before quoting):**

    - Checkpoint `{MODEL_ID}` loaded at **{PRECISION}** on **{ENV.get('gpu')}**; load time {LOAD_INFO['model_load_s']:.1f}s (+{LOAD_INFO['download_s']:.1f}s download).
    - Dev accuracy {d['accuracy']:.2f} ({d['n_correct']}/{d['n']}); held-out test accuracy {t['accuracy']:.2f} ({t['n_correct']}/{t['n']}), 95% Wilson CI {t['accuracy_ci95'][0]:.2f}-{t['accuracy_ci95'][1]:.2f}.
    - Mean / median test latency {t['time_mean_s']:.1f}s / {t['time_median_s']:.1f}s; aggregate throughput {t['tokens_per_sec_aggregate']:.1f} tokens/s; peak GPU memory {t['peak_gpu_mem_gb_max']} GB.
    - Responses cut off by the token limit: dev {d['n_hit_max_tokens']}, test {t['n_hit_max_tokens']}; unfinished reasoning traces: dev {d['n_unfinished_thinking']}, test {t['n_unfinished_thinking']}.
    - Observed failures logged: {len(FAILS)} (see `{MODEL_KEY}_failure_cases.json`).
    \"\"\")
    """))
    c.append(md(cfg["conclusion_md"]))
    return notebook(c, cfg["title"])


# ----------------------------- model-specific configs -------------------------------------------
G19 = {
    "key": "G19", "folder": "G19_Gemma", "title": "G19_Gemma_Implementation.ipynb",
    "auth_step": "Add a Colab secret named `HF_TOKEN` (key icon in the left bar, enable notebook access) **after** accepting the Gemma licence on Hugging Face for `google/gemma-2b-it` (see section 4). Without it the notebook asks you to paste a token.",
    "intro_md": """
    # G19 - Gemma (original family): instruction-tuned text checkpoint `google/gemma-2b-it`
    **M.Tech Laboratory CA - Pretrained Models.**  Paper: *Gemma: Open Models Based on Gemini Research and Technology* (Gemma Team, Google DeepMind, 2024) - https://arxiv.org/abs/2403.08295 ·
    Official repo: https://github.com/google-deepmind/gemma · Model card: https://huggingface.co/google/gemma-2b-it

    **Checkpoint selection (the faculty sheet does not name one).** The original Gemma 1 family has 2B and 7B sizes, each as pretrained and instruction-tuned (`-it`) checkpoints.
    We use **`google/gemma-2b-it`** (Gemma 1.0, 2B, instruction-tuned): it is an official checkpoint of the *original* family named in the paper, fits a free Colab T4 in half precision, and is the checkpoint
    whose model card we verified (licence `gemma`, gated: licence acceptance + HF token required, trained context 8192 tokens, `<start_of_turn>`/`<end_of_turn>` chat format).
    Gemma 2 / Gemma 3 are **later generations and are deliberately not used**. `google/gemma-1.1-2b-it` is a newer revision of the same family (the v4 paper's tables report 1.1 IT results); we did not verify its card in this session, so we did not choose it.

    | Item | Value (source) |
    |---|---|
    | Parameters | ~2.5B: 524,550,144 embedding + 1,981,884,416 non-embedding (paper Table 2) |
    | Architecture | decoder-only Transformer; 18 layers, d_model 2048, FFN hidden 32768, 8 heads, **1 KV head (multi-query attention)**, head size 256, vocab 256,128, RoPE, GeGLU, RMSNorm, tied embeddings (paper Table 1 + text) |
    | Context | trained on 8,192 tokens (paper) |
    | Tokenizer | SentencePiece subset of Gemini tokenizer, 256k vocab, splits digits, byte fallback (paper) |
    | Alignment | SFT + RLHF; control tokens `<start_of_turn>user/model`, `<end_of_turn>` (paper Tables 3-4) |
    | Licence | Gemma Terms of Use (custom, gated) - model card `License: gemma` |
    """,
    "config_code": """
    MODEL_KEY, MODEL_ID, PRIMARY_NAME = "G19", "google/gemma-2b-it", "Gemma 1.0 2B-IT"
    EXPECTS_THINK = False          # Gemma does not emit a reasoning trace
    FORCE_THINK_PREFIX = False
    QUICK_RUN = False              # True -> tiny smoke test (results flagged as NOT comparable)
    MAX_NEW_TOKENS = 512
    # Gemma card default is greedy decoding; we keep greedy for determinism (documented fairness difference vs. G20's sampling)
    GEN_KW = dict(do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    BASE_DIR = "/content/Pretrained_Model_CA"
    RESULTS_DIR = f"{BASE_DIR}/G19_Gemma/results"; CHART_DIR = RESULTS_DIR + "/charts"
    """,
    "auth_md": """
    ## 4. Model authentication (gated checkpoint)
    1. Create / log in to a Hugging Face account. 2. Open https://huggingface.co/google/gemma-2b-it and **accept the Gemma licence** (approval is immediate). 3. Create a *read* token at
    https://huggingface.co/settings/tokens and store it as the Colab secret `HF_TOKEN` (or paste it when asked).
    """,
    "auth_code": """
    from huggingface_hub import login, hf_hub_download
    HF_TOKEN = None
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass
    HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        import getpass; HF_TOKEN = getpass.getpass("Paste your Hugging Face READ token: ")
    login(token=HF_TOKEN, add_to_git_credential=False)
    try:
        hf_hub_download(MODEL_ID, "config.json", token=HF_TOKEN); print("Access to", MODEL_ID, "confirmed.")
    except Exception as e:
        name = type(e).__name__
        if "Gated" in name or "401" in str(e) or "403" in str(e):
            raise SystemExit(f"Not authorised for {MODEL_ID} ({name}). Accept the licence at https://huggingface.co/{MODEL_ID} with the SAME account as your token, then re-run.")
        raise
    """,
    "load_md": "Precision: bfloat16 is the checkpoint's native dtype (model card). T4 GPUs have no native bf16, so we use fp16 there and record the choice. If the GPU runs out of memory the loader falls back to 8-bit then 4-bit (still the same `gemma-2b-it` weights) and records it.",
    "precision_code": """
    PRIMARY_PRECISION = "bf16" if ENV.get("bf16_native") else "fp16"
    FALLBACKS = ["8bit", "4bit"]
    print("primary precision:", PRIMARY_PRECISION, "| native bf16:", ENV.get("bf16_native"))
    """,
    "prompt_md": "Gemma-IT must be prompted with its control tokens (paper, 'Formatting'): *without the formatter the input is out-of-distribution*. We use the tokenizer's built-in chat template (single user turn; Gemma has no system role) and verify it equals the documented manual format.",
    "tokenizer_code": """
    msgs = [{"role": "user", "content": "Write a hello world program"}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    manual = "<bos><start_of_turn>user\\nWrite a hello world program<end_of_turn>\\n<start_of_turn>model\\n"
    print(repr(rendered)); print("template == documented manual format:", rendered == manual)
    ids = tok(rendered, add_special_tokens=False)["input_ids"]       # the template already contains <bos>
    print(len(ids), "tokens:", tok.convert_ids_to_tokens(ids))
    print("vocab size:", len(tok), "| bos/eos/pad:", tok.bos_token, tok.eos_token, tok.pad_token, "| special:", tok.all_special_tokens[:8])
    print("digit splitting demo:", tok.tokenize("12345 + 6789"))
    print("eos ids used by generate():", model.generation_config.eos_token_id)
    print(model.config)   # layer count, hidden/intermediate sizes, heads, kv heads (compare with paper Table 1)
    """,
    "infer_md": "`ca.make_generate_fn` renders the chat template, tokenises (`add_special_tokens=False`), calls `model.generate` with `torch.cuda.synchronize()` before/after the timer, decodes only the newly generated tokens, strips control tokens and returns text, token counts, seconds, a *hit-max-tokens* flag and allocated / reserved / peak memory.",
    "interactive_code": """
    MY_PROMPTS = ["Give me three tips for learning linear algebra.", "Translate 'good morning' into French."]
    INTERACTIVE = False
    for p in MY_PROMPTS:
        g = gen(p); print(">>>", p); print(g["text"]); print(f"[{g['new_tokens']} tokens, {g['seconds']:.1f}s]\\n")
    while INTERACTIVE:
        p = input("prompt> ").strip()
        if not p: break
        g = gen(p); print(g["text"], f"\\n[{g['new_tokens']} tokens, {g['seconds']:.1f}s]")
    """,
    "consistency_md": "Gemma runs with greedy decoding, so identical answers are *expected*; this check mainly verifies determinism of our pipeline (it is not evidence about sampling variability).",
    "theoretical_md": """
    **Part B - THEORETICAL limitations (not observed in this run; from the paper / model card)**
    - Small scale: the 2B model is far behind the 7B on reasoning benchmarks (paper Table 6, pretrained: GSM8K 17.7 vs 46.4, MATH 11.8 vs 24.3).
    - Trained on primarily-English data; not optimised for multilingual tasks (paper).
    - Factual accuracy: "not knowledge bases" - may produce incorrect or outdated statements; common-sense and nuance gaps (model card).
    - Trained context 8,192 tokens; long-context behaviour beyond it is unsupported.
    - Safety: bias, toxic-language and PII-leakage risks acknowledged; users must run use-case-specific safety testing (paper).
    """,
    "resource_md": "(a) **input length** and (b) **generation length** on the primary model, then (c) **precision variants** (bf16/fp16 vs 8-bit vs 4-bit) reloading the *same* `gemma-2b-it` weights on a 5-prompt dev subset. Variables not under test are held fixed.",
    "resource_code": """
    MAX_CTX = 8192   # trained context length stated in the Gemma paper
    LENGTHS = [256, 1024, 2048, 4096, 6000]
    RES_INPUT = ca.resource_input_length_experiment(model, tok, gen, LENGTHS, max_ctx=MAX_CTX, max_new_tokens=32, expects_think=False)
    RES_GEN = ca.resource_generation_length_experiment(gen, [64, 256, 512])
    SUBSET_IDS = ["GK02", "MR01", "MS02", "LR02", "JS01"]
    SUBSET = [s for s in DEV if s["id"] in SUBSET_IDS] if not QUICK_RUN else DEV[:2]
    ca.release(model); model = None
    precs = ["bf16", "fp16", "8bit", "4bit"]
    RES_PRECISION = ca.resource_precision_experiment(MODEL_ID, HF_TOKEN, precs, SUBSET, GEN_KW, expects_think=False, force_think_prefix=False)
    ca.save_json(f"{RESULTS_DIR}/{MODEL_KEY}_resource.json", {"input_length": RES_INPUT, "generation_length": RES_GEN, "precision": RES_PRECISION})
    display(pd.DataFrame(RES_INPUT)); display(pd.DataFrame(RES_GEN))
    pd.DataFrame([{"precision": r["precision"], "status": r["status"][:60], **({"load_s": r["load"]["model_load_s"], "peak_gb": r["peak_mem_gb"], "acc": r["metrics"]["accuracy"], "mean_s": r["metrics"]["time_mean_s"], "tok_per_s": r["metrics"]["tokens_per_sec_aggregate"]} if r["status"] == "OK" else {})} for r in RES_PRECISION])
    """,
    "conclusion_md": """
    ### Observations (fill in after your run - these are *your* findings, not pre-written)
    - Which categories did Gemma-2B-IT handle best / worst? Compare with the 3-items-per-category table above.
    - Did it obey format constraints (IF, JSON, table) or add prose? See `compliance` and `structured_validity`.
    - Did bf16/fp16 vs 8-bit vs 4-bit change accuracy or output text for the 5-prompt subset? What was the memory / speed trade-off?
    - At what prompt length (if any) did needle retrieval fail?

    ### Conclusion
    *(Write 5-8 sentences from the tables above. Do not generalise beyond n=30 items; report limitations of the evaluation.)*
    """,
}

G20 = {
    "key": "G20", "folder": "G20_DeepSeek", "title": "G20_DeepSeek_R1_Implementation.ipynb",
    "auth_step": "No Hugging Face token is required for `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (public, MIT). A token, if provided as secret `HF_TOKEN`, only raises download rate limits.",
    "intro_md": """
    # G20 - DeepSeek-R1-Distill-Qwen-7B (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`)
    **M.Tech Laboratory CA - Pretrained Models.**  Paper: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (DeepSeek-AI, 2025) - https://arxiv.org/abs/2501.12948 ·
    Official repo: https://github.com/deepseek-ai/DeepSeek-R1 · Model card: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

    **Important distinction.** The paper's RL pipeline (R1-Zero: pure RL with GRPO; R1: cold-start SFT -> reasoning RL -> rejection-sampling SFT -> all-scenario RL) produced the 671B-parameter MoE **DeepSeek-R1**.
    The checkpoint used here is a **dense 7B model obtained by supervised fine-tuning (distillation only, no RL stage)** of the base model **Qwen2.5-Math-7B** on ~800k samples generated by DeepSeek-R1
    (paper §2.4; repo README table). It did **not** itself undergo the R1 RL training.

    | Item | Value (source) |
    |---|---|
    | Exact checkpoint | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (never replaced) |
    | Base model | Qwen2.5-Math-7B (repo README) |
    | Parameters | 7B class (HF reports ~7.6B; counted at load time below); weights 8.61 GB + 6.62 GB safetensors (HF file listing) |
    | Licence | MIT (repo + weights); derived from Qwen2.5 (Apache 2.0) |
    | Context | read from `config.json` at load time (printed below); evaluation in the paper used up to 32,768 generated tokens |
    | Recommended use (repo README) | temperature 0.5-0.7 (0.6), **no system prompt**, put all instructions in the user turn, ask to reason step by step, **force the response to start with `<think>\\n`** |
    | Paper-reported (NOT ours) | AIME24 pass@1 55.5, MATH-500 92.8, GPQA-Diamond 49.1, LiveCodeBench 37.6, Codeforces 1189 (Table 5; sampling T=0.6, top-p 0.95, up to 64 samples/question, 32k tokens) |
    """,
    "config_code": """
    MODEL_KEY, MODEL_ID, PRIMARY_NAME = "G20", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B"
    EXPECTS_THINK = True           # generated text = reasoning + '</think>' + answer
    FORCE_THINK_PREFIX = True      # README: force the response to start with '<think>\\n' (applied only if the template does not already do it)
    QUICK_RUN = False
    # Paper evaluates with up to 32,768 new tokens. On a Colab T4 that is infeasible (hours/sample), so we cap generation and REPORT how many answers hit the cap.
    MAX_NEW_TOKENS = 1536
    GEN_KW = dict(do_sample=True, temperature=0.6, top_p=0.95, max_new_tokens=MAX_NEW_TOKENS)   # README-recommended sampling
    BASE_DIR = "/content/Pretrained_Model_CA"
    RESULTS_DIR = f"{BASE_DIR}/G20_DeepSeek/results"; CHART_DIR = RESULTS_DIR + "/charts"
    """,
    "auth_md": "## 4. Model authentication\nThe checkpoint is public (MIT) - no licence gate. An optional token avoids anonymous rate limits.",
    "auth_code": """
    from huggingface_hub import login
    HF_TOKEN = None
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass
    HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN") or None
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False); print("Logged in with provided token.")
    else:
        print("No token provided - using anonymous access (fine for this public model).")
    """,
    "load_md": "A 7B model needs ~15.2 GB in fp16, which does **not** fit a 15 GB T4 together with activations. Policy: if the GPU has >= 22 GB use bf16/fp16 else use **4-bit NF4 (bitsandbytes, fp16 compute)**; if that fails there is no smaller model substitution - the run stops. The precision used is stored in the results and listed as a fairness difference in the comparison.",
    "precision_code": """
    gpu_gb = ENV.get("gpu_total_gb") or 0
    PRIMARY_PRECISION = ("bf16" if ENV.get("bf16_native") else "fp16") if gpu_gb >= 22 else "4bit"
    FALLBACKS = ["8bit", "4bit"] if PRIMARY_PRECISION in ("bf16", "fp16") else []
    print("GPU memory (GB):", gpu_gb, "-> primary precision:", PRIMARY_PRECISION)
    """,
    "prompt_md": "Qwen-style ChatML template from the checkpoint's tokenizer; per the repo README there is **no system prompt**, and the response should start with `<think>\\n`. The cell shows what the template actually produces and whether we add the prefix.",
    "tokenizer_code": """
    msgs = [{"role": "user", "content": "What is 15% of 240? Please reason step by step."}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(repr(rendered)); print("template already ends with '<think>':", rendered.rstrip().endswith("<think>"), "-> forced prefix added by us:", FORCE_THINK_PREFIX and not rendered.rstrip().endswith("<think>"))
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    print(len(ids), "tokens:", tok.convert_ids_to_tokens(ids)[:40], "...")
    print("vocab size:", len(tok), "| bos/eos/pad:", tok.bos_token, tok.eos_token, tok.pad_token)
    print("'<think>' / '</think>' token ids:", tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>"))
    print("digit tokenisation demo:", tok.tokenize("12345 + 6789"))
    print("model generation_config:", model.generation_config)
    print(model.config)   # layer count, hidden/intermediate sizes, heads, kv heads, max_position_embeddings
    """,
    "infer_md": "Same reusable function as G19 (`ca.make_generate_fn`) with README sampling settings. The output is split into the **reasoning trace** (before `</think>`) and the **final answer** (after); *only the final answer is evaluated*. If `</think>` never appears (token budget exhausted while thinking) the answer is empty and the item is counted as a failure of type *excessive response / truncation*.",
    "interactive_code": """
    MY_PROMPTS = ["How many r's are in the word 'strawberry'? Please reason step by step.", "Give me three tips for learning linear algebra."]
    INTERACTIVE = False
    for p in MY_PROMPTS:
        g = gen(p); parts = ca.split_reasoning(g["text"], expects_think=True)
        print(">>>", p); print("[reasoning chars]", len(parts["reasoning"]), "| closed:", parts["think_closed"]); print(parts["final"] or "(no final answer - truncated)"); print(f"[{g['new_tokens']} tokens, {g['seconds']:.1f}s]\\n")
    while INTERACTIVE:
        p = input("prompt> ").strip()
        if not p: break
        g = gen(p); print(g["text"], f"\\n[{g['new_tokens']} tokens, {g['seconds']:.1f}s]")
    """,
    "consistency_md": "DeepSeek uses sampling (T=0.6, top-p 0.95) as the README recommends (greedy decoding raises repetition, paper §3). Different seeds can therefore give different reasoning traces and sometimes different answers; the table shows how many *distinct final answers* appeared across 3 seeds.",
    "theoretical_md": """
    **Part B - THEORETICAL limitations (not observed in this run; from the paper / README)**
    - Prompt sensitivity: *few-shot prompting consistently degrades* R1 performance; zero-shot with a clear output format is recommended (paper §5).
    - Language mixing outside Chinese/English; may reason in English for other languages (paper §5).
    - Weaker than DeepSeek-V3 on function calling, multi-turn, role-play and JSON output (paper §5, about the full R1 - the distilled 7B is expected to be no better).
    - May skip the thinking pattern (`<think>\\n\\n</think>`) unless forced (README).
    - Long reasoning traces: high latency and memory per answer; endless repetition at low/zero temperature (README).
    - The distilled models received SFT only - the paper notes RL on them would likely help further and was not done.
    - Safety: full R1 refuses some queries after safety RL (paper); the distilled checkpoint's safety behaviour is not characterised in the paper.
    """,
    "resource_md": "(a) **input length** and (b) **generation length** on the primary (4-bit on T4) model, then (c) **precision variants** reloading the *same* checkpoint: 4-bit vs 8-bit, plus fp16 **only if the GPU has >= 22 GB** (otherwise recorded as SKIPPED with the reason). 4 dev prompts keep the 7B reruns affordable.",
    "resource_code": """
    MAX_CTX = LOAD_INFO.get("max_position_embeddings")
    LENGTHS = [256, 1024, 2048, 4096, 6000]
    RES_INPUT = ca.resource_input_length_experiment(model, tok, gen, LENGTHS, max_ctx=MAX_CTX, max_new_tokens=768, expects_think=True)
    RES_GEN = ca.resource_generation_length_experiment(gen, [64, 256, 512])
    SUBSET_IDS = ["GK02", "MR01", "LR02", "JS01"]
    SUBSET = [s for s in DEV if s["id"] in SUBSET_IDS] if not QUICK_RUN else DEV[:2]
    ca.release(model); model = None
    gpu_gb = ENV.get("gpu_total_gb") or 0
    precs = ["4bit", "8bit"] + (["bf16" if ENV.get("bf16_native") else "fp16"] if gpu_gb >= 22 else [])
    RES_GEN_KW = dict(GEN_KW, max_new_tokens=min(MAX_NEW_TOKENS, 768))
    RES_PRECISION = ca.resource_precision_experiment(MODEL_ID, HF_TOKEN, precs, SUBSET, RES_GEN_KW, expects_think=True, force_think_prefix=True)
    if gpu_gb < 22:
        RES_PRECISION.append({"precision": "fp16/bf16", "status": f"SKIPPED: needs ~16+ GB for weights alone; this GPU has {gpu_gb} GB"})
    ca.save_json(f"{RESULTS_DIR}/{MODEL_KEY}_resource.json", {"input_length": RES_INPUT, "generation_length": RES_GEN, "precision": RES_PRECISION})
    display(pd.DataFrame(RES_INPUT)); display(pd.DataFrame(RES_GEN))
    pd.DataFrame([{"precision": r["precision"], "status": r["status"][:60], **({"load_s": r["load"]["model_load_s"], "peak_gb": r["peak_mem_gb"], "acc": r["metrics"]["accuracy"], "mean_s": r["metrics"]["time_mean_s"], "tok_per_s": r["metrics"]["tokens_per_sec_aggregate"]} if r["status"] == "OK" else {})} for r in RES_PRECISION])
    """,
    "conclusion_md": """
    ### Observations (fill in after your run - these are *your* findings, not pre-written)
    - How often did the model finish thinking within `MAX_NEW_TOKENS`? (`n_unfinished_thinking`, `n_hit_max_tokens`.) How much of the latency is reasoning vs final answer?
    - Which categories gained from long reasoning (maths, multi-step, logic) and which suffered (format-constrained outputs, short factual answers)?
    - Did 4-bit vs 8-bit (and fp16 if run) change accuracy / answers on the subset? Memory / speed trade-off?
    - Consistency across seeds.

    ### Conclusion
    *(Write 5-8 sentences from the tables above. Remember: this is a distilled 7B SFT model, not the full R1; do not generalise beyond n=30 items.)*
    """,
}


# ---------------------------------------------------------------------------------------------
# Comparison notebook
# ---------------------------------------------------------------------------------------------
def comparison_notebook(research_rows, tech_rows, lit_rows):
    c = []
    c.append(md("""
    # G19_G20_Comparison - Gemma (`google/gemma-2b-it`) vs DeepSeek-R1-Distill-Qwen-7B
    Separate comparative assignment: **A. research papers**, **B. technical/implementation**, **C. experiments**.
    Part C **consumes the real result files** written by `G19_Gemma_Implementation.ipynb` and `G20_DeepSeek_R1_Implementation.ipynb`
    (`G19_results.json`, `G20_results.json`). If a file is missing the notebook says **NOT RUN** and shows no numbers - it never fills gaps.

    **Where to put the files:** upload the two `*_results.json` (and optionally the zips) to Colab, or mount Drive and set `SEARCH_DIRS`. Charts are saved to `/content/Pretrained_Model_CA/Comparative_Study/charts/`.
    """))
    c.append(md("## Setup"))
    c.append(code("""
    import os, json, glob, math, statistics
    import pandas as pd, numpy as np
    import matplotlib; import matplotlib.pyplot as plt
    from IPython.display import display, HTML, Markdown
    pd.set_option("display.max_colwidth", 400, "display.width", 220)
    OUT = "/content/Pretrained_Model_CA/Comparative_Study"; CHARTS = OUT + "/charts"; RES = OUT + "/results"
    for d in (CHARTS, RES): os.makedirs(d, exist_ok=True)
    def table(df):   # wrapped, left-aligned HTML table for long text
        return display(HTML(df.to_html(escape=True, index=False).replace("<td>", "<td style='text-align:left;vertical-align:top;white-space:pre-wrap'>")))
    """))
    c.append(md("## A. Research-paper comparison\nSources: Gemma paper (arXiv 2403.08295) and DeepSeek-R1 paper (arXiv 2501.12948) + official repos/model cards. Literature statements only - no experiments here."))
    c.append(code("RESEARCH = " + json.dumps(research_rows, indent=1, ensure_ascii=False) + "\ntable(pd.DataFrame(RESEARCH, columns=['Dimension', 'G19 Gemma (2B-IT; paper covers 2B/7B)', 'G20 DeepSeek-R1-Distill-Qwen-7B (paper covers R1-Zero, R1 and distillation)']))"))
    c.append(md("## B. Technical and implementation comparison"))
    c.append(code("TECH = " + json.dumps(tech_rows, indent=1, ensure_ascii=False) + "\ntable(pd.DataFrame(TECH, columns=['Aspect', 'G19 Gemma', 'G20 DeepSeek']))"))
    c.append(md("""
    ### Literature-reported results (NOT produced by us)
    Different benchmarks, shot counts, decoding and model variants - **not comparable with each other or with our experiments**. Gemma numbers are for *pretrained* checkpoints (paper Table 6), DeepSeek numbers are the paper's reported distilled-model results.
    """))
    c.append(code("LIT = " + json.dumps(lit_rows, indent=1, ensure_ascii=False) + "\ntable(pd.DataFrame(LIT, columns=['Benchmark', 'Gemma 2B (pretrained)', 'Gemma 7B (pretrained)', 'DeepSeek-R1-Distill-Qwen-7B', 'Source / caveat']))"))
    c.append(md("## C. Experimental comparison (same held-out prompts, real prediction files)"))
    c.append(code("""
    SEARCH_DIRS = ["/content", "/content/Pretrained_Model_CA/G19_Gemma/results", "/content/Pretrained_Model_CA/G20_DeepSeek/results", "/content/drive/MyDrive/Pretrained_Model_CA"]
    def find(name):
        for d in SEARCH_DIRS:
            hits = glob.glob(os.path.join(d, "**", name), recursive=True)
            if hits: return hits[0]
    paths = {k: find(f"{k}_results.json") for k in ("G19", "G20")}
    if not all(paths.values()):
        try:
            from google.colab import files
            print("Missing:", [k for k, v in paths.items() if not v], "- upload the *_results.json file(s) now:")
            up = files.upload()
            for fn in up: os.rename(fn, f"/content/{fn}")
            paths = {k: find(f"{k}_results.json") for k in ("G19", "G20")}
        except Exception: pass
    R = {k: json.load(open(p)) for k, p in paths.items() if p}
    print({k: paths[k] or "NOT FOUND" for k in paths})
    HAVE = len(R) == 2
    if not HAVE:
        display(Markdown("### EXPERIMENTS NOT RUN / FILES MISSING\\nRun the individual notebooks first. Cells below are skipped; **no values are invented**."))
    for k, r in R.items():
        if r["meta"].get("quick_run"): display(Markdown(f"**WARNING: {k} results come from QUICK_RUN (reduced samples) - not valid for comparison.**"))
    """))
    c.append(md("### C1. Same held-out test prompts?  Fairness audit"))
    c.append(code("""
    if HAVE:
        ids = {k: [r["id"] for r in R[k]["test"]["records"]] for k in R}
        prompts = {k: [r["prompt"] for r in R[k]["test"]["records"]] for k in R}
        print("identical test ids:", ids["G19"] == ids["G20"], "| identical prompt text:", prompts["G19"] == prompts["G20"])
        m = {k: R[k]["meta"] for k in R}
        fair = pd.DataFrame([
            ["Checkpoint", m["G19"]["checkpoint"], m["G20"]["checkpoint"]],
            ["Parameters actually loaded (B)", round(m["G19"]["load"]["n_parameters_loaded"] / 1e9, 2), round(m["G20"]["load"]["n_parameters_loaded"] / 1e9, 2)],
            ["Precision", m["G19"]["precision"], m["G20"]["precision"]],
            ["GPU", m["G19"]["env"].get("gpu"), m["G20"]["env"].get("gpu")],
            ["Decoding", json.dumps(m["G19"]["generation_config"]), json.dumps(m["G20"]["generation_config"])],
            ["Reasoning mode", "none (direct answer)", "long chain-of-thought inside <think>, final answer evaluated"],
            ["Answers cut by token cap (dev+test)", R["G19"]["dev"]["metrics"]["n_hit_max_tokens"] + R["G19"]["test"]["metrics"]["n_hit_max_tokens"], R["G20"]["dev"]["metrics"]["n_hit_max_tokens"] + R["G20"]["test"]["metrics"]["n_hit_max_tokens"]],
            ["Runtime/GPU same session?", "see run_utc", f"{m['G19']['run_utc']} vs {m['G20']['run_utc']}"],
        ], columns=["Factor affecting fairness", "G19", "G20"])
        table(fair)
        flags = []
        if m["G19"]["precision"] != m["G20"]["precision"]: flags.append("different numeric precision / quantisation")
        if m["G19"]["generation_config"].get("do_sample") != m["G20"]["generation_config"].get("do_sample"): flags.append("greedy vs sampling decoding")
        if m["G19"]["env"].get("gpu") != m["G20"]["env"].get("gpu"): flags.append("different GPUs -> timings/memory not strictly comparable")
        flags.append("model sizes differ (~2.5B vs ~7.6B parameters) - capability gaps are confounded with size")
        display(Markdown("**Differences that limit fairness:** " + "; ".join(flags)))
    """))
    c.append(md("### C2. Required comparison table"))
    c.append(code("""
    def lm(k, split): return R[k][split]["metrics"]
    def pct(x): return "n/a" if x is None else f"{100*x:.1f}%"
    def f(x, d=2): return "n/a" if x is None else f"{x:.{d}f}"
    if HAVE:
        rows = []
        for split, label in (("test", "HELD-OUT TEST (n=10)"), ("dev", "DEV (n=20)")):
            a, b = lm("G19", split), lm("G20", split)
            rows += [[label, "", "", ""],
                ["Checkpoint", R["G19"]["meta"]["checkpoint"], R["G20"]["meta"]["checkpoint"], ""],
                ["Loading time (s, tokenizer+model from cache)", f(R["G19"]["meta"]["load"]["total_load_s"], 1), f(R["G20"]["meta"]["load"]["total_load_s"], 1), ""],
                ["Mean inference time (s)", f(a["time_mean_s"]), f(b["time_mean_s"]), ""],
                ["Median inference time (s)", f(a["time_median_s"]), f(b["time_median_s"]), ""],
                ["Tokens per second (aggregate)", f(a["tokens_per_sec_aggregate"], 1), f(b["tokens_per_sec_aggregate"], 1), "tokens include G20's reasoning trace"],
                ["Peak GPU memory (GB, allocated)", f(a["peak_gpu_mem_gb_max"]), f(b["peak_gpu_mem_gb_max"]), "different precision!"],
                ["Accuracy", pct(a["accuracy"]) + f" ({a['n_correct']}/{a['n']})", pct(b["accuracy"]) + f" ({b['n_correct']}/{b['n']})", f"95% CI {a['accuracy_ci95'][0]:.2f}-{a['accuracy_ci95'][1]:.2f} | {b['accuracy_ci95'][0]:.2f}-{b['accuracy_ci95'][1]:.2f}"],
                ["Exact Match (short-answer items)", pct(a["exact_match"]), pct(b["exact_match"]), "strict; verbose answers score 0"],
                ["Token F1 (short-answer items)", pct(a["token_f1"]), pct(b["token_f1"]), ""],
                ["Reasoning correctness", pct(a["reasoning_correctness"]) + f" (n={a['n_reasoning']})", pct(b["reasoning_correctness"]) + f" (n={b['n_reasoning']})", ""],
                ["Instruction compliance", pct(a["instruction_compliance"]) + f" (n={a['n_compliance']})", pct(b["instruction_compliance"]) + f" (n={b['n_compliance']})", ""],
                ["Structured output validity", pct(a["structured_validity"]) + f" (n={a['n_structured']})", pct(b["structured_validity"]) + f" (n={b['n_structured']})", ""],
                ["ROUGE-L (summaries)", f(a["rougeL"], 3), f(b["rougeL"], 3), f"n={a['n_rouge']}"],
                ["Answers hitting token cap", a["n_hit_max_tokens"], b["n_hit_max_tokens"], ""]]
        MAIN = pd.DataFrame(rows, columns=["Parameter", "G19 Gemma", "G20 DeepSeek", "Note"]); table(MAIN)
        MAIN.to_csv(f"{RES}/comparison_main_table.csv", index=False)
    """))
    c.append(md("### C3. Category-wise comparison (dev+test pooled, 3 items per category) and which model is better"))
    c.append(code("""
    if HAVE:
        def percat(k):
            recs = R[k]["dev"]["records"] + R[k]["test"]["records"]; out = {}
            for r in recs: d = out.setdefault(r["category"], [0, 0]); d[0] += int(r["eval"]["correct"]); d[1] += 1
            return out
        pa, pb = percat("G19"), percat("G20")
        cats = sorted(set(pa) | set(pb)); rows = []
        for ct in cats:
            ka, na = pa.get(ct, (0, 0)); kb, nb = pb.get(ct, (0, 0))
            verdict = "tie" if ka == kb else ("G19 Gemma ahead" if ka > kb else "G20 DeepSeek ahead")
            rows.append([ct, f"{ka}/{na}", f"{kb}/{nb}", verdict, "n=3: a 1-item gap is NOT statistically meaningful"])
        CAT = pd.DataFrame(rows, columns=["Category", "G19 correct", "G20 correct", "Observed", "Caveat"]); table(CAT)
        CAT.to_csv(f"{RES}/comparison_category_table.csv", index=False)
        display(Markdown("**How to explain each difference (write after inspecting the prediction files):** open `G19_predictions_*.json` / `G20_predictions_*.json`, read the failing items and state *why* (e.g. G20 truncated while thinking, G19 format error, arithmetic slip). The notebook does not guess causes."))
        fails = {k: {} for k in R}
        for k in R:
            for r in R[k]["dev"]["records"] + R[k]["test"]["records"]:
                if not r["eval"]["correct"]: fails[k][r["id"]] = (r["category"], (r["final_output"] or r["raw_output"])[:140].replace("\\n", " "), r.get("think_closed"))
        for k in fails:
            print(f"\\n{k} failed items ({len(fails[k])}):")
            for i, v in fails[k].items(): print("  ", i, v[0], "| think_closed:", v[2], "|", v[1])
    """))
    c.append(md("### C4. Charts"))
    c.append(code("""
    COL = {"G19": "#3b6ea5", "G20": "#c0643a"}
    def bar2(ax, vals, title, ylabel, fmt="{:.1f}"):
        xs = ["G19 Gemma", "G20 DeepSeek"]; bars = ax.bar(xs, vals, color=[COL["G19"], COL["G20"]]); ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel)
        for b_, v in zip(bars, vals): ax.text(b_.get_x() + b_.get_width() / 2, v, "n/a" if v is None else fmt.format(v), ha="center", va="bottom", fontsize=9)
    if HAVE:
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        bar2(ax[0], [R[k]["meta"]["load"]["total_load_s"] for k in ("G19", "G20")], "Model loading time", "seconds")
        bar2(ax[1], [lm(k, "test")["time_mean_s"] for k in ("G19", "G20")], "Mean inference time (held-out test)", "seconds")
        bar2(ax[2], [lm(k, "test")["peak_gpu_mem_gb_max"] for k in ("G19", "G20")], "Peak GPU memory (held-out test)", "GB", "{:.2f}")
        plt.tight_layout(); plt.savefig(f"{CHARTS}/load_time_inference_time_memory.png", dpi=150); plt.show()
        mets = [("accuracy", "Accuracy"), ("exact_match", "Exact Match"), ("reasoning_correctness", "Reasoning"), ("instruction_compliance", "Compliance"), ("structured_validity", "Struct. validity")]
        x = np.arange(len(mets)); w = 0.38; fig, ax = plt.subplots(figsize=(9, 4))
        for i, k in enumerate(("G19", "G20")): ax.bar(x + (i - .5) * w, [(lm(k, "test")[m_] or 0) for m_, _ in mets], w, label=k, color=COL[k])
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in mets]); ax.set_ylim(0, 1.05); ax.set_title("Evaluation scores - held-out test (n=10)"); ax.legend(); plt.tight_layout(); plt.savefig(f"{CHARTS}/evaluation_scores_test.png", dpi=150); plt.show()
        cats = sorted(pa); x = np.arange(len(cats)); fig, ax = plt.subplots(figsize=(11, 4.2))
        for i, (k, pp) in enumerate((("G19", pa), ("G20", pb))): ax.bar(x + (i - .5) * w, [pp[c_][0] / pp[c_][1] for c_ in cats], w, label=k, color=COL[k])
        ax.set_xticks(x); ax.set_xticklabels(cats, rotation=30, ha="right"); ax.set_ylim(0, 1.05); ax.set_title("Category-wise accuracy (dev+test, 3 items/category)"); ax.legend(); plt.tight_layout(); plt.savefig(f"{CHARTS}/category_accuracy.png", dpi=150); plt.show()
        vs = sorted({v for k in R for v in R[k]["robustness"]["summary"]}); x = np.arange(len(vs)); fig, ax = plt.subplots(figsize=(10, 4))
        for i, k in enumerate(("G19", "G20")): ax.bar(x + (i - .5) * w, [R[k]["robustness"]["summary"].get(v, {}).get("accuracy", 0) for v in vs], w, label=k, color=COL[k])
        ax.set_xticks(x); ax.set_xticklabels(vs, rotation=25, ha="right"); ax.set_ylim(0, 1.05); ax.set_title("Robustness accuracy by input variation"); ax.legend(); plt.tight_layout(); plt.savefig(f"{CHARTS}/robustness.png", dpi=150); plt.show()
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
        for k in R:
            ok = [r for r in R[k]["resource"]["precision"] if r.get("status") == "OK"]
            ax[0].bar([f"{k}\\n{r['precision']}" for r in ok], [r["peak_mem_gb"] or 0 for r in ok], color=COL[k]); ax[1].bar([f"{k}\\n{r['precision']}" for r in ok], [r["metrics"]["time_mean_s"] for r in ok], color=COL[k])
        ax[0].set_title("Resource experiment: peak GPU memory (GB)"); ax[1].set_title("Resource experiment: mean time / prompt (s)"); plt.tight_layout(); plt.savefig(f"{CHARTS}/resource_precision.png", dpi=150); plt.show()
    """))
    c.append(md("### C5. Resource experiments side by side"))
    c.append(code("""
    if HAVE:
        for k in R:
            print(k, "- precision experiment (same checkpoint, different precision):")
            display(pd.DataFrame([{"precision": r["precision"], "status": r["status"][:70], **({"load_s": round(r["load"]["model_load_s"], 1), "peak_GB": round(r["peak_mem_gb"] or 0, 2), "acc": r["metrics"]["accuracy"], "mean_s": round(r["metrics"]["time_mean_s"], 1), "tok/s": round(r["metrics"]["tokens_per_sec_aggregate"], 1)} if r["status"] == "OK" else {})} for r in R[k]["resource"]["precision"]]))
            print(k, "- input length:"); display(pd.DataFrame(R[k]["resource"]["input_length"]))
    """))
    c.append(md("### C6. Export all numbers to Excel"))
    c.append(code("""
    if HAVE:
        with pd.ExcelWriter(f"{RES}/Experimental_Results.xlsx") as xw:
            MAIN.to_excel(xw, sheet_name="main_table", index=False); CAT.to_excel(xw, sheet_name="category", index=False)
            for k in R:
                pd.DataFrame([{**{kk: r[kk] for kk in ("id", "category", "seconds", "new_tokens", "tokens_per_sec", "hit_max_tokens")}, "correct": r["eval"]["correct"], "extracted": r["eval"]["extracted"], "output": (r["final_output"] or "")[:500]} for r in R[k]["dev"]["records"] + R[k]["test"]["records"]]).to_excel(xw, sheet_name=f"{k}_predictions", index=False)
                pd.DataFrame(R[k]["robustness"]["summary"]).T.to_excel(xw, sheet_name=f"{k}_robustness")
                pd.DataFrame(R[k]["failures"]).to_excel(xw, sheet_name=f"{k}_failures", index=False)
        print("wrote", f"{RES}/Experimental_Results.xlsx")
    """))
    c.append(md("""
    ## D. Discussion template (complete after reading the tables)
    1. **Per-category winner and why** - use C3 + the printed failed items. Do not call a universal winner: the evidence is n=30 items, different sizes (~2.5B vs ~7.6B), possibly different precision/decoding (C1).
    2. **Cost of reasoning** - compare tokens/s vs seconds-per-correct-answer; G20 spends many tokens thinking.
    3. **Format obedience** - compliance / structured validity.
    4. **Threats to validity** - small n, rule-based graders, single run for sampling-based G20, token cap, quantisation.
    """))
    return notebook(c, "G19_G20_Comparison.ipynb")


if __name__ == "__main__":
    pass
