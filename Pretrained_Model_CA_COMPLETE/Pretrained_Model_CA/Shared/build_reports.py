"""Shared/build_reports.py - regenerates every PDF in Reports/ + per-folder documentation.
Usage: python build_reports.py [PACKAGE_ROOT]   (default: parent of this script's folder)
If <model>_results.json files exist anywhere under PACKAGE_ROOT (copy them from Colab into G19_Gemma/results and G20_DeepSeek/results),
real numbers are rendered; otherwise every measured cell says NOT RUN. Nothing is ever estimated or invented."""
import os, sys, glob, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pdfkit import make_pdf
import docs_text as T
from content_data import RESEARCH_ROWS, TECH_ROWS, LIT_ROWS
import dataset_def as D

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
NAMES = {"G19": ("G19_Gemma", "google/gemma-2b-it"), "G20": ("G20_DeepSeek", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")}
NR = "NOT RUN"

def find(k):
    c = glob.glob(os.path.join(ROOT, "**", k + "_results.json"), recursive=True)
    return json.load(open(c[0], encoding="utf-8")) if c else None
R = {k: find(k) for k in NAMES}
def P(*a): return os.path.join(ROOT, *a)
def f(x, d=3, pct=False):
    if x is None: return NR
    return ("%.1f%%" % (100 * x)) if pct else (("%." + str(d) + "f") % x)
def ci(m): 
    c = m.get("accuracy_ci95"); return "[%.0f%%, %.0f%%]" % (100 * c[0], 100 * c[1]) if c else NR

def metric_rows(k, split):
    r = R[k]
    if not r: return [[n, NR] for n in ["Accuracy (n correct / n)", "95% Wilson CI", "Exact match", "Token F1", "ROUGE-1 / 2 / L", "Reasoning correctness", "Instruction compliance", "Structured validity",
                                      "Mean / median / p90 seconds per sample", "Tokens per second (aggregate)", "Mean new tokens", "Peak GPU memory (GB)", "Hit max_new_tokens"]]
    m = r[split]["metrics"]
    return [["Accuracy (n correct / n)", "%s (%d/%d)" % (f(m["accuracy"], pct=True), m["n_correct"], m["n"])], ["95% Wilson CI", ci(m)], ["Exact match", f(m["exact_match"])], ["Token F1", f(m["token_f1"])],
            ["ROUGE-1 / 2 / L", "%s / %s / %s (n=%d)" % (f(m["rouge1"]), f(m["rouge2"]), f(m["rougeL"]), m["n_rouge"])], ["Reasoning correctness", "%s (n=%d)" % (f(m["reasoning_correctness"]), m["n_reasoning"])],
            ["Instruction compliance", "%s (n=%d)" % (f(m["instruction_compliance"]), m["n_compliance"])], ["Structured validity", "%s (n=%d)" % (f(m["structured_validity"]), m["n_structured"])],
            ["Mean / median / p90 seconds per sample", "%s / %s / %s" % (f(m["time_mean_s"], 1), f(m["time_median_s"], 1), f(m["time_p90_s"], 1))],
            ["Tokens per second (aggregate)", f(m["tokens_per_sec_aggregate"], 1)], ["Mean new tokens", f(m["mean_new_tokens"], 0)], ["Peak GPU memory (GB)", f(m["peak_gpu_mem_gb_max"], 2)],
            ["Hit max_new_tokens / unfinished thinking", "%s / %s" % (m["n_hit_max_tokens"], m["n_unfinished_thinking"])]]

def percat(k, split):
    r = R[k]
    cats = sorted({s["category"] for s in D.DATASET})
    if not r: return [[c, NR] for c in cats]
    pc = r[split]["metrics"]["per_category"]
    return [[c, "%s/%s" % (pc[c]["correct"], pc[c]["n"]) if c in pc else "-"] for c in cats]

def failures(k):
    r = R[k]
    if not r: return [("note", "Failure cases: " + NR + ". The notebook's section 'Failure analysis' fills <KEY>_failure_cases.json with every observed failure; theoretical failure modes below are labelled as such.")]
    fl = r.get("failures", [])
    b = [("p", "%d observed failures were auto-detected (the notebook asks you to review them manually before citing)." % len(fl))]
    rows = [[x["id"], x["category"], x["failure_type"], x["input"][:160], x["actual_output"][:200]] for x in fl[:15]]
    if rows: b.append(("table", ["ID", "Category", "Type", "Input (trunc.)", "Actual output (trunc.)"], rows, [1, 1.5, 2, 3, 3.5]))
    if len(fl) < 5: b.append(("note", "Fewer than 5 failures were observed in this run; per the brief we do not fabricate more. Run the full dataset plus robustness set to see additional ones."))
    return b

def robustness(k):
    r = R[k]
    if not r: return [("p", "Robustness (5 variations, 26 prompts, saved with input/expected/actual/evaluation/time/observation): " + NR)]
    s = r["robustness"]["summary"]
    return [("table", ["Variation", "n", "Accuracy"], [[v, str(s[v].get("n", "")), f(s[v].get("accuracy"), pct=True)] for v in s], [4, 1, 2])]

def resource(k):
    r = R[k]
    if not r: return [("p", "Resource experiments (precision, input length, generation length): " + NR)]
    b = []
    pr = r["resource"]["precision"]
    rows = []
    for x in pr:
        if x.get("status") == "OK":
            rows.append([x["precision"], "OK", f(x["load"]["model_load_s"], 1), f(x.get("peak_mem_gb"), 2), f(x["metrics"]["accuracy"], pct=True), f(x["metrics"]["time_mean_s"], 1), f(x["metrics"]["tokens_per_sec_aggregate"], 1)])
        else: rows.append([x["precision"], str(x.get("status"))[:80], "-", "-", "-", "-", "-"])
    b.append(("table", ["Precision", "Status", "Load s", "Peak GB", "Acc.", "Mean s", "tok/s"], rows, [1.2, 3, 1, 1, 1, 1, 1]))
    il = r["resource"].get("input_length") or []
    if il: b.append(("table", ["Input length experiment"], [[json.dumps(x, default=str)[:200]] for x in il]))
    return b

def meta(k):
    r = R[k]
    if not r: return [("note", "Run metadata: " + NR + " (no results file found).")]
    m = r["meta"]; e = m.get("env", {})
    return [("table", ["Field", "Value"], [["Checkpoint", m["checkpoint"]], ["Precision", str(m["precision"])], ["Quick run", str(m["quick_run"])], ["Run (UTC)", m["run_utc"]],
            ["Environment", json.dumps(e, default=str)[:300]], ["Generation config", json.dumps(m["generation_config"])], ["Load info", json.dumps(m.get("load"), default=str)[:300]]], [1.5, 5])]

DESIGN = lambda k: [
    ("h1", "Dataset and evaluation design"),
    ("p", "30 independently authored samples, 10 categories x 3 (general knowledge, instruction following, logical reasoning, mathematical reasoning, multi-step, summarisation, coding, analytical, structured output, ambiguous/misleading). Each has id, category, prompt, reference, difficulty and criteria. Items 1-2 of every category form the 20-sample development split; item 3 forms the 10-sample held-out test split. Test prompts were never used to tune prompts, checkers or decoding. No model weights are trained or fine-tuned; the dataset is evaluation-only."),
    ("table", ["Metric", "How computed", "Limits"], [
        ["Accuracy", "Rule-based per-sample checker (numeric, yes/no, keywords, constraints, unit tests, schema)", "Checkers can be stricter or looser than a human; n is tiny, see Wilson CI"],
        ["Exact match / token F1", "Normalised string equality / token overlap vs reference", "Penalise correct paraphrases; meaningful mainly for short answers"],
        ["ROUGE-1/2/L", "Own implementation (no stemming) on summarisation items", "Surface overlap, blind to factuality"],
        ["Reasoning correctness", "Final answer after </think> or in the response for reasoning items", "Does not grade the quality of intermediate steps"],
        ["Math correctness", "Numeric extraction (Final answer > \\boxed > last number) with tolerance", "Extraction mistakes possible if the format is unusual"],
        ["Instruction compliance", "Programmatic constraint checks (word count, bullets, forbidden words ...)", "Only constraints that can be checked automatically"],
        ["Structured validity", "Parse JSON / CSV / markdown table / key-value then check spec", "Valid syntax does not imply correct content"]], [1.5, 4, 3.5]),
    ("p", "Robustness set: 26 prompts in 5 variations (V1 prompt wording, V2 input complexity, V3 ambiguous instructions, V4 math complexity, V5 structured formats). Consistency: repeated seeds on selected prompts.")]

def impl_report(k):
    ck = NAMES[k][1]; g = k == "G19"; have = R[k] is not None
    b = [T.status_note(have), ("h1", "1. Implementation overview"),
         ("p", "Checkpoint: **%s**. Framework: Hugging Face transformers + PyTorch on Google Colab. Notebook: %s. The notebook runs top to bottom from a fresh Colab session: installs dependencies, prints version/GPU/RAM, authenticates%s, loads the exact checkpoint and tokenizer, shows the chat template, defines one reusable inference function, runs interactive tests, then the dataset evaluation, robustness, failure analysis and resource experiments, saves charts and JSON/CSV results, and writes a zip." % (ck, "G19_Gemma_Implementation.ipynb" if g else "G20_DeepSeek_R1_Implementation.ipynb", " (HF token + accepted Gemma licence required)" if g else " (no token needed for this public MIT model)")),
         ("h1", "2. Design decisions"),
         ("bullets", (["bf16 on GPUs that support it, else fp16 (fp16 revision exists); greedy decoding (do_sample=False), max_new_tokens=512, so results are deterministic up to GPU nondeterminism.", "Prompt via tokenizer.apply_chat_template; no system role (not part of Gemma's format).", "Gemma's output is direct; no reasoning-trace splitting is required."] if g else
                      ["Exact checkpoint, no substitution. 4-bit NF4 by default on a T4 (15.2 GB weights do not fit); bf16/fp16 only if the GPU has >= ~22 GB, otherwise that precision is skipped and the reason recorded.", "Sampling T=0.6, top-p 0.95 as recommended [DR]; max_new_tokens=1536 (the paper allows 32,768) to keep Colab time bounded - truncation is counted, not hidden.", "No system prompt; '<think>\\n' forced at the start of the response per README; output split on '</think>' into reasoning trace and final answer, and only the final answer is graded."])),
         ("h1", "3. Run metadata")] + meta(k) + DESIGN(k) + [
         ("h1", "4. Results - development split (20 prompts)"), ("table", ["Metric", "Value"], metric_rows(k, "dev"), [3, 4]),
         ("h1", "5. Results - held-out test split (10 prompts)"), ("table", ["Metric", "Value"], metric_rows(k, "test"), [3, 4]),
         ("table", ["Category", "Test correct"], percat(k, "test"), [3, 2]),
         ("note", "With only 10 held-out prompts a single answer moves accuracy by 10 points; interpret per-category cells (1 prompt each) as anecdotes, not statistics."),
         ("h1", "6. Robustness")] + robustness(k) + [("h1", "7. Failure analysis (observed)")] + failures(k) + [
         ("h2", "Theoretical failure modes (NOT observed unless listed above)"),
         ("bullets", (["Small-model factual hallucination and arithmetic slips; instruction constraints (exact word counts) violated; ambiguous prompts answered without clarification; invalid JSON when asked for strict output."] if g else
                      ["Endless or very long reasoning that exhausts the token cap; repetition; language mixing; overthinking trivial questions; 4-bit quantisation noise; sampling makes answers vary between runs."])),
         ("h1", "8. Resource experiments")] + resource(k) + [
         ("h1", "9. Reproducibility"), ("bullets", ["Seeds are recorded in the results JSON; GPU type, library versions and the full generation config are saved in meta.", "Estimate-based statements in this report (e.g. weights size) are labelled as estimates; measured memory comes only from torch.cuda counters in the notebook."]),
         ("h1", "10. Limitations"), ("bullets", ["30 prompts, one run per prompt (plus a small consistency test): large uncertainty.", "Rule-based graders were validated by 67 unit checks (test_common.py) but are not a human judge.", "Everything here is inference-only; no training, fine-tuning or RL was performed."])]
    return b

def comparative():
    have = all(R.values())
    def row(lbl, fn):
        return [lbl] + [fn(k) if R[k] else NR for k in ("G19", "G20")]
    tab = [row("Test accuracy", lambda k: f(R[k]["test"]["metrics"]["accuracy"], pct=True)), row("Dev accuracy", lambda k: f(R[k]["dev"]["metrics"]["accuracy"], pct=True)),
           row("Mean seconds / sample (test)", lambda k: f(R[k]["test"]["metrics"]["time_mean_s"], 1)), row("Tokens / s (test)", lambda k: f(R[k]["test"]["metrics"]["tokens_per_sec_aggregate"], 1)),
           row("Mean new tokens (test)", lambda k: f(R[k]["test"]["metrics"]["mean_new_tokens"], 0)), row("Peak GPU GB (test)", lambda k: f(R[k]["test"]["metrics"]["peak_gpu_mem_gb_max"], 2)),
           row("ROUGE-L (test)", lambda k: f(R[k]["test"]["metrics"]["rougeL"]))]
    cats = sorted({s["category"] for s in D.DATASET})
    pc = [[c, percat("G19", "test")[i][1], percat("G20", "test")[i][1]] for i, c in enumerate(cats)]
    return have, tab, pc

FAIR = [("h1", "Fairness of the comparison"),
        ("table", ["Factor", "G19 Gemma-2b-it", "G20 R1-Distill-Qwen-7B", "Effect"], [
            ["Parameters", "~2.5B", "7B class", "Larger model advantage for G20"], ["Precision", "bf16/fp16", "4-bit NF4 on T4 (default)", "Quantisation hurts G20 slightly and changes memory/speed"],
            ["Decoding", "greedy", "sampling T=0.6 top-p 0.95", "G20 is stochastic; consistency test quantifies"], ["Token cap", "512", "1536", "G20 needs far more tokens; truncation counted"],
            ["Specialisation", "general chat model", "reasoning-specialised (SFT from R1)", "Task mix matters: maths vs conversational"], ["Prompt format", "user turn only", "user turn only, '<think>' forced", "Both avoid system role"],
            ["Hardware", "same Colab GPU if run in one session", "same", "Timing comparable only on the same GPU"]], [1.3, 2, 2.5, 3]),
        ("p", "Because of these differences, results answer 'what do these two released checkpoints do on our 30 prompts under their recommended settings', not 'which architecture or training method is better'. We make no universal-winner claim.")]

def build_all():
    outs = []
    # per-model research + implementation
    for k, (folder, ck) in NAMES.items():
        outs.append(make_pdf(P(folder, "research", k + "_Research_Analysis.pdf"), k + " - Research Analysis", ck, T.research_blocks(k)))
        outs.append(make_pdf(P(folder, "documentation", k + "_Implementation_Report.pdf"), k + " - Implementation Report", ck, impl_report(k)))
    have, tab, pc = comparative()
    outs.append(make_pdf(P("Comparative_Study", "research", "G19_G20_Comparative_Research_Analysis.pdf"), "G19 vs G20 - Comparative Research Analysis", "Gemma 1.0 2B-it vs DeepSeek-R1-Distill-Qwen-7B (paper-level comparison)",
        [("h1", "1. Side-by-side research comparison"), ("table", ["Dimension", "Gemma (G19)", "DeepSeek-R1 (G20)"], RESEARCH_ROWS, [1.4, 4, 4]),
         ("h1", "2. What the comparison means"), ("bullets", ["Gemma is a pretraining + SFT + RLHF release of small general models; DeepSeek-R1 is a post-training (RL) recipe whose product is distilled into dense students.", "The checkpoint we run for G20 is the student: SFT-only, no RL of its own.", "Literature numbers (below) come from different protocols and are not comparable to each other or to our results."]),
         ("table", ["Benchmark", "Gemma 2B", "Gemma 7B", "R1-Distill-Qwen-7B", "Source"], LIT_ROWS, [2, 1.5, 1.5, 1.5, 3])]))
    outs.append(make_pdf(P("Comparative_Study", "implementation", "G19_G20_Comparative_Implementation_Analysis.pdf"), "G19 vs G20 - Comparative Implementation Analysis", "Technical comparison of how the two checkpoints are run",
        [("h1", "1. Technical comparison"), ("table", ["Aspect", "G19 Gemma-2b-it", "G20 R1-Distill-Qwen-7B"], TECH_ROWS, [1.4, 4, 4]),
         ("h1", "2. Practical observations"), ("bullets", ["G19 needs a gated-licence token; G20 does not.", "G20 must run quantised on a 15 GB T4; this is a confounder, not a feature.", "G20 outputs must be post-processed ('</think>' split); G19 outputs can be graded directly.", "Memory and timing values are measured only by the notebooks (NOT RUN here)."]),
         ("h1", "3. Official code references"), ("p", "See Paper_to_Implementation_Mapping.pdf for verified file/class mappings; note the DeepSeek-R1 repo has no code and Gemma-1 JAX code is only in 2024 history (commit 59ffafe).")] + FAIR))
    exp = [T.status_note(have), ("h1", "1. Experimental setup"), ("p", "Both notebooks evaluate the same 30 prompts (20 dev / 10 held-out test) and the same 26-prompt robustness set, then G19_G20_Comparison.ipynb reads G19_results.json and G20_results.json and produces the tables, charts and Experimental_Results.xlsx. This PDF is rebuilt from those files by Shared/build_reports.py."),
           ("h1", "2. Required comparison table"), ("table", ["Measure", "G19", "G20"], tab, [3, 2, 2]), ("h1", "3. Per-category comparison (test split, correct/total)"), ("table", ["Category", "G19", "G20"], pc, [3, 2, 2]),
           ("h1", "4. Charts")]
    for fn in ("load_time_inference_time_memory", "evaluation_scores_test", "category_accuracy", "robustness", "resource_precision"):
        exp.append(("image", P("Comparative_Study", "charts", fn + ".png"), 440))
    exp += FAIR + [("h1", "5. Interpretation"), ("p", ("Interpretation is written only after real numbers exist. " + NR + ". After running both notebooks and the comparison notebook, rebuild this PDF; then read the tables above, state per-category winners with their confidence intervals and avoid generalising beyond the 30-prompt set.") if not have else
                    "Read the table and per-category cells above together with the Wilson intervals; with 10 test prompts only large gaps are informative, and the confounders listed in the fairness table can explain much of any gap."),
            ("h1", "6. Threats to validity"), ("bullets", ["Tiny sample; sampling noise for G20; checkers imperfect; precision mismatch; token-cap mismatch; possible different GPUs across sessions."])]
    outs.append(make_pdf(P("Comparative_Study", "results", "G19_G20_Comparative_Experimental_Report.pdf"), "G19 vs G20 - Comparative Experimental Report", "Measured results (NOT RUN where no result files exist)", exp))
    outs.append(make_pdf(P("Reports", "Paper_to_Implementation_Mapping.pdf"), "Paper to Implementation Mapping", "Gemma (arXiv:2403.08295) and DeepSeek-R1 (arXiv:2501.12948)", T.mapping_blocks()))
    outs.append(make_pdf(P("Viva", "Viva_Preparation.pdf"), "Viva Preparation", "%d questions with answers (G19, G20, comparison)" % len(T.VIVA), T.viva_blocks()))
    # copy reports into Reports/
    os.makedirs(P("Reports"), exist_ok=True)
    for o in outs:
        d = P("Reports", os.path.basename(o))
        if os.path.abspath(o) != os.path.abspath(d): shutil.copy(o, d)
    return outs

def dataset_files():
    ds = D.dataset_json()
    for folder in ("G19_Gemma", "G20_DeepSeek"):
        d = P(folder, "dataset"); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "evaluation_dataset.json"), "w", encoding="utf-8").write(ds if isinstance(ds, str) else json.dumps(ds, indent=2, ensure_ascii=False))
        import csv
        with open(os.path.join(d, "evaluation_dataset.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["id", "split", "category", "difficulty", "prompt", "reference", "criteria"])
            for s in D.DATASET: w.writerow([s["id"], s["split"], s["category"], s["difficulty"], s["prompt"], s["reference"], s["criteria"]])
        open(os.path.join(d, "DATASET_CARD.md"), "w").write("# Evaluation dataset card\n\n- 30 hand-authored samples, 10 categories x 3; dev = items 1-2 per category (20), test = item 3 per category (10).\n- Fields: id, split, category, difficulty, prompt, reference, criteria (+ eval_type/eval_spec in the JSON used by the notebooks).\n- Plus a 26-prompt robustness set in 5 variations (in the JSON).\n- Purpose: evaluation only. No training or fine-tuning of any model. Test prompts were not used to tune anything.\n- Authored for this lab; not drawn from public benchmarks, so contamination with training data is unlikely but not provable.\n- Limitation: tiny; confidence intervals are wide.\n")

if __name__ == "__main__":
    dataset_files()
    for o in build_all(): print("built", os.path.relpath(o, ROOT))
    print("results found:", {k: bool(v) for k, v in R.items()})
