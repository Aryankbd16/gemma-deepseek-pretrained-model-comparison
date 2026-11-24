"""ca_common.py -- shared evaluation + inference helpers for the Pretrained-Model CA.

Used verbatim by all three notebooks (G19 Gemma, G20 DeepSeek-R1-Distill-Qwen-7B,
G19-vs-G20 comparison).  The pure-python part (metrics, extraction, evaluators,
aggregation) has no heavy dependencies and is unit-tested outside Colab.
The HF/torch part (model loading, generation, memory measurement) imports torch
lazily and can only run where a GPU/CPU torch install exists (Colab).

NOTE ON METRICS: ROUGE here is a small self-contained implementation
(lower-cased word tokens, no stemming).  Absolute values therefore differ
slightly from the `rouge-score` package; it is applied identically to both models.
"""
import json, re, time, math, statistics, string, subprocess, sys, tempfile, os, gc, csv, io

# ----------------------------------------------------------------------------
# 1. Text normalisation and overlap metrics
# ----------------------------------------------------------------------------
_PUNCT = set(string.punctuation) | {"\u2018", "\u2019", "\u201c", "\u201d", "\u20b9", "\u2013", "\u2014"}


def normalize(s):
    """SQuAD-style normalisation: lower-case, strip punctuation/articles/extra spaces."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def words(s):
    return normalize(s).split()


def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))


def token_f1(pred, ref):
    p, r = words(pred), words(ref)
    if not p or not r:
        return float(p == r)
    common = {}
    for w in p:
        if w in r:
            common[w] = min(p.count(w), r.count(w))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def _ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def rouge_n(pred, ref, n=1):
    p, r = _ngrams(words(pred), n), _ngrams(words(ref), n)
    if not p or not r:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    from collections import Counter
    cp, cr = Counter(p), Counter(r)
    overlap = sum((cp & cr).values())
    prec, rec = overlap / len(p), overlap / len(r)
    f = 0.0 if overlap == 0 else 2 * prec * rec / (prec + rec)
    return {"p": prec, "r": rec, "f": f}


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    p, r = words(pred), words(ref)
    if not p or not r:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    l = _lcs(p, r)
    prec, rec = l / len(p), l / len(r)
    f = 0.0 if l == 0 else 2 * prec * rec / (prec + rec)
    return {"p": prec, "r": rec, "f": f}


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------------
# 2. Output post-processing (reasoning trace vs final answer)
# ----------------------------------------------------------------------------
SPECIAL_KEEP = ("<think>", "</think>")
SPECIAL_EXTRA = ("<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>", "<pad>",
                 "<\uff5cend\u2581of\u2581sentence\uff5c>", "<\uff5cbegin\u2581of\u2581sentence\uff5c>")


def clean_special_tokens(text, special_tokens=()):
    for st in list(special_tokens) + list(SPECIAL_EXTRA):
        if st in SPECIAL_KEEP:
            continue
        text = text.replace(st, "")
    return text.strip()


def split_reasoning(text, expects_think=False):
    """Return dict(reasoning, final, think_closed).

    expects_think=True (DeepSeek-R1 family): the chat template (or our forced prefix)
    already opened <think>, so the generated text normally contains only '</think>'.
    If no '</think>' was generated, the model never finished thinking (usually the
    token budget ran out): final answer is empty and think_closed=False.
    """
    text = text or ""
    if "</think>" in text:
        head, tail = text.split("</think>", 1)
        return {"reasoning": head.replace("<think>", "").strip(), "final": tail.strip(), "think_closed": True}
    if expects_think:
        return {"reasoning": text.replace("<think>", "").strip(), "final": "", "think_closed": False}
    return {"reasoning": "", "final": text.strip(), "think_closed": None}


# ----------------------------------------------------------------------------
# 3. Answer extraction
# ----------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


def _to_float(tok):
    tok = tok.strip().strip(",.")
    tok = tok.replace(",", "")
    try:
        return float(tok)
    except ValueError:
        return None


def extract_number(final):
    """Priority: 'Final answer: ..' line  >  \\boxed{..}  >  last number in text."""
    if not final:
        return None
    m = re.findall(r"final\s+answer\s*[:\-\u2013]?\s*(.*)", final, flags=re.I)
    cands = []
    if m:
        cands.append(m[-1])
    b = re.findall(r"\\boxed\{([^{}]*)\}", final)
    if b:
        cands.append(b[-1])
    for c in cands:
        nums = _NUM.findall(c)
        if nums:
            v = _to_float(nums[0])
            if v is not None:
                return v
    nums = _NUM.findall(final)
    for tok in reversed(nums):
        v = _to_float(tok)
        if v is not None:
            return v
    return None


def extract_yes_no(final):
    head = (final or "")[:120].lower()
    m = re.search(r"\b(yes|no)\b", head)
    return m.group(1) if m else None


def fmt_num(v):
    if v is None:
        return ""
    return str(int(v)) if float(v).is_integer() else ("%g" % v)


# ----------------------------------------------------------------------------
# 4. Constraint / format checkers
# ----------------------------------------------------------------------------
def count_sentences(text):
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return len(parts)


def check_constraints(text, spec):
    """Generic instruction-following checker.  Returns dict(check -> bool)."""
    out = {}
    t = (text or "").strip()
    lines = [l for l in t.splitlines() if l.strip()]
    if "n_lines" in spec:
        out["n_lines"] = len(lines) == spec["n_lines"]
    if spec.get("no_digits"):
        out["no_digits"] = not re.search(r"\d", t)
    if spec.get("no_bullets"):
        out["no_bullets"] = all(not re.match(r"\s*([-*\u2022]|\d+[.)])\s", l) for l in lines) and bool(lines)
    if "line_max_words" in spec:
        out["line_max_words"] = bool(lines) and all(len(l.split()) <= spec["line_max_words"] for l in lines)
    if "n_sentences" in spec:
        out["n_sentences"] = count_sentences(t) == spec["n_sentences"]
    if "max_words" in spec:
        out["max_words"] = 0 < len(t.split()) <= spec["max_words"]
    if "min_words" in spec:
        out["min_words"] = len(t.split()) >= spec["min_words"]
    for w in spec.get("forbidden_words", []):
        out["no_word_" + w] = not re.search(r"\b" + re.escape(w) + r"\b", t, flags=re.I)
    for w in spec.get("forbidden_substrings", []):
        out["no_substr_" + w] = w.lower() not in t.lower()
    return out


def keyword_groups(text, groups):
    """Each group is a list of alternatives; every group needs >=1 alternative present (prefix match)."""
    low = (text or "").lower()
    hits = []
    for g in groups:
        hits.append(any(re.search(r"\b" + re.escape(a.lower()), low) for a in g))
    return hits


def _strip_fences(text):
    t = (text or "").strip()
    m = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```\s*$", t, flags=re.S)
    return m.group(1).strip() if m else t


def _find_fenced(text):
    m = re.search(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", text or "", flags=re.S)
    return m.group(1).strip() if m else None


def check_structured(text, fmt, spec):
    """fmt in json_object | json_array | csv | md_table | key_value.
    Returns dict(valid, schema_ok, values_ok, clean)."""
    raw = (text or "").strip()
    body = _strip_fences(raw)
    res = {"valid": False, "schema_ok": False, "values_ok": False, "clean": raw == body or raw.startswith("```")}
    if fmt in ("json_object", "json_array"):
        obj = None
        try:
            obj = json.loads(body)
        except Exception:
            fenced = _find_fenced(raw)
            if fenced:
                try:
                    obj = json.loads(fenced)
                    res["clean"] = False
                except Exception:
                    obj = None
        if obj is None:
            return res
        res["valid"] = True
        if fmt == "json_object" and isinstance(obj, dict):
            res["schema_ok"] = set(spec["keys"]).issubset(obj.keys())
            res["values_ok"] = res["schema_ok"] and all(
                str(obj.get(k)).strip().lower() == str(v).strip().lower() for k, v in spec["values"].items())
        elif fmt == "json_array" and isinstance(obj, list):
            n = spec.get("len")
            ok = (n is None or len(obj) == n) and all(isinstance(o, dict) and set(spec["keys"]).issubset(o.keys()) for o in obj)
            res["schema_ok"] = ok
            vs = spec.get("values_seq")
            if ok and vs:
                res["values_ok"] = len(obj) == len(vs) and all(
                    all(str(o.get(k)).strip().lower() == str(v).strip().lower() for k, v in e.items())
                    for o, e in zip(obj, vs))
            elif ok:
                res["values_ok"] = True
        return res
    if fmt == "csv":
        try:
            rows = list(csv.reader(io.StringIO(body)))
            rows = [r for r in rows if any(c.strip() for c in r)]
        except Exception:
            return res
        res["valid"] = len(rows) >= 2 and len({len(r) for r in rows}) == 1
        if res["valid"]:
            hdr = [c.strip().lower() for c in rows[0]]
            res["schema_ok"] = hdr == [c.lower() for c in spec["columns"]]
            exp = [[str(c).lower() for c in r] for r in spec["rows"]]
            got = [[c.strip().lower() for c in r] for r in rows[1:]]
            res["values_ok"] = res["schema_ok"] and got == exp
        return res
    if fmt == "md_table":
        tl = [l.strip() for l in body.splitlines() if l.strip().startswith("|")]
        if len(tl) < 3:
            return res
        cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
        sep_ok = all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) or re.fullmatch(r":?-+:?", c) for c in cells(tl[1]))
        res["valid"] = sep_ok
        if sep_ok:
            hdr = [c.lower() for c in cells(tl[0])]
            res["schema_ok"] = hdr == [c.lower() for c in spec["columns"]]
            clean = lambda c: re.sub(r"[\u20b9$,\s]", "", c.lower())
            got = [[clean(c) for c in cells(l)] for l in tl[2:]]
            exp = [[clean(str(c)) for c in r] for r in spec["rows"]]
            res["values_ok"] = res["schema_ok"] and got == exp
        res["clean"] = all(l.strip().startswith("|") for l in body.splitlines() if l.strip())
        return res
    if fmt == "key_value":
        pairs = {}
        for l in body.splitlines():
            if ":" in l:
                k, v = l.split(":", 1)
                pairs[k.strip().lower()] = v.strip().lower()
        res["valid"] = len(pairs) >= 1
        res["schema_ok"] = set(k.lower() for k in spec["keys"]).issubset(pairs)
        res["values_ok"] = res["schema_ok"] and all(pairs.get(k.lower()) == str(v).lower() for k, v in spec["values"].items())
        return res
    raise ValueError("unknown fmt " + fmt)


_CLARIFY = re.compile(
    r"clarif|which (one|georgia|text|file|item|unit|language|thing)|do you mean|could you (please )?(specify|provide|tell|share)|"
    r"can you (please )?(specify|provide|tell|share)|what (do you mean|are you referring|text|would you like)|not (sure|clear) (what|which)|"
    r"unclear|ambiguous|more (context|information|details)|need (more|to know)|depends on|refer(s|ring)? to|please (provide|specify|share)|"
    r"you('ve| have) not (said|specified|provided|mentioned)|didn't (specify|provide)|missing", re.I)


def run_code_tests(code, func_name, tests, timeout=10):
    """Run model-generated python in a subprocess (timeout) against [(args, expected), ...].
    SECURITY: executes untrusted generated code -- only run inside Colab / a sandbox."""
    script = (code + "\n\nimport json as _j, sys as _s\n_t = _j.loads(" + repr(json.dumps(tests)) + ")\n_r = []\n"
              "for _a, _e in _t:\n    try:\n        _r.append(bool(" + func_name + "(*_a) == _e))\n"
              "    except Exception:\n        _r.append(False)\n_s.stdout.write('@@RESULT@@' + _j.dumps(_r))\n")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gen_code.py")
        with open(p, "w") as f:
            f.write(script)
        try:
            proc = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return {"passed": 0, "total": len(tests), "error": "timeout"}
    m = re.search(r"@@RESULT@@(.*)$", proc.stdout, flags=re.S)
    if not m:
        return {"passed": 0, "total": len(tests), "error": (proc.stderr or "no output")[-300:]}
    r = json.loads(m.group(1))
    return {"passed": sum(r), "total": len(r), "error": None}


def extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.S)
    if m:
        return max(m, key=len).strip()
    return (text or "").strip()


# ----------------------------------------------------------------------------
# 5. Per-sample evaluation
# ----------------------------------------------------------------------------
def evaluate_sample(sample, final_text):
    """Return a dict of per-sample evaluation results (see aggregate() for how they are used)."""
    et, spec = sample["eval_type"], sample.get("eval_spec", {})
    ev = {"correct": False, "extracted": "", "em": None, "f1": None, "compliance": None,
          "structured_valid": None, "rouge": None, "details": {}}
    ref_short = sample.get("short_answer")
    final_text = final_text or ""
    if et == "numeric":
        v = extract_number(final_text)
        ev["extracted"] = fmt_num(v)
        answers = spec.get("answers", [spec.get("answer")])
        tol = spec.get("tol", 0.01)
        ev["correct"] = v is not None and any(abs(v - a) <= tol for a in answers)
        ev["em"] = ev["f1"] = float(ev["correct"])
    elif et == "yesno":
        v = extract_yes_no(final_text)
        ev["extracted"] = v or ""
        ev["correct"] = v == spec["answer"]
        ev["em"] = ev["f1"] = float(ev["correct"])
    elif et == "contains_any":
        low = normalize(final_text)
        ev["correct"] = any(re.search(r"\b" + re.escape(normalize(a)) + r"\b", low) for a in spec["answers"])
        ev["extracted"] = final_text.strip()[:200]
        ev["em"] = max(exact_match(final_text, a) for a in spec["answers"])
        ev["f1"] = max(token_f1(final_text, a) for a in spec["answers"])
    elif et == "keywords":
        hits = keyword_groups(final_text, spec["groups"])
        ev["details"]["group_hits"] = hits
        ev["correct"] = all(hits)
        ev["extracted"] = final_text.strip()[:200]
    elif et == "ifcheck":
        chk = check_constraints(final_text, spec)
        ev["details"]["checks"] = chk
        ev["correct"] = bool(chk) and all(chk.values())
        ev["compliance"] = ev["correct"]
        ev["extracted"] = final_text.strip()[:200]
    elif et == "summary":
        chk = check_constraints(final_text, {"max_words": spec["max_words"]})
        hits = keyword_groups(final_text, spec["groups"])
        ref = sample["reference"]
        ev["rouge"] = {"rouge1": rouge_n(final_text, ref, 1)["f"], "rouge2": rouge_n(final_text, ref, 2)["f"],
                       "rougeL": rouge_l(final_text, ref)["f"]}
        ev["details"] = {"checks": chk, "group_hits": hits}
        ev["compliance"] = all(chk.values())
        ev["correct"] = ev["compliance"] and all(hits)
        ev["extracted"] = final_text.strip()[:300]
    elif et == "code":
        code = extract_code(final_text)
        r = run_code_tests(code, spec["func"], spec["tests"])
        ev["details"] = r
        ev["correct"] = r["total"] > 0 and r["passed"] == r["total"]
        ev["extracted"] = code[:300]
    elif et == "structured":
        r = check_structured(final_text, spec["fmt"], spec["spec"])
        ev["details"] = r
        ev["structured_valid"] = bool(r["valid"])
        ev["correct"] = bool(r["valid"] and r["schema_ok"] and r["values_ok"])
        ev["compliance"] = bool(r["clean"] and r["valid"])  # "output only the structure"
        ev["extracted"] = final_text.strip()[:300]
    elif et == "ambiguity":
        clar = bool(_CLARIFY.search(final_text))
        terms = spec.get("all_terms")
        both = bool(terms) and all(t.lower() in final_text.lower() for t in terms)
        ev["details"] = {"asks_or_flags_ambiguity": clar, "covers_all_readings": both}
        ev["correct"] = clar or both
        ev["extracted"] = final_text.strip()[:200]
    else:
        raise ValueError("unknown eval_type " + et)
    if ref_short and ev["em"] is None:
        ev["em"] = exact_match(ev["extracted"], ref_short)
        ev["f1"] = token_f1(ev["extracted"], ref_short)
    ev["reasoning_correct"] = ev["correct"] if sample.get("reasoning") else None
    return ev


# ----------------------------------------------------------------------------
# 6. Running + aggregation
# ----------------------------------------------------------------------------
def run_samples(samples, generate_fn, expects_think=False, log=print, tag=""):
    """generate_fn(prompt) -> dict(text, input_tokens, new_tokens, seconds, hit_max_tokens, mem=dict(...))."""
    records = []
    for i, s in enumerate(samples, 1):
        g = generate_fn(s["prompt"])
        parts = split_reasoning(g["text"], expects_think=expects_think)
        ev = evaluate_sample(s, parts["final"])
        rec = {k: s.get(k) for k in ("id", "split", "category", "difficulty", "prompt", "reference", "criteria")}
        rec.update({
            "variation": s.get("variation"), "expected_behaviour": s.get("expected_behaviour"),
            "raw_output": g["text"], "final_output": parts["final"], "reasoning_chars": len(parts["reasoning"]),
            "think_closed": parts["think_closed"], "input_tokens": g["input_tokens"], "new_tokens": g["new_tokens"],
            "seconds": g["seconds"], "tokens_per_sec": g["new_tokens"] / g["seconds"] if g["seconds"] > 0 else None,
            "hit_max_tokens": g["hit_max_tokens"], "mem": g.get("mem"), "eval": ev,
        })
        rec["observation"] = auto_observation(rec)
        records.append(rec)
        log("[%s %d/%d] %s %-5s %.1fs %d tok  %s" % (tag, i, len(samples), s["id"], "PASS" if ev["correct"] else "FAIL",
                                                      g["seconds"], g["new_tokens"], (parts["final"] or g["text"])[:60].replace("\n", " ")))
    return records


def auto_observation(rec):
    ev = rec["eval"]
    bits = ["PASS" if ev["correct"] else "FAIL", "%d new tokens" % rec["new_tokens"], "%.1fs" % rec["seconds"]]
    if rec.get("hit_max_tokens"):
        bits.append("stopped at max_new_tokens limit")
    if rec.get("think_closed") is False:
        bits.append("reasoning trace never closed (no </think>) -> no final answer")
    if not ev["correct"] and ev.get("extracted"):
        bits.append("extracted=%r" % ev["extracted"][:60])
    return "; ".join(bits)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(records):
    n = len(records)
    k = sum(1 for r in records if r["eval"]["correct"])
    lo, hi = wilson_interval(k, n)
    secs = [r["seconds"] for r in records]
    tps = [r["tokens_per_sec"] for r in records if r["tokens_per_sec"]]
    cats = {}
    for r in records:
        c = cats.setdefault(r["category"], {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += int(r["eval"]["correct"])
    for c in cats.values():
        c["accuracy"] = c["correct"] / c["n"]
    rs = [r["eval"]["rouge"] for r in records if r["eval"]["rouge"]]
    reasoning = [r["eval"]["reasoning_correct"] for r in records if r["eval"]["reasoning_correct"] is not None]
    comp = [r["eval"]["compliance"] for r in records if r["eval"]["compliance"] is not None]
    sv = [r["eval"]["structured_valid"] for r in records if r["eval"]["structured_valid"] is not None]
    tot_new, tot_sec = sum(r["new_tokens"] for r in records), sum(secs)
    peaks = [r["mem"]["peak_allocated_gb"] for r in records if r.get("mem") and r["mem"].get("peak_allocated_gb") is not None]
    return {
        "n": n, "n_correct": k, "accuracy": k / n if n else None, "accuracy_ci95": [lo, hi],
        "exact_match": _mean([r["eval"]["em"] for r in records]),
        "token_f1": _mean([r["eval"]["f1"] for r in records]),
        "reasoning_correctness": _mean([float(x) for x in reasoning]), "n_reasoning": len(reasoning),
        "instruction_compliance": _mean([float(x) for x in comp]), "n_compliance": len(comp),
        "structured_validity": _mean([float(x) for x in sv]), "n_structured": len(sv),
        "rouge1": _mean([x["rouge1"] for x in rs]), "rouge2": _mean([x["rouge2"] for x in rs]),
        "rougeL": _mean([x["rougeL"] for x in rs]), "n_rouge": len(rs),
        "per_category": cats,
        "time_mean_s": _mean(secs), "time_median_s": statistics.median(secs) if secs else None,
        "time_p90_s": sorted(secs)[int(0.9 * (len(secs) - 1))] if secs else None, "time_total_s": tot_sec,
        "tokens_per_sec_aggregate": tot_new / tot_sec if tot_sec > 0 else None,
        "tokens_per_sec_median": statistics.median(tps) if tps else None,
        "mean_new_tokens": tot_new / n if n else None,
        "peak_gpu_mem_gb_max": max(peaks) if peaks else None,
        "n_hit_max_tokens": sum(1 for r in records if r.get("hit_max_tokens")),
        "n_unfinished_thinking": sum(1 for r in records if r.get("think_closed") is False),
    }


_FAIL_TYPE = {
    "numeric": "Mathematical / multi-step reasoning error", "yesno": "Logical reasoning error",
    "contains_any": "Factual inaccuracy (possible hallucination)", "keywords": "Analytical / content-coverage failure",
    "ifcheck": "Instruction-following failure", "summary": "Summarisation constraint or coverage failure",
    "code": "Coding error", "structured": "Incorrect structured output", "ambiguity": "Ambiguous-prompt mis-handling",
}


def build_failure_table(records_with_types):
    """records_with_types: list of (record, eval_type). Returns list of failure dicts (observed failures only)."""
    out = []
    for rec, et in records_with_types:
        ev = rec["eval"]
        if ev["correct"]:
            continue
        ftype = None
        if rec.get("think_closed") is False:
            ftype = "Excessive response / truncation (reasoning never finished within token budget)"
        elif rec.get("hit_max_tokens"):
            ftype = "Excessive response / truncation (hit max_new_tokens)"
        else:
            ftype = _FAIL_TYPE.get(et, "Other")
        out.append({
            "id": rec["id"], "category": rec["category"], "input": rec["prompt"],
            "expected_behaviour": rec.get("expected_behaviour") or ("reference: %s" % rec.get("reference")),
            "actual_output": (rec["final_output"] or rec["raw_output"])[:800], "failure_type": ftype,
            "explanation": "Automatic evaluator marked this FAIL (%s). Extracted=%r. %s" % (
                json.dumps(ev.get("details", {}), default=str)[:240], ev.get("extracted", "")[:80],
                "Manual review recommended before citing in the report."),
            "observed": True,
        })
    return out


def consistency_test(generate_fn, sample, seeds=(0, 1, 2), expects_think=False):
    outs = []
    for sd in seeds:
        g = generate_fn(sample["prompt"], seed=sd)
        parts = split_reasoning(g["text"], expects_think=expects_think)
        ev = evaluate_sample(sample, parts["final"])
        outs.append({"seed": sd, "extracted": ev["extracted"], "correct": ev["correct"], "new_tokens": g["new_tokens"]})
    answers = {o["extracted"] for o in outs}
    return {"id": sample["id"], "runs": outs, "n_distinct_answers": len(answers), "all_correct": all(o["correct"] for o in outs)}


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# 7. HF / torch runtime (Colab only)
# ----------------------------------------------------------------------------
def env_report():
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_total_gb"] = round(p.total_memory / 1024 ** 3, 2)
            try:
                info["bf16_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
            except TypeError:
                info["bf16_native"] = bool(torch.cuda.get_device_capability(0)[0] >= 8)
    except Exception as e:  # pragma: no cover
        info["torch_error"] = str(e)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    for pkg in ("accelerate", "bitsandbytes", "huggingface_hub", "tokenizers"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = None
    try:
        import psutil
        info["system_ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 2)
    except Exception:
        pass
    return info


def gpu_mem():
    import torch
    if not torch.cuda.is_available():
        return {"allocated_gb": None, "reserved_gb": None, "peak_allocated_gb": None, "peak_reserved_gb": None}
    g = 1024 ** 3
    return {"allocated_gb": torch.cuda.memory_allocated() / g, "reserved_gb": torch.cuda.memory_reserved() / g,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / g, "peak_reserved_gb": torch.cuda.max_memory_reserved() / g}


def load_hf_model(model_id, precision, token=None, revision=None, log=print):
    """precision in {'bf16','fp16','8bit','4bit'}.  Download time and load time are measured separately."""
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from huggingface_hub import snapshot_download
    t0 = time.perf_counter()
    last_err = None
    for attempt in range(1, 4):  # retry flaky downloads
        try:
            snapshot_download(model_id, token=token, revision=revision)
            last_err = None
            break
        except Exception as e:
            last_err = e
            name = type(e).__name__
            if name in ("GatedRepoError", "RepositoryNotFoundError") or "401" in str(e) or "403" in str(e):
                raise  # authentication/licence problems will not fix themselves
            log("download attempt %d failed (%s); retrying in %ds" % (attempt, name, 5 * attempt))
            time.sleep(5 * attempt)
    if last_err is not None:
        raise last_err
    download_s = time.perf_counter() - t0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, token=token, revision=revision)
    tok_s = time.perf_counter() - t1
    kw = {"token": token, "revision": revision, "device_map": {"": 0} if torch.cuda.is_available() else "cpu"}
    try:
        from packaging.version import parse
        dtype_key = "dtype" if parse(transformers.__version__.split("+")[0]) >= parse("4.56.0") else "torch_dtype"
    except Exception:
        dtype_key = "torch_dtype"
    if precision == "bf16":
        kw[dtype_key] = torch.bfloat16
    elif precision == "fp16":
        kw[dtype_key] = torch.float16
    elif precision == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kw[dtype_key] = torch.float16
    elif precision == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        kw[dtype_key] = torch.float16
    else:
        raise ValueError(precision)
    t2 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t2
    mem = gpu_mem()
    n_params = sum(p.numel() for p in model.parameters())
    info = {"model_id": model_id, "revision": revision, "precision": precision, "download_s": download_s,
            "tokenizer_init_s": tok_s, "model_load_s": load_s, "total_load_s": tok_s + load_s, "mem_after_load": mem,
            "n_parameters_loaded": n_params, "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
            "vocab_size": getattr(model.config, "vocab_size", None)}
    log("loaded %s [%s]: download %.1fs, load %.1fs, GPU alloc %s GB" % (
        model_id, precision, download_s, load_s, None if mem["allocated_gb"] is None else round(mem["allocated_gb"], 2)))
    return model, tok, info


def make_generate_fn(model, tok, gen_kwargs, force_think_prefix=False, default_seed=1234):
    """Returns fn(prompt, seed=None, **gen_overrides) -> dict with timing + memory.
    Uses the tokenizer's own chat template (single user turn, NO system prompt)."""
    import torch
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)

    def fn(prompt, seed=None, **over):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if force_think_prefix and not text.rstrip().endswith("<think>"):
            text = text + "<think>\n"
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        kw = dict(gen_kwargs)
        kw.update(over)
        torch.manual_seed(default_seed if seed is None else seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[0, enc["input_ids"].shape[1]:]
        raw = tok.decode(new, skip_special_tokens=False)
        raw = clean_special_tokens(raw, tok.all_special_tokens)
        last = int(new[-1].item()) if new.shape[0] else None
        hit_max = bool(new.shape[0] >= kw.get("max_new_tokens", 10 ** 9) and last not in eos_ids)
        return {"text": raw, "input_tokens": int(enc["input_ids"].shape[1]), "new_tokens": int(new.shape[0]), "seconds": dt,
                "hit_max_tokens": hit_max, "mem": gpu_mem(), "prompt_rendered": text if over.get("_return_prompt") else None}
    return fn


def warmup(gen_fn):
    gen_fn("Say hello.", max_new_tokens=8)


def release(*objs):
    import torch
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_long_prompt(tok, target_tokens, needle_code="K7-4921", question="What is the secret access code mentioned in the log? Reply with the code only."):
    """Needle-in-a-haystack style prompt of ~target_tokens tokens (needle placed in the middle)."""
    filler, i = [], 0
    while True:
        i += 1
        filler.append("Log entry %d: station %d recorded %d millimetres of rain and a wind speed of %d km/h." % (i, i % 41 + 1, i % 17 + 2, i % 23 + 5))
        if i % 25 == 0:
            if len(tok("\n".join(filler), add_special_tokens=False)["input_ids"]) >= target_tokens:
                break
    mid = len(filler) // 2
    filler.insert(mid, "NOTE: the secret access code is %s." % needle_code)
    text = "Read the following weather log.\n\n" + "\n".join(filler) + "\n\n" + question
    return text, needle_code


def resource_precision_experiment(model_id, token, precisions, samples, gen_kwargs, expects_think, force_think_prefix, log=print):
    """Load the SAME checkpoint at several precisions; compare load time, memory, speed and quality on `samples`."""
    out = []
    for p in precisions:
        row = {"precision": p, "status": "OK"}
        model = tok = None
        try:
            model, tok, info = load_hf_model(model_id, p, token=token, log=log)
            gen = make_generate_fn(model, tok, gen_kwargs, force_think_prefix=force_think_prefix)
            warmup(gen)
            recs = run_samples(samples, gen, expects_think=expects_think, log=log, tag="res-" + p)
            agg = aggregate(recs)
            row.update({"load": info, "metrics": agg, "peak_mem_gb": agg["peak_gpu_mem_gb_max"],
                        "predictions": [{"id": r["id"], "correct": r["eval"]["correct"], "extracted": r["eval"]["extracted"],
                                         "seconds": r["seconds"], "new_tokens": r["new_tokens"]} for r in recs]})
        except Exception as e:  # OOM, unsupported dtype, missing bitsandbytes ...
            row["status"] = "FAILED: %s: %s" % (type(e).__name__, str(e)[:300])
            log(row["status"])
        finally:
            try:
                release(model, tok)
            except Exception:
                pass
        out.append(row)
    return out


def resource_input_length_experiment(model, tok, gen_fn, lengths, max_ctx=None, expects_think=False, max_new_tokens=None, log=print):
    rows = []
    for L in lengths:
        if max_ctx and L > max_ctx - 600:
            rows.append({"target_tokens": L, "status": "SKIPPED: exceeds context limit %s" % max_ctx})
            continue
        prompt, needle = build_long_prompt(tok, L)
        sample = {"id": "LEN%d" % L, "eval_type": "contains_any", "eval_spec": {"answers": [needle]}, "reference": needle}
        try:
            kw = {"max_new_tokens": max_new_tokens} if max_new_tokens else {}
            g = gen_fn(prompt, **kw)
            parts = split_reasoning(g["text"], expects_think=expects_think)
            ev = evaluate_sample(sample, parts["final"])
            rows.append({"target_tokens": L, "actual_input_tokens": g["input_tokens"], "seconds": g["seconds"], "new_tokens": g["new_tokens"],
                         "peak_mem_gb": g["mem"]["peak_allocated_gb"], "found_needle": ev["correct"], "answer_head": parts["final"][:80],
                         "status": "OK"})
        except Exception as e:
            rows.append({"target_tokens": L, "status": "FAILED: %s: %s" % (type(e).__name__, str(e)[:200])})
        log(rows[-1])
    return rows


def resource_generation_length_experiment(gen_fn, lengths, prompt="Write a detailed explanation of how photosynthesis works, covering light reactions, the Calvin cycle and their locations in the cell.", log=print):
    rows = []
    for L in lengths:
        g = gen_fn(prompt, max_new_tokens=L)
        rows.append({"max_new_tokens": L, "new_tokens": g["new_tokens"], "seconds": g["seconds"],
                     "tokens_per_sec": g["new_tokens"] / g["seconds"], "peak_mem_gb": g["mem"]["peak_allocated_gb"]})
        log(rows[-1])
    return rows
