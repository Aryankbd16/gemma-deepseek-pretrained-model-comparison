# Evaluation dataset card

- 30 hand-authored samples, 10 categories x 3; dev = items 1-2 per category (20), test = item 3 per category (10).
- Fields: id, split, category, difficulty, prompt, reference, criteria (+ eval_type/eval_spec in the JSON used by the notebooks).
- Plus a 26-prompt robustness set in 5 variations (in the JSON).
- Purpose: evaluation only. No training or fine-tuning of any model. Test prompts were not used to tune anything.
- Authored for this lab; not drawn from public benchmarks, so contamination with training data is unlikely but not provable.
- Limitation: tiny; confidence intervals are wide.
