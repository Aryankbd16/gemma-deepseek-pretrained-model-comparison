# Verified facts used across notebooks and PDFs. Every item cites where it was verified in this session:
#   [GP] Gemma paper arXiv:2403.08295 (fetched pdf v2 text)        [GC] HF model card google/gemma-2b-it
#   [GR] google-deepmind/gemma repo (HEAD + commit 59ffafe, 2024)  [DP] DeepSeek-R1 paper arXiv:2501.12948 (html v1)
#   [DR] deepseek-ai/DeepSeek-R1 repo README (HEAD 0cf7856)        [DC] HF file listing of the distill checkpoint

RESEARCH_ROWS = [
    ["Research objectives",
     "Release lightweight open models (2B, 7B) built from Gemini research/technology, with both pretrained and instruction-tuned checkpoints and an open inference codebase, and analyse safety/responsible release. [GP]",
     "Test whether reasoning can be incentivised by *pure reinforcement learning* without supervised data (R1-Zero), fix its readability/language-mixing problems with a multi-stage pipeline (R1), and distil the reasoning ability into small dense models. [DP]"],
    ["Architecture",
     "Decoder-only Transformer, 8,192-token training context. 2B: 18 layers, d_model 2048, FFN hidden 32,768, 8 heads, 1 KV head (multi-query attention), head size 256; 7B: 28 layers, d_model 3072, 16 heads/16 KV heads (multi-head). RoPE, approximate-GeGLU, RMSNorm, shared input/output embeddings, 256k vocabulary. [GP Tables 1-2]",
     "R1-Zero / R1 are trained on DeepSeek-V3-Base, a Mixture-of-Experts model (671B total, 37B activated; 128K context per the repo README); the architecture is described in the DeepSeek-V3 report, not in the R1 paper. The distilled models are *dense Qwen2.5 / Llama* networks - the 7B uses Qwen2.5-Math-7B as base. [DP, DR]"],
    ["Training methodology",
     "Pretraining only (then alignment): 3T (2B) and 6T (7B) tokens of mostly English web text, mathematics and code; TPUv5e (512 chips for 2B, 4096 for 7B), JAX + Pathways, staged data mixture with filtering of unsafe/personal data and evaluation-set decontamination. [GP]",
     "No new pretraining. Post-training of the V3 base: R1-Zero = RL directly on the base; R1 = 4 stages (cold-start SFT -> reasoning RL -> rejection-sampling SFT -> RL for all scenarios). Distilled models: plain SFT of open base models. [DP]"],
    ["Instruction tuning",
     "SFT on English, text-only synthetic + human prompt-response pairs, then RLHF; role/turn control tokens (<start_of_turn>, <end_of_turn>) used at both training and inference. [GP]",
     "Cold-start: thousands of long chain-of-thought examples. Later SFT: ~600k reasoning samples (rejection sampling from the RL checkpoint) + ~200k non-reasoning samples = ~800k, 2 epochs on V3-Base. Distilled checkpoints are fine-tuned on those 800k samples. [DP §2.3-2.4]"],
    ["Reinforcement learning",
     "RLHF: Bradley-Terry reward model from human preference pairs; policy optimised with 'a novel reinforcement learning algorithm' (not specified in the paper); LLM-judge side-by-side evaluations used for tuning and to mitigate reward hacking. [GP]",
     "GRPO (Group Relative Policy Optimization): no critic; advantage A_i = (r_i - mean(r))/std(r) over a group of G sampled outputs; clipped ratio objective with KL penalty to a reference policy. Rewards: rule-based accuracy + format (<think> tags), language-consistency reward; neural reward models only in the final all-scenario stage. *Not applied to the distilled checkpoints.* [DP §2.2, 2.3]"],
    ["Knowledge distillation",
     "Not part of the method described in the Gemma 1 paper.",
     "Central contribution: 800k R1-generated samples used for SFT of Qwen2.5-Math-1.5B/7B, Qwen2.5-14B/32B, Llama-3.1-8B and Llama-3.3-70B-Instruct. Finding: distilling beat running large-scale RL on a 32B base (Table 6). [DP §2.4, §4.1]"],
    ["Evaluation methodology",
     "Automated academic benchmarks (MMLU 5-shot, GSM8K maj@1, HumanEval pass@1, MBPP, BBH, ...), independent HF evaluation, human side-by-side win rates vs Mistral-7B-Instruct v0.2 (~1000 instruction + ~400 safety prompts), safety benchmarks (RealToxicity, BBQ, TruthfulQA ...), memorisation tests. [GP]",
     "Reasoning/knowledge/coding benchmarks (AIME 2024, MATH-500, GPQA-Diamond, LiveCodeBench, Codeforces, MMLU, SimpleQA, AlpacaEval 2.0, ArenaHard ...) with sampling (T=0.6, top-p 0.95), up to 64 samples/question for pass@1, cons@64, max 32,768 generated tokens; zero-shot prompts (few-shot hurts). [DP §3]"],
    ["Contributions",
     "Open pretrained + IT checkpoints and code at 2B/7B; state-of-the-art for size at release on 11/18 text tasks; detailed safety and memorisation analysis. [GP]",
     "First open research validating RL-only incentivisation of reasoning; GRPO-based pipeline; open R1, R1-Zero and six distilled dense models; report of unsuccessful attempts (PRM, MCTS). [DP]"],
    ["Advantages",
     "Small enough for laptops/Colab; clean control-token format; good maths/code for its size (paper); permissive-for-research open weights (custom licence).",
     "Very strong maths/code reasoning for 7B (paper: MATH-500 92.8, AIME24 55.5); emergent self-verification/reflection; MIT licence; usable like any Qwen model."],
    ["Limitations",
     "English-centric, text-only, 8k context; may produce incorrect facts; weak absolute reasoning at 2B; residual bias/toxicity/PII risks - needs use-case safety testing. [GP, GC]",
     "Prompt-sensitive (few-shot degrades), language mixing beyond EN/ZH, weaker than V3 on function calling/JSON/multi-turn (full R1), long latency from reasoning traces, SE tasks little improved; distilled models received no RL. [DP §5]"],
]

TECH_ROWS = [
    ["Checkpoint", "google/gemma-2b-it (Gemma 1.0, 2B instruction-tuned)", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    ["Parameter count", "~2.51B (524,550,144 embedding + 1,981,884,416 non-embedding) [GP Table 2]", "7B class (HF pages show ~7.6B-8B; the notebook prints the exact count) [DC]"],
    ["Model architecture", "18-layer decoder, MQA (1 KV head), RoPE, GeGLU, RMSNorm, tied embeddings [GP]", "Dense Qwen2-family decoder (HF tag `qwen2`), base Qwen2.5-Math-7B; SFT-distilled from R1 [DR, DC]"],
    ["Tokenizer", "SentencePiece subset of Gemini tokenizer, 256,128 entries, digit splitting, byte fallback [GP]", "Qwen-based tokenizer, 'slightly changed' by DeepSeek [DR]; vocab size printed at runtime; adds <think></think> handling"],
    ["Context length", "Trained at 8,192 tokens [GP]", "Value in checkpoint config.json (printed at runtime); paper evaluates with up to 32,768 generated tokens [DP]"],
    ["Dependencies", "Colab: torch, transformers, accelerate (bitsandbytes for 4/8-bit). Official repo: JAX/Flax (+ sentencepiece, kagglehub ...); current main needs Python >= 3.12 [GR]", "torch, transformers, accelerate (bitsandbytes for 4/8-bit); README serves with vLLM or SGLang [DR]"],
    ["Inference framework used here", "Hugging Face Transformers (PyTorch). Official JAX sampler (Gemma-1 era `gemma/sampler.py`, greedy argmax) described but not run in Colab", "Hugging Face Transformers (PyTorch)"],
    ["Model loading", "AutoModelForCausalLM.from_pretrained('google/gemma-2b-it'); gated -> accept licence + HF token [GC]", "AutoModelForCausalLM.from_pretrained(...); public, no token needed [DC]"],
    ["Prompt formatting", "<bos><start_of_turn>user\\n{text}<end_of_turn>\\n<start_of_turn>model\\n ; no system role [GC, GP]", "Chat template of the checkpoint; **no system prompt**; response should begin with '<think>\\n' [DR]"],
    ["Recommended decoding", "Model card examples use default greedy generate", "temperature 0.5-0.7 (0.6), top-p 0.95 in paper evaluation; avoid greedy (repetition) [DR, DP]"],
    ["Hardware requirements", "Repo README: 8 GB+ GPU RAM for 2B, 24 GB+ for 7B [GR]; weights ~5 GB in bf16 (2.5B x 2 bytes, estimate)", "Weights 8.61 GB + 6.62 GB = 15.2 GB safetensors [DC] -> does not fit a 15 GB T4 in fp16 with activations; 4-bit needs roughly 5-6 GB (estimate; measured in notebook)"],
    ["Precision / quantisation", "Native bf16; an fp16 revision exists; int8/int4 via bitsandbytes shown on the card [GC]", "Shipped in BF16 (file sizes consistent with 2 bytes/param); we use 4-bit NF4 on T4, 8-bit and fp16 where memory allows"],
    ["Memory requirements", "Measured by notebook (allocated / reserved / peak)", "Measured by notebook (allocated / reserved / peak)"],
    ["Output style", "Direct answer", "Long chain-of-thought inside <think>...</think> then answer"],
    ["Licence", "Gemma Terms of Use (custom, gated) [GC]", "MIT (weights + code); base model Qwen2.5 is Apache-2.0 [DR]"],
]

LIT_ROWS = [
    ["MMLU (5-shot)", "42.3", "64.3", "not reported", "Gemma paper Table 6 (pretrained, not -it)"],
    ["GSM8K (maj@1)", "17.7", "46.4", "not reported", "Gemma paper Table 6"],
    ["MATH", "11.8 (4-shot)", "24.3 (4-shot)", "92.8 (MATH-500 pass@1)", "Different benchmark subsets and protocols"],
    ["HumanEval pass@1", "22.0", "32.3", "not reported", "Gemma paper Table 6"],
    ["MBPP (3-shot)", "29.2", "44.4", "not reported", "Gemma paper Table 6"],
    ["AIME 2024 pass@1", "not reported", "not reported", "55.5", "DeepSeek paper Table 5 (T=0.6, top-p 0.95, 32k tokens)"],
    ["GPQA-Diamond pass@1", "not reported", "not reported", "49.1", "DeepSeek paper Table 5"],
    ["LiveCodeBench pass@1", "not reported", "not reported", "37.6", "DeepSeek paper Table 5"],
    ["Codeforces rating", "not reported", "not reported", "1189", "DeepSeek paper Table 5"],
    ["Human win rate vs Mistral-7B-Instruct v0.2", "IF 41.6% / safety 56.5% (Gemma 1.0 IT 2B)", "IF 51.7% / safety 58% (Gemma 1.0 IT 7B)", "not applicable", "Gemma paper Table 9"],
]
