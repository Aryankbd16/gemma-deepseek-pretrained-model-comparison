"""Glue-code test: executes every code cell of the 3 notebooks against STUBS (fake torch / fake model) in a scratch dir.
import matplotlib; matplotlib.use("Agg")
Stub outputs are throw-away; nothing from here is shipped."""
import json, sys, os, types, tempfile, re, shutil, builtins
NB = sys.argv[1]; TMP = tempfile.mkdtemp(prefix="nbtest_")
def cells(fn):
    nb = json.load(open(os.path.join(NB, fn)));  return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
def prep(src):
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
    src = re.sub(r"from IPython\.display import .*", "Markdown = HTML = lambda x: x", src)
    return src.replace('"/content', '"' + TMP).replace("'/content", "'" + TMP).replace("f\"/content", "f\"" + TMP)
# 1. compile everything
for fn in sorted(os.listdir(NB)):
    if fn.endswith(".ipynb"):
        for i, s in enumerate(cells(fn)):
            compile(prep(s), f"{fn}[{i}]", "exec")
        print("compiled OK:", fn)
# 2. fake modules
class Cuda:
    OutOfMemoryError = type("OOM", (RuntimeError,), {})
    is_available = staticmethod(lambda: True)
    synchronize = staticmethod(lambda *a: None); empty_cache = staticmethod(lambda: None)
torch = types.ModuleType("torch"); torch.cuda = Cuda; torch.manual_seed = lambda s: None; torch.bfloat16 = "bf16"; torch.float16 = "fp16"
sys.modules["torch"] = torch
hub = types.ModuleType("huggingface_hub"); hub.login = lambda **k: None; hub.hf_hub_download = lambda *a, **k: "x"; sys.modules["huggingface_hub"] = hub
os.environ["HF_TOKEN"] = "dummy"
sys.path.insert(0, TMP); os.chdir(TMP)
import importlib
sys.path.insert(0, "/home/claude/build"); import ca_common as ca   # real module (stubs patched in below)
class Tok:
    bos_token, eos_token, pad_token, pad_token_id, eos_token_id = "<bos>", "<eos>", "<pad>", 0, 1
    all_special_tokens = ["<bos>", "<eos>", "<pad>"]
    def apply_chat_template(self, m, tokenize=False, add_generation_prompt=True): return "<bos><start_of_turn>user\n" + m[0]["content"] + "<end_of_turn>\n<start_of_turn>model\n"
    def __call__(self, t, **k): return {"input_ids": list(range(len(t.split())))}
    def convert_ids_to_tokens(self, ids): return ["t%d" % i for i in ids]
    def tokenize(self, t): return t.split()
    def convert_tokens_to_ids(self, t): return 5
    def __len__(self): return 256128
class GC: eos_token_id = [1, 107]
class Model:
    generation_config = GC(); config = types.SimpleNamespace(max_position_embeddings=8192, vocab_size=256128); device = "cpu"
def fake_env(): return {"python": "3", "torch": "stub", "gpu": "STUB-GPU", "gpu_total_gb": 15.0, "bf16_native": False, "system_ram_gb": 12}
def fake_load(model_id, precision, token=None, revision=None, log=print):
    return Model(), Tok(), {"model_id": model_id, "precision": precision, "download_s": 1.0, "tokenizer_init_s": .1, "model_load_s": 2.0, "total_load_s": 2.1,
        "mem_after_load": {"allocated_gb": 1.0}, "n_parameters_loaded": 2_500_000_000, "max_position_embeddings": 8192, "vocab_size": 256128}
def fake_gen_factory(model, tok, gen_kwargs, force_think_prefix=False, default_seed=0):
    def fn(prompt, seed=None, **over):
        L = over.get("max_new_tokens", 50)
        txt = "STUB OUTPUT. Final answer: 36"
        if force_think_prefix: txt = "reasoning...</think>\n" + txt
        return {"text": txt, "input_tokens": len(prompt.split()), "new_tokens": min(L, 20), "seconds": 0.01 + 0.0001 * len(prompt), "hit_max_tokens": False,
                "mem": {"allocated_gb": 1, "reserved_gb": 1.2, "peak_allocated_gb": 1.5, "peak_reserved_gb": 1.6}}
    return fn
ca.env_report = fake_env; ca.load_hf_model = fake_load; ca.make_generate_fn = fake_gen_factory; ca.release = lambda *a: None
class Ctx(dict): pass
import builtins
def run_nb(fn):
    g = {"__name__": "__main__", "display": lambda x=None: None}
    for i, s in enumerate(cells(fn)):
        try: exec(compile(prep(s), f"{fn}[{i}]", "exec"), g)
        except SystemExit as e: print("SystemExit", e); raise
        except Exception as e:
            print("FAILED cell", i, "of", fn); print(s[:400]); raise
    return g
for fn in ("G19_Gemma_Implementation.ipynb", "G20_DeepSeek_R1_Implementation.ipynb"):
    g = run_nb(fn); print("EXECUTED (stub):", fn, "| results files:", len(os.listdir(g["RESULTS_DIR"])))
# comparison on stub results
res = {}
for k, folder in (("G19", "G19_Gemma"), ("G20", "G20_DeepSeek")):
    assert os.path.exists(f"{TMP}/Pretrained_Model_CA/{folder}/results/{k}_results.json")
g = run_nb("G19_G20_Comparison.ipynb"); print("EXECUTED (stub results):", "G19_G20_Comparison.ipynb", "| HAVE =", g["HAVE"], "| charts:", sorted(os.listdir(g["CHARTS"])))
print("xlsx:", os.path.exists(g["RES"] + "/Experimental_Results.xlsx"))
# comparison with NO results -> must not crash and must say NOT RUN
shutil.rmtree(f"{TMP}/Pretrained_Model_CA/G19_Gemma"); shutil.rmtree(f"{TMP}/Pretrained_Model_CA/G20_DeepSeek")
for f_ in os.listdir(TMP):
    if f_.endswith(("_results.json", "_results.zip")): os.remove(os.path.join(TMP, f_))
g = run_nb("G19_G20_Comparison.ipynb"); print("no-results run OK, HAVE =", g["HAVE"])
shutil.rmtree(TMP)
