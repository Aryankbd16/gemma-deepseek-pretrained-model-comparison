import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_notebooks as mn
from content_data import RESEARCH_ROWS, TECH_ROWS, LIT_ROWS
out = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/build/nb_out"
os.makedirs(out, exist_ok=True)
nbs = {"G19_Gemma_Implementation.ipynb": mn.model_notebook(mn.G19), "G20_DeepSeek_R1_Implementation.ipynb": mn.model_notebook(mn.G20),
       "G19_G20_Comparison.ipynb": mn.comparison_notebook(RESEARCH_ROWS, TECH_ROWS, LIT_ROWS)}
for n, nb in nbs.items():
    with open(os.path.join(out, n), "w", encoding="utf-8") as f: json.dump(nb, f, indent=1, ensure_ascii=False)
    print(n, len(nb["cells"]), "cells")
