"""Authoring of the common evaluation dataset (30 samples) and the robustness suite.
All prompts/references are written by us for this assignment (not copied from benchmarks)."""
import json

COT = " Think step by step, then end with a line of the form 'Final answer: <number>'."
SHORT = " Answer with just the name or number, no explanation."


def S(id, split, category, difficulty, prompt, reference, eval_type, eval_spec, criteria, reasoning=False, short_answer=None, expected=None):
    return {"id": id, "split": split, "category": category, "difficulty": difficulty, "prompt": prompt,
            "reference": reference, "eval_type": eval_type, "eval_spec": eval_spec, "criteria": criteria,
            "reasoning": reasoning, "short_answer": short_answer, "expected_behaviour": expected or ("Reference answer: %s" % reference)}


P_SUM1 = ("Bees visit flowers to collect nectar and pollen for food. As they move from flower to flower, grains of pollen stick to their "
          "bodies and are carried to other plants, allowing those plants to produce seeds and fruit. Because many crops depend on this "
          "process, a decline in bee populations can reduce harvests and raise food prices.")
P_SUM2 = ("Many companies now allow employees to work from home several days a week. Supporters say this saves commuting time and lets "
          "people balance family duties, while critics worry that it weakens team bonds and makes it harder for new staff to learn from "
          "colleagues. Surveys of workers often show high satisfaction, but managers remain divided about its long-term effect on productivity.")
P_SUM3 = ("Cities are often several degrees warmer than the countryside around them, an effect called the urban heat island. Dark roads and "
          "roofs absorb sunlight, concrete stores heat overnight, and there are fewer trees to provide shade and cool the air through "
          "evaporation. Planting trees, installing reflective roofs and adding green spaces can reduce the effect and lower summer energy use.")
SUMQ = "Summarize the following passage in one sentence of at most 25 words.\n\nPassage: "

DATASET = [
    # ---- General knowledge
    S("GK01", "dev", "general_knowledge", "easy", "What is the chemical symbol for gold?" + SHORT, "Au", "contains_any", {"answers": ["Au"]},
      "Correct symbol 'Au'", short_answer="Au"),
    S("GK02", "dev", "general_knowledge", "easy", "What is the capital city of Australia?" + SHORT, "Canberra", "contains_any", {"answers": ["Canberra"]},
      "Correct capital (common distractors: Sydney, Melbourne)", short_answer="Canberra"),
    S("GK03", "test", "general_knowledge", "medium", "In which year did the Chernobyl nuclear power plant accident occur?" + SHORT, "1986", "contains_any",
      {"answers": ["1986"]}, "Correct year 1986", short_answer="1986"),
    # ---- Instruction following
    S("IF01", "dev", "instruction_following", "easy", "List exactly three fruits, one per line, with no numbering, no bullets and no extra text.", "Three lines, each a fruit name",
      "ifcheck", {"n_lines": 3, "no_digits": True, "no_bullets": True, "line_max_words": 3},
      "Exactly 3 non-empty lines; no digits/bullets; <=3 words per line", expected="Exactly three short lines containing fruit names, no numbering/bullets/extra text"),
    S("IF02", "dev", "instruction_following", "medium", "Write exactly one sentence about the ocean without using the word 'the'.", "One sentence, no word 'the'",
      "ifcheck", {"n_sentences": 1, "forbidden_words": ["the"]}, "Exactly 1 sentence; word 'the' absent", expected="A single sentence about the ocean that never uses the word 'the'"),
    S("IF03", "test", "instruction_following", "medium", "Write exactly two sentences explaining why sleep is important. Do not use the word 'health' (or any word containing it).",
      "Two sentences, no 'health'", "ifcheck", {"n_sentences": 2, "forbidden_substrings": ["health"]}, "Exactly 2 sentences; substring 'health' absent",
      expected="Exactly two sentences about sleep, avoiding 'health'/'healthy'"),
    # ---- Logical reasoning
    S("LR01", "dev", "logical_reasoning", "easy", "All bloops are razzies. All razzies are lazzies. Are all bloops definitely lazzies? Answer 'Yes' or 'No' first, then explain in one sentence.",
      "Yes", "yesno", {"answer": "yes"}, "Transitive syllogism -> Yes", reasoning=True, short_answer="yes"),
    S("LR02", "dev", "logical_reasoning", "medium", "If it rains, the match is cancelled. The match was not cancelled. Can we conclude that it did not rain? Answer 'Yes' or 'No' first, then explain in one sentence.",
      "Yes", "yesno", {"answer": "yes"}, "Modus tollens -> Yes", reasoning=True, short_answer="yes"),
    S("LR03", "test", "logical_reasoning", "medium", "All doctors in the clinic are tall. Priya is tall. Is Priya definitely a doctor in the clinic? Answer 'Yes' or 'No' first, then explain in one sentence.",
      "No", "yesno", {"answer": "no"}, "Affirming the consequent is invalid -> No", reasoning=True, short_answer="no"),
    # ---- Mathematical reasoning
    S("MR01", "dev", "mathematical_reasoning", "easy", "What is 15% of 240?" + COT, "36", "numeric", {"answer": 36}, "Final numeric answer 36", reasoning=True, short_answer="36"),
    S("MR02", "dev", "mathematical_reasoning", "easy", "Solve for x: 3x + 7 = 28." + COT, "7", "numeric", {"answer": 7}, "Final numeric answer 7", reasoning=True, short_answer="7"),
    S("MR03", "test", "mathematical_reasoning", "medium", "A train travels 180 km in 2.5 hours. What is its average speed in km/h?" + COT, "72", "numeric", {"answer": 72},
      "Final numeric answer 72", reasoning=True, short_answer="72"),
    # ---- Multi-step problem solving
    S("MS01", "dev", "multi_step", "easy", "A shop sells pens at 12 rupees each. Riya buys 7 pens and pays with a 100 rupee note. How many rupees of change does she get?" + COT,
      "16", "numeric", {"answer": 16}, "7*12=84; 100-84=16", reasoning=True, short_answer="16"),
    S("MS02", "dev", "multi_step", "medium", "Tom is twice as old as Sam. In 6 years the sum of their ages will be 48. How old is Tom now?" + COT, "24", "numeric", {"answer": 24},
      "s + 2s + 12 = 48 -> s=12, Tom=24", reasoning=True, short_answer="24"),
    S("MS03", "test", "multi_step", "hard", "A tank holds 600 litres and is 40% full. A pipe adds 25 litres per minute. How many minutes until the tank is completely full?" + COT,
      "14.4", "numeric", {"answer": 14.4, "tol": 0.05}, "40% of 600 = 240; remaining 360; 360/25 = 14.4", reasoning=True, short_answer="14.4"),
    # ---- Summarisation
    S("SU01", "dev", "summarization", "easy", SUMQ + P_SUM1, "Bees pollinate plants while feeding on nectar, so falling bee numbers threaten crop yields and food prices.", "summary",
      {"max_words": 25, "groups": [["bee", "bees"], ["pollinat", "pollen"], ["crop", "harvest", "food"]]},
      "<=25 words; mentions bees, pollination, crops/food", expected="One concise sentence (<=25 words) covering bees, pollination and crop/food impact"),
    S("SU02", "dev", "summarization", "medium", SUMQ + P_SUM2, "Working from home saves time and improves balance, but critics fear weaker teamwork and managers are divided on productivity.", "summary",
      {"max_words": 25, "groups": [["remote", "home", "hybrid"], ["saves", "flexib", "balance", "satisf", "benefit"], ["critic", "concern", "worr", "drawback", "divided", "weak"]]},
      "<=25 words; covers remote work, benefits and concerns", expected="One concise sentence (<=25 words) covering both benefits and concerns of remote work"),
    S("SU03", "test", "summarization", "medium", SUMQ + P_SUM3, "Cities run hotter than surroundings because dark surfaces and concrete trap heat, but trees and reflective roofs can cool them.", "summary",
      {"max_words": 25, "groups": [["urban", "city", "cities"], ["heat", "warm", "hot", "temperature"], ["tree", "green", "reflective", "plant", "shade", "cool"]]},
      "<=25 words; covers urban heat, cause, mitigation", expected="One concise sentence (<=25 words) covering urban heat, its cause and mitigation"),
    # ---- Coding
    S("CO01", "dev", "coding", "easy", "Write a Python function `is_palindrome(s)` that returns True if the string s reads the same forwards and backwards when case is ignored, otherwise False. Return only the code in one ```python block.",
      "def is_palindrome(s): t=s.lower(); return t==t[::-1]", "code",
      {"func": "is_palindrome", "tests": [[["Level"], True], [["abc"], False], [[""], True], [["Aa"], True], [["ab"], False]]}, "Passes 5 unit tests"),
    S("CO02", "dev", "coding", "easy", "Write a Python function `sum_even(nums)` that returns the sum of the even integers in the list nums (0 for an empty list). Return only the code in one ```python block.",
      "def sum_even(nums): return sum(n for n in nums if n%2==0)", "code",
      {"func": "sum_even", "tests": [[[[1, 2, 3, 4]], 6], [[[]], 0], [[[-2, 5]], -2], [[[7, 9]], 0]]}, "Passes 4 unit tests"),
    S("CO03", "test", "coding", "medium", "Write a Python function `second_largest(nums)` that returns the second largest DISTINCT value in a list of integers, or None if there are fewer than two distinct values. Return only the code in one ```python block.",
      "def second_largest(nums): d=sorted(set(nums)); return d[-2] if len(d)>1 else None", "code",
      {"func": "second_largest", "tests": [[[[3, 1, 2]], 2], [[[5, 5]], None], [[[1, 2, 2, 3]], 2], [[[-1, -5, -3]], -3], [[[]], None]]}, "Passes 5 unit tests"),
    # ---- Analytical
    S("AN01", "dev", "analytical", "medium", "A store's monthly sales were 120 in January, 150 in February and 135 in March. Which month had the largest month-over-month percentage increase, and what was that percentage?",
      "February, 25%", "keywords", {"groups": [["february", "feb"], ["25"]]}, "Names February and 25%", reasoning=True),
    S("AN02", "dev", "analytical", "medium", "Plan A costs 500 rupees per month flat. Plan B costs 200 rupees per month plus 10 rupees per GB used. At how many GB of monthly usage do the two plans cost the same?" + COT,
      "30", "numeric", {"answer": 30}, "200 + 10x = 500 -> x = 30", reasoning=True, short_answer="30"),
    S("AN03", "test", "analytical", "medium", "Dataset: 4, 8, 6, 5, 3, 9, 100. Is the mean or the median more representative of the typical value, and why? State the median value.",
      "Median (6) because 100 is an outlier that inflates the mean", "keywords", {"groups": [["median"], ["6"], ["outlier", "extreme", "skew", "100"]]},
      "Chooses median, states 6, justifies with outlier", reasoning=True),
    # ---- Structured output
    S("JS01", "dev", "structured_output", "easy", 'Return a JSON object with the keys "name" (string) and "age" (integer) for a person named Asha who is 30 years old. Output only the JSON.',
      '{"name": "Asha", "age": 30}', "structured", {"fmt": "json_object", "spec": {"keys": ["name", "age"], "values": {"name": "Asha", "age": 30}}},
      "Valid JSON object, correct keys and values, no extra prose", expected='Only a JSON object {"name": "Asha", "age": 30}'),
    S("JS02", "dev", "structured_output", "medium", 'Return a JSON array of exactly 3 objects, each with the keys "city" and "country", for Paris, Tokyo and Cairo in that order. Output only the JSON.',
      '[{"city":"Paris","country":"France"},{"city":"Tokyo","country":"Japan"},{"city":"Cairo","country":"Egypt"}]', "structured",
      {"fmt": "json_array", "spec": {"len": 3, "keys": ["city", "country"], "values_seq": [{"city": "Paris", "country": "France"}, {"city": "Tokyo", "country": "Japan"}, {"city": "Cairo", "country": "Egypt"}]}},
      "Valid JSON array of 3 correct objects in order", expected="Only a JSON array of three city/country objects in the requested order"),
    S("JS03", "test", "structured_output", "medium", "Produce a Markdown table with the columns Item, Qty, Price for: Pen 10 5; Book 2 120; Bag 1 450. Output only the table.",
      "| Item | Qty | Price |", "structured", {"fmt": "md_table", "spec": {"columns": ["Item", "Qty", "Price"], "rows": [["Pen", "10", "5"], ["Book", "2", "120"], ["Bag", "1", "450"]]}},
      "Valid Markdown table, correct header and 3 correct rows, nothing else", expected="Only a Markdown table with header Item|Qty|Price and the three rows"),
    # ---- Ambiguous / misleading
    S("AM01", "dev", "ambiguous_misleading", "medium", "What is the capital of Georgia?", "Ambiguous: country (Tbilisi) vs US state (Atlanta)", "ambiguity",
      {"all_terms": ["Tbilisi", "Atlanta"]}, "Asks which Georgia OR answers for both readings",
      expected="Either asks which Georgia is meant or gives both Tbilisi (country) and Atlanta (US state)", reasoning=True),
    S("AM02", "dev", "ambiguous_misleading", "medium", "How many months of the year have 28 days? Think carefully, then give a short answer.", "12 (all months)", "keywords",
      {"groups": [["12", "twelve", "all 12", "all of them", "all months", "every month", "all twelve"]]}, "Answers 12 / all months (trap: February only)", reasoning=True),
    S("AM03", "test", "ambiguous_misleading", "hard", "A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How much does the ball cost, in cents?" + COT.replace("<number>", "<number of cents>"),
      "5 cents", "numeric", {"answers": [5, 0.05], "tol": 0.001}, "Ball = 5 cents (intuitive wrong answer: 10)", reasoning=True, short_answer="5"),
]

# ------------------------------- robustness suite -------------------------------
def R(id, var, name, prompt, expected, eval_type, eval_spec, category="robustness", reasoning=False):
    return {"id": id, "split": "robustness", "category": category, "difficulty": "n/a", "variation": var, "variation_name": name, "prompt": prompt,
            "reference": expected, "criteria": expected, "expected_behaviour": expected, "eval_type": eval_type, "eval_spec": eval_spec,
            "reasoning": reasoning, "short_answer": None}

V1, V2, V3, V4, V5 = ("V1_prompt_wording", "V2_input_complexity", "V3_ambiguous_instructions", "V4_math_complexity", "V5_structured_formats")
CAN = {"answers": ["Canberra"]}
ROBUST = [
    R("R1a", V1, "terse + brevity request", "What is the capital of Australia? Answer with just the city name.", "Canberra", "contains_any", CAN),
    R("R1b", V1, "plain paraphrase", "Name Australia's capital city.", "Canberra", "contains_any", CAN),
    R("R1c", V1, "indirect phrasing", "I forgot: which city serves as the seat of government of Australia? One word please.", "Canberra", "contains_any", CAN),
    R("R1d", V1, "telegraphic / lower-case", "australia capital??", "Canberra", "contains_any", CAN),
    R("R1e", V1, "polite verbose", "Could you kindly tell me, in a full sentence, what the capital of Australia is?", "Canberra", "contains_any", CAN),
    R("R1f", V1, "math: symbolic", "What is 15% of 240?" + COT, "36", "numeric", {"answer": 36}, reasoning=True),
    R("R1g", V1, "math: words", "Calculate fifteen percent of two hundred forty." + COT, "36", "numeric", {"answer": 36}, reasoning=True),
    R("R1h", V1, "math: roundabout", "If you take 240 and find 15 percent of it, what do you get?" + COT, "36", "numeric", {"answer": 36}, reasoning=True),
    R("R2a", V2, "1 step", "Ravi has 12 apples. He buys 8 more. How many apples does he have?" + COT, "20", "numeric", {"answer": 20}, reasoning=True),
    R("R2b", V2, "2 steps", "Ravi has 12 apples. He buys 8 more and then gives 5 away. How many apples does he have?" + COT, "15", "numeric", {"answer": 15}, reasoning=True),
    R("R2c", V2, "3 steps", "Ravi has 12 apples. He buys 8 more, gives 5 away, then triples what he has left. How many apples does he have?" + COT, "45", "numeric", {"answer": 45}, reasoning=True),
    R("R2d", V2, "4 steps", "Ravi has 12 apples. He buys 8 more, gives 5 away, triples what he has left, eats 5, and finally shares the rest equally among himself and 3 friends (4 people in total). How many apples does each person get?" + COT,
      "10", "numeric", {"answer": 10}, reasoning=True),
    R("R3a", V3, "missing referent (edit)", "Make it shorter.", "Asks what text to shorten (or flags missing input)", "ambiguity", {}),
    R("R3b", V3, "pronoun without antecedent", "Is it safe to eat?", "Asks what 'it' refers to", "ambiguity", {}),
    R("R3c", V3, "unspecified units", "Convert 100 to the right units.", "Asks which quantity/units", "ambiguity", {}),
    R("R3d", V3, "unspecified language", "Translate this into the usual language.", "Asks for the text and the target language", "ambiguity", {}),
    R("R4a", V4, "level 1: addition", "What is 48 + 27?" + COT, "75", "numeric", {"answer": 75}, reasoning=True),
    R("R4b", V4, "level 2: 2-digit product", "What is 17 multiplied by 23?" + COT, "391", "numeric", {"answer": 391}, reasoning=True),
    R("R4c", V4, "level 3: linear equation", "Solve for x: 5x - 9 = 26." + COT, "7", "numeric", {"answer": 7}, reasoning=True),
    R("R4d", V4, "level 4: quadratic roots", "What is the sum of the roots of x^2 - 9x + 14 = 0?" + COT, "9", "numeric", {"answer": 9}, reasoning=True),
    R("R4e", V4, "level 5: compound interest", "12000 rupees are invested at 10% interest per year, compounded annually. What is the total amount after 2 years, in rupees?" + COT, "14520", "numeric", {"answer": 14520, "tol": 0.5}, reasoning=True),
    R("R5a", V5, "JSON object", "Return a JSON object with keys name, age, city for Meera, 28, Pune (age as an integer). Output only the JSON.", "JSON object", "structured",
      {"fmt": "json_object", "spec": {"keys": ["name", "age", "city"], "values": {"name": "Meera", "age": 28, "city": "Pune"}}}),
    R("R5b", V5, "JSON array of one object", "Return a JSON array containing exactly one object with keys name, age, city for Meera, 28, Pune (age as an integer). Output only the JSON.", "JSON array", "structured",
      {"fmt": "json_array", "spec": {"len": 1, "keys": ["name", "age", "city"], "values_seq": [{"name": "Meera", "age": 28, "city": "Pune"}]}}),
    R("R5c", V5, "CSV", "Return CSV with the header name,age,city and one data row for Meera, 28, Pune. Output only the CSV.", "CSV", "structured",
      {"fmt": "csv", "spec": {"columns": ["name", "age", "city"], "rows": [["Meera", "28", "Pune"]]}}),
    R("R5d", V5, "Markdown table", "Return a Markdown table with columns name, age, city and one row for Meera, 28, Pune. Output only the table.", "Markdown table", "structured",
      {"fmt": "md_table", "spec": {"columns": ["name", "age", "city"], "rows": [["Meera", "28", "Pune"]]}}),
    R("R5e", V5, "key: value lines", "Return exactly three lines of the form 'key: value' with keys name, age, city for Meera, 28, Pune. Output only those lines.", "key: value lines", "structured",
      {"fmt": "key_value", "spec": {"keys": ["name", "age", "city"], "values": {"name": "Meera", "age": 28, "city": "Pune"}}}),
]
for r in ROBUST:
    if r["eval_type"] == "ambiguity":
        r["eval_spec"] = {}

assert len(DATASET) == 30 and sum(d["split"] == "dev" for d in DATASET) == 20 and sum(d["split"] == "test" for d in DATASET) == 10
assert len({d["id"] for d in DATASET}) == 30


def dataset_json():
    return json.dumps({"name": "Pretrained_Model_CA common evaluation set", "version": "1.0",
                       "note": "Evaluation only - pretrained models are NOT trained on this data. dev (20) used to verify the pipeline; test (10) held out, "
                               "never used to adjust prompts/extractors.", "samples": DATASET, "robustness": ROBUST}, indent=1, ensure_ascii=False)
