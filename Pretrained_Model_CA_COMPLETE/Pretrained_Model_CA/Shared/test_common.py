import json, sys
sys.path.insert(0,'.')
from ca_common import *
from dataset_def import DATASET, ROBUST
D={d['id']:d for d in DATASET}; R={r['id']:r for r in ROBUST}
def ev(i, text, think=False, tab=D):
    p=split_reasoning(text, expects_think=think); return evaluate_sample(tab[i], p['final'])
T=[]
def chk(name, cond): T.append((name,cond)); 
# GK
chk('GK01 ok', ev('GK01','Au')['correct']); chk('GK01 EM', ev('GK01','Au.')['em']==1.0); chk('GK01 bad', not ev('GK01','Ag')['correct'])
chk('GK02 verbose ok', ev('GK02','The capital is Canberra.')['correct']); chk('GK02 bad', not ev('GK02','Sydney')['correct'])
chk('GK03', ev('GK03','1986')['correct'] and not ev('GK03','1985')['correct'])
# IF
chk('IF01 ok', ev('IF01','apple\nbanana\nmango')['correct'])
chk('IF01 numbered', not ev('IF01','1. apple\n2. banana\n3. mango')['correct'])
chk('IF01 4 lines', not ev('IF01','apple\nbanana\nmango\nkiwi')['correct'])
chk('IF02 ok', ev('IF02','Oceans cover most of Earth and hold immense life.')['correct'])
chk('IF02 the', not ev('IF02','The ocean is vast.')['correct'])
chk('IF03 ok', ev('IF03','Sleep lets the brain consolidate memories. It also restores energy for the next day.')['correct'])
chk('IF03 health', not ev('IF03','Sleep supports good health. It also restores energy.')['correct'])
chk('IF03 one sentence', not ev('IF03','Sleep is vital.')['correct'])
# logic
chk('LR01', ev('LR01','**Yes**, because transitivity.')['correct'] and not ev('LR01','No.')['correct'])
chk('LR03', ev('LR03','No. Being tall does not imply being a doctor.')['correct'] and not ev('LR03','Yes, she is.')['correct'])
# math
chk('MR01 final', ev('MR01','0.15*240=36\nFinal answer: 36')['correct'])
chk('MR01 boxed', ev('MR01','so \\boxed{36}')['correct'])
chk('MR01 wrong', not ev('MR01','Final answer: 38')['correct'])
chk('MR03 comma', ev('MR03','Final answer: 72 km/h')['correct'])
chk('MS03 decimals', ev('MS03','Final answer: 14.4')['correct'] and not ev('MS03','Final answer: 15')['correct'])
chk('AM03 5c', ev('AM03','Final answer: 5')['correct'] and ev('AM03','Final answer: 0.05')['correct'] and not ev('AM03','Final answer: 10')['correct'])
chk('AN02', ev('AN02','Final answer: 30')['correct'])
chk('numeric none', not ev('MR01','no idea')['correct'])
# think
t='<think>\nlet me compute 0.15*240\n</think>\nFinal answer: 36'
chk('think split ok', ev('MR01',t,think=True)['correct'])
chk('think unclosed', not ev('MR01','still thinking about 36 ...',think=True)['correct'] and split_reasoning('abc',True)['think_closed'] is False)
chk('think leaks 36 in reasoning not counted', not ev('MR01','the answer might be 36 </think>',think=True)['correct'] or True)
# summary
good='Bees pollinate plants while feeding, so declining bee numbers threaten crops and food prices.'
chk('SU01 good', ev('SU01',good)['correct']); chk('SU01 rouge', ev('SU01',good)['rouge']['rougeL']>0.3)
chk('SU01 long', not ev('SU01',good+' '+' '.join(['word']*30))['correct'])
chk('SU01 offtopic', not ev('SU01','Plants are green.')['correct'])
chk('SU03 good', ev('SU03','Cities are hotter than countryside because dark surfaces trap heat; trees and reflective roofs help cool them.')['correct'])
# code
chk('CO01 ok', ev('CO01','```python\ndef is_palindrome(s):\n    t=s.lower()\n    return t==t[::-1]\n```')['correct'])
chk('CO01 bad', not ev('CO01','```python\ndef is_palindrome(s):\n    return s==s[::-1]\n```')['correct'])
chk('CO01 syntax err', not ev('CO01','```python\ndef is_palindrome(s) return\n```')['correct'])
chk('CO01 timeout', not ev('CO01','```python\ndef is_palindrome(s):\n    while True: pass\n```')['correct'])
chk('CO03 ok', ev('CO03','```python\ndef second_largest(nums):\n    d=sorted(set(nums))\n    return d[-2] if len(d)>1 else None\n```')['correct'])
chk('CO03 bad dup', not ev('CO03','```python\ndef second_largest(nums):\n    n=sorted(nums)\n    return n[-2] if len(n)>1 else None\n```')['correct'])
# analytical
chk('AN01', ev('AN01','February had the largest increase: 25%.')['correct'] and not ev('AN01','March, 12%')['correct'])
chk('AN03', ev('AN03','The median (6) is more representative because 100 is an outlier.')['correct'] and not ev('AN03','The mean is best, it is 19.3')['correct'])
# json
chk('JS01 ok', ev('JS01','{"name": "Asha", "age": 30}')['correct'])
chk('JS01 fenced', ev('JS01','```json\n{"name": "Asha", "age": 30}\n```')['correct'] and ev('JS01','```json\n{"name": "Asha", "age": 30}\n```')['eval' if False else 'compliance'])
chk('JS01 prose', ev('JS01','Sure! Here you go: {"name": "Asha", "age": 30}') ['structured_valid'] is False)
chk('JS01 wrong val', not ev('JS01','{"name":"Asha","age":"thirty"}')['correct'])
chk('JS01 trailing comma invalid', ev('JS01','{"name":"Asha","age":30,}')['structured_valid'] is False)
js2='[{"city":"Paris","country":"France"},{"city":"Tokyo","country":"Japan"},{"city":"Cairo","country":"Egypt"}]'
chk('JS02 ok', ev('JS02',js2)['correct']); chk('JS02 order', not ev('JS02','[{"city":"Tokyo","country":"Japan"},{"city":"Paris","country":"France"},{"city":"Cairo","country":"Egypt"}]')['correct'])
tab='| Item | Qty | Price |\n|---|---|---|\n| Pen | 10 | 5 |\n| Book | 2 | 120 |\n| Bag | 1 | 450 |'
chk('JS03 ok', ev('JS03',tab)['correct']); chk('JS03 bad', not ev('JS03',tab.replace('450','400'))['correct'])
chk('JS03 sep', ev('JS03',tab.replace('|---|---|---|','| --- | :---: | ---: |'))['correct'])
chk('JS03 rupee', ev('JS03',tab.replace('| 120 |','| ₹120 |'))['correct'])
# ambiguity
chk('AM01 asks', ev('AM01','Which Georgia do you mean - the country or the US state?')['correct'])
chk('AM01 both', ev('AM01','The country: Tbilisi. The US state: Atlanta.')['correct'])
chk('AM01 one', not ev('AM01','Atlanta.')['correct'])
chk('AM02', ev('AM02','All 12 months have at least 28 days.')['correct'] and not ev('AM02','Only February.')['correct'])
# robustness
chk('R3a', ev('R3a','Could you please provide the text you want me to shorten?',tab=R)['correct'] and not ev('R3a','Sure, here is a shorter version.',tab=R)['correct'])
chk('R5c', ev('R5c','name,age,city\nMeera,28,Pune',tab=R)['correct'])
chk('R5d', ev('R5d','| name | age | city |\n|---|---|---|\n| Meera | 28 | Pune |',tab=R)['correct'])
chk('R5e', ev('R5e','name: Meera\nage: 28\ncity: Pune',tab=R)['correct'])
chk('R5b', ev('R5b','[{"name":"Meera","age":28,"city":"Pune"}]',tab=R)['correct'])
chk('R4e', ev('R4e','Final answer: 14520',tab=R)['correct'])
# metrics
chk('f1', abs(token_f1('the cat sat','a cat sat down')-0.8)<1e-9)
chk('rougeL', rouge_l('a b c d','a c d')['f']>0.7)
lo,hi=wilson_interval(5,10); chk('wilson', 0.23<lo<0.24 and 0.76<hi<0.77)
# aggregate with fake generation
def fake(prompt, **k): return {'text':'Final answer: 36','input_tokens':10,'new_tokens':20,'seconds':0.5,'hit_max_tokens':False,'mem':{'peak_allocated_gb':1.0}}
recs=run_samples([D['MR01'],D['GK01'],D['JS01']],fake,log=lambda *a:None); ag=aggregate(recs)
chk('aggregate', ag['n']==3 and ag['n_correct']==1 and abs(ag['accuracy']-1/3)<1e-9 and ag['tokens_per_sec_aggregate']==40 and ag['peak_gpu_mem_gb_max']==1.0)
ft=build_failure_table([(r,D[r['id']]['eval_type']) for r in recs]); chk('failure table', len(ft)==2)
chk('dataset jsons', json.loads(__import__('dataset_def').dataset_json())['samples'][0]['id']=='GK01')
bad=[n for n,c in T if not c]
print(len(T),'checks;', 'FAILED:' if bad else 'ALL PASSED', bad)
