import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
"""Generate gold_answer for ERAG queries via Bedrock gpt-5.4."""
import json, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from memondemand.core.api_adapter import APIError, call as api_call

GOLD_PATH = 'manifests/erag_query_gold_evaluator_only.jsonl'
GOLD_BAK = 'manifests/erag_query_gold_evaluator_only.jsonl.bak'
OUT_FULL = 'manifests/erag_query_gold_with_answers.jsonl'
QUERIES_PARQUET = 'manifests/erag_queries.parquet'
L0_PARQUET = 'manifests/erag_250M_l0_nodes.parquet'  # largest, covers all docs

SYSTEM = 'You are an expert who provides concise, factual answers based on provided documents. Answer the question using only the information in the provided document(s). Be specific and complete.'

MAX_CHARS_PER_DOC = 2000
MAX_TOTAL_DOC_CHARS = 8000  # safety cap
MAX_WORKERS = 8

print_lock = threading.Lock()
def log(s):
    with print_lock:
        print(s, flush=True)

def load_question_texts():
    df = pd.read_parquet(QUERIES_PARQUET)
    return dict(zip(df['query_id'], df['query_text']))

def load_doc_lookup():
    log('Loading L0 parquet (250M)...')
    df = pd.read_parquet(L0_PARQUET)
    log(f'L0 rows: {len(df)}')
    out = {}
    for _, r in df.iterrows():
        out[r['doc_id']] = (r['title'] or '', r['content'] or '')
    log(f'doc lookup ready: {len(out)} docs')
    return out

def build_user_msg(question, docs):
    # docs: list of (title, content) tuples
    parts = []
    total = 0
    for i, (t, c) in enumerate(docs):
        snippet = c[:MAX_CHARS_PER_DOC]
        block = f'--- Document {i+1}: {t} ---\n{snippet}'
        if total + len(block) > MAX_TOTAL_DOC_CHARS:
            break
        parts.append(block)
        total += len(block)
    doc_block = '\n\n'.join(parts) if parts else '(no documents)'
    return f'Document(s):\n{doc_block}\n\nQuestion: {question}'

def gen_one(rec, question_texts, doc_lookup, cost_tracker):
    qid = rec['question_id']
    eids = rec.get('expected_doc_ids', [])
    if not eids:
        return {'question_id': qid, 'expected_doc_ids': eids, 'question_text': question_texts.get(qid, ''), 'gold_answer': '', 'status': 'skipped_empty_eids'}
    q_text = question_texts.get(qid, '')
    docs = []
    for e in eids:
        if e in doc_lookup:
            docs.append(doc_lookup[e])
    if not docs:
        return {'question_id': qid, 'expected_doc_ids': eids, 'question_text': q_text, 'gold_answer': '', 'status': 'skipped_no_docs'}
    user = build_user_msg(q_text, docs)
    messages = [{'role':'system','content':SYSTEM},{'role':'user','content':user}]
    for attempt in range(3):
        try:
            resp = api_call('gpt_5_4', messages, max_tokens=400, temperature=0.0, timeout=90.0, max_retries=2)
            text = resp.get('text','').strip()
            usage = resp.get('usage', {})
            in_t = usage.get('input_tokens', 0)
            out_t = usage.get('output_tokens', 0)
            # gpt-5.4 pricing approx: .25/M in, 0/M out (based on Bedrock /pricing rough estimate)
            cost = in_t / 1_000_000 * 1.25 + out_t / 1_000_000 * 10.0
            with print_lock:
                cost_tracker['cost'] += cost
                cost_tracker['n_filled'] += 1
            return {'question_id': qid, 'expected_doc_ids': eids, 'question_text': q_text, 'gold_answer': text, 'status': 'ok', 'in_t': in_t, 'out_t': out_t}
        except Exception as e:
            log(f'  {qid} attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:120]}')
            time.sleep(2 ** attempt)
    return {'question_id': qid, 'expected_doc_ids': eids, 'question_text': q_text, 'gold_answer': '', 'status': 'failed_after_retries'}

def main():
    import shutil
    shutil.copy(GOLD_PATH, GOLD_BAK)
    log(f'Backed up to {GOLD_BAK}')

    question_texts = load_question_texts()
    doc_lookup = load_doc_lookup()

    with open(GOLD_PATH) as f:
        gold_rows = [json.loads(l) for l in f]
    log(f'Loaded {len(gold_rows)} gold rows')

    cost_tracker = {'cost': 0.0, 'n_filled': 0}
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(gen_one, r, question_texts, doc_lookup, cost_tracker): r['question_id'] for r in gold_rows}
        for i, fut in enumerate(as_completed(futures)):
            try:
                res = fut.result()
                results[res['question_id']] = res
            except Exception as e:
                qid = futures[fut]
                log(f'  {qid} exception: {e}')
                results[qid] = {'question_id': qid, 'expected_doc_ids': [], 'question_text': '', 'gold_answer': '', 'status': 'exception'}
            if (i+1) % 25 == 0:
                elapsed = time.time() - t0
                log(f'  progress: {i+1}/{len(gold_rows)}  elapsed={elapsed:.0f}s  cost=${cost_tracker["cost"]:.3f}')

    # Write full file (with question_text)
    n_filled = sum(1 for r in results.values() if r.get('gold_answer'))
    n_skipped = sum(1 for r in results.values() if r.get('status','').startswith('skipped'))
    n_failed = sum(1 for r in results.values() if r.get('status','').startswith('failed') or r.get('status','')=='exception')
    log(f'Final: n_filled={n_filled}, n_skipped={n_skipped}, n_failed={n_failed}, cost=${cost_tracker["cost"]:.3f}')

    with open(OUT_FULL, 'w') as f:
        for r in gold_rows:
            qid = r['question_id']
            res = results.get(qid, {})
            f.write(json.dumps({
                'question_id': qid,
                'expected_doc_ids': r.get('expected_doc_ids', []),
                'question_text': res.get('question_text', question_texts.get(qid, '')),
                'gold_answer': res.get('gold_answer', ''),
            }) + '\n')
    log(f'Wrote {OUT_FULL}')

    # Update original file (fill gold_answer)
    with open(GOLD_PATH, 'w') as f:
        for r in gold_rows:
            qid = r['question_id']
            res = results.get(qid, {})
            f.write(json.dumps({
                'question_id': qid,
                'expected_doc_ids': r.get('expected_doc_ids', []),
                'gold_answer': res.get('gold_answer', ''),
            }) + '\n')
    log(f'Updated {GOLD_PATH}')

if __name__ == '__main__':
    main()
