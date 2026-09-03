import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
"""V5 fix5: post-hoc citation extraction.

Takes fix3 answers.jsonl as input. For each query, asks GPT-5.4 which of the
detailed_context_node_ids were actually used to write the answer. Writes new
answers.jsonl with updated cited_evidence_ids.
"""
import json, time, re, threading, argparse, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from memondemand.core.api_adapter import APIError, call as api_call

SYSTEM = '''You are a citation auditor. Your task is to identify which documents were actually used to write an answer. Return ONLY a JSON array of document IDs.'''

USER_TMPL = '''Question: {q}

Answer: {a}

Candidate documents:
{docs}

Return ONLY a JSON array of document IDs (e.g. ["dsid_abc", "dsid_xyz"]) whose specific content is DIRECTLY present in the answer above. Return [] if none were used. Do not include topically related but unused documents.

JSON only, no preamble:'''

print_lock = threading.Lock()
def log(s):
    with print_lock:
        print(s, flush=True)

JSON_ARR_RE = re.compile(r'\[\s*(?:"[^"]*"\s*,?\s*)*\]')

def parse_citation_array(text, candidate_ids):
    cset = set(candidate_ids)
    m = JSON_ARR_RE.search(text or '')
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list):
            return [str(x) for x in arr if str(x) in cset]
    except Exception:
        pass
    return []

def build_doc_lookup(l0_parquet):
    log(f'Loading L0 parquet {l0_parquet} ...')
    df = pd.read_parquet(l0_parquet)
    log(f'L0 rows: {len(df)}')
    out = {}
    for _, r in df.iterrows():
        title = r.get('title','') or ''
        content = r.get('content','') or ''
        snippet = (title + ': ' + content)[:300]
        out[r['doc_id']] = snippet
    log(f'doc lookup ready: {len(out)} docs')
    return out

def extract_one(rec, doc_lookup, cost_tracker):
    qid = rec['query_id']
    q = rec.get('query_text','')
    a = rec.get('answer_text','')
    cand_ids = rec.get('detailed_context_node_ids', []) or []
    orig_cited = rec.get('cited_evidence_ids', []) or []
    if not a or not cand_ids:
        # No answer or no candidates: keep original
        return {'query_id': qid, 'new_cited': orig_cited, 'fallback': True, 'cost': 0.0}
    # Build candidate doc snippets (cap to 12)
    cand_ids = cand_ids[:12]
    doc_lines = []
    for did in cand_ids:
        snippet = doc_lookup.get(did, '')[:300]
        doc_lines.append(f'[{did}]: {snippet}')
    docs = '\n'.join(doc_lines)
    user = USER_TMPL.format(q=q[:500], a=a[:1500], docs=docs)
    messages = [{'role':'system','content':SYSTEM},{'role':'user','content':user}]
    for attempt in range(3):
        try:
            resp = api_call("general", messages, max_tokens=300, temperature=0.0, timeout=90.0, max_retries=2)
            text = resp.get('text','').strip()
            usage = resp.get('usage', {})
            in_t = usage.get('input_tokens', 0)
            out_t = usage.get('output_tokens', 0)
            cost = in_t / 1_000_000 * 1.25 + out_t / 1_000_000 * 10.0
            new_cited = parse_citation_array(text, cand_ids)
            with print_lock:
                cost_tracker['cost'] += cost
            if not new_cited:
                # Empty LLM response -> fallback to original (avoid empty citations)
                return {'query_id': qid, 'new_cited': orig_cited, 'fallback': True, 'cost': cost, 'raw': text[:200]}
            return {'query_id': qid, 'new_cited': new_cited, 'fallback': False, 'cost': cost, 'raw': text[:200]}
        except Exception as e:
            log(f'  {qid} attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:100]}')
            time.sleep(2 ** attempt)
    return {'query_id': qid, 'new_cited': orig_cited, 'fallback': True, 'cost': 0.0, 'error': True}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_answers', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--l0_parquet', default='manifests/erag_10M_l0_nodes.parquet')
    ap.add_argument('--max_workers', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_answers = os.path.join(args.out_dir, 'answers.jsonl')

    doc_lookup = build_doc_lookup(args.l0_parquet)

    with open(args.in_answers) as f:
        recs = [json.loads(l) for l in f]
    log(f'Loaded {len(recs)} answer records')

    cost_tracker = {'cost': 0.0}
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(extract_one, r, doc_lookup, cost_tracker): r['query_id'] for r in recs}
        for i, fut in enumerate(as_completed(futures)):
            try:
                res = fut.result()
                results[res['query_id']] = res
            except Exception as e:
                qid = futures[fut]
                log(f'  {qid} exception: {e}')
                results[qid] = {'query_id': qid, 'new_cited': [], 'fallback': True, 'cost': 0.0, 'error': True}
            if (i+1) % 50 == 0:
                log(f'  progress: {i+1}/{len(recs)}  elapsed={time.time()-t0:.0f}s  cost=${cost_tracker["cost"]:.3f}')

    n_fallback = sum(1 for r in results.values() if r.get('fallback'))
    n_changed = 0
    n_empty_new = 0
    # Write new answers.jsonl with updated cited
    with open(out_answers, 'w') as f:
        for r in recs:
            qid = r['query_id']
            res = results.get(qid, {})
            new_cited = res.get('new_cited', r.get('cited_evidence_ids', []))
            orig = set(r.get('cited_evidence_ids', []) or [])
            new_set = set(new_cited)
            if new_set != orig:
                n_changed += 1
            if not new_cited:
                n_empty_new += 1
            r_out = dict(r)
            r_out['cited_evidence_ids'] = new_cited
            r_out['_fix5_fallback'] = res.get('fallback', False)
            f.write(json.dumps(r_out) + '\n')
    log(f'Wrote {out_answers}')
    log(f'Final: n_fallback={n_fallback}, n_changed={n_changed}, n_empty_new={n_empty_new}, cost=${cost_tracker["cost"]:.3f}')

if __name__ == '__main__':
    main()
