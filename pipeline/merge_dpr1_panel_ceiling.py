#!/usr/bin/env python3
"""Merge DPR1 panel shards and apply the frozen information-ceiling gate."""

from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

SCHEMA="shohin-dpr1-panel-ceiling-shard-v1"; OUTPUT_SCHEMA="shohin-dpr1-panel-ceiling-v1"
class DPR1MergeError(RuntimeError): pass
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def merge(args):
    reports=[json.loads(p.read_text()) for p in args.shard_report]; count=int(reports[0]["shard_count"])
    if len(reports)!=count or sorted(r["shard_index"] for r in reports)!=list(range(count)) or any(r.get("schema")!=SCHEMA or r.get("status")!="complete" for r in reports): raise DPR1MergeError("DPR1 shards differ")
    stable=("data_sha256","data_report_sha256","model_revision","owner_checkpoint_sha256","owner_architecture","owner_update","owner_draft_control","panel_size","max_new_tokens","seed","shard_count","full_rows")
    if any(any(r.get(k)!=reports[0].get(k) for k in stable) for r in reports[1:]): raise DPR1MergeError("DPR1 shard custody differs")
    rows=[]
    for p,r in sorted(zip(args.shard_report,reports),key=lambda x:x[1]["shard_index"]):
        op=Path(r["output"])
        if sha(op)!=r["output_sha256"]: raise DPR1MergeError("DPR1 output hash differs")
        rows += [json.loads(x) for x in op.read_text().splitlines() if x]
    ids=[r["identity_sha256"] for r in rows]
    if len(rows)!=reports[0]["full_rows"] or len(ids)!=len(set(ids)): raise DPR1MergeError("DPR1 coverage differs")
    fixed=[sum(r["fixed_index_correct"][i] for r in reports) for i in range(8)]; domains={t:sum(r.get("domain_oracle",{}).get(t,0) for r in reports) for t in ("math500","bbh_logic","mbpp")}; oracle=sum(r["oracle_correct"] for r in reports); diverse=sum(r["diverse_rows"] for r in reports); exhausted=sum(r["max_token_exhausted"] for r in reports)
    gates={"oracle_at_least_350":oracle>=350,"math_oracle_at_least_90":domains["math500"]>=90,"logic_oracle_at_least_245":domains["bbh_logic"]>=245,"code_oracle_at_least_15":domains["mbpp"]>=15,"diverse_rows_at_least_25_percent":diverse>=323,"exhaustion_at_most_80":exhausted<=80}
    payload={"schema":OUTPUT_SCHEMA,"status":"complete","split":"development","rows":len(rows),"panel_size":8,"fixed_index_correct":fixed,"oracle_correct":oracle,"domain_oracle":domains,"diverse_rows":diverse,"diverse_fraction":diverse/len(rows),"generated_tokens":sum(r["generated_tokens"] for r in reports),"max_token_exhausted":exhausted,"gates":gates,"gate_pass":all(gates.values()),"training_authorized":all(gates.values()),"decision":"freeze_panel_revision_fit" if all(gates.values()) else "close_exact_dpr1","inputs":[{"report":str(p.resolve()),"report_sha256":sha(p),"output_sha256":r["output_sha256"]} for p,r in zip(args.shard_report,reports)]}
    if args.output.exists(): raise DPR1MergeError("DPR1 result exists")
    tmp=args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,args.output); return payload
def main():
    p=argparse.ArgumentParser(); p.add_argument("--shard-report",type=Path,action="append",required=True); p.add_argument("--output",type=Path,required=True); r=merge(p.parse_args()); print(json.dumps(r,sort_keys=True)); return 0 if r["gate_pass"] else 3
if __name__=="__main__": raise SystemExit(main())

