#!/usr/bin/env python3
"""Merge complete, disjoint DPR1 training-panel shards."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
SCHEMA="shohin-dpr1-train-panel-shard-v1"; ROW_SCHEMA="shohin-dpr1-train-panel-row-v1"; OUT_SCHEMA="shohin-dpr1-train-panels-v1"
class DPR1PanelMergeError(RuntimeError): pass
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main_merge(args):
    reports=[json.loads(p.read_text()) for p in args.shard_report]; n=int(reports[0]["shard_count"]); stable=("source_sha256","source_report_sha256","unique_sources","owner_checkpoint_sha256","owner_architecture","owner_update","owner_draft_control","model_revision","panel_size","max_new_tokens","seed","shard_count")
    if len(reports)!=n or sorted(r["shard_index"] for r in reports)!=list(range(n)) or any(r.get("schema")!=SCHEMA or r.get("status")!="complete" for r in reports) or any(any(r.get(k)!=reports[0].get(k) for k in stable) for r in reports[1:]): raise DPR1PanelMergeError("DPR1 train-panel shards differ")
    rows=[]
    for r in sorted(reports,key=lambda x:x["shard_index"]):
        p=Path(r["output"])
        if sha(p)!=r["output_sha256"]: raise DPR1PanelMergeError("DPR1 panel hash differs")
        part=[json.loads(x) for x in p.read_text().splitlines() if x]
        if len(part)!=r["rows"] or any(x.get("schema")!=ROW_SCHEMA or len(x.get("candidates",[]))!=8 for x in part): raise DPR1PanelMergeError("DPR1 panel rows differ")
        rows+=part
    rows.sort(key=lambda x:x["source_identity_sha256"]); ids=[x["source_identity_sha256"] for x in rows]
    if len(rows)!=reports[0]["unique_sources"] or len(ids)!=len(set(ids)): raise DPR1PanelMergeError("DPR1 panel coverage differs")
    if args.output.exists() or args.report.exists(): raise DPR1PanelMergeError("DPR1 panel merge exists")
    args.output.parent.mkdir(parents=True,exist_ok=True); t=args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}"); d=hashlib.sha256()
    with t.open("xb") as f:
        for r in rows: b=(json.dumps(r,sort_keys=True)+"\n").encode(); f.write(b); d.update(b)
        f.flush(); os.fsync(f.fileno())
    os.replace(t,args.output); payload={"schema":OUT_SCHEMA,"status":"complete","source_sha256":reports[0]["source_sha256"],"source_report_sha256":reports[0]["source_report_sha256"],"owner_checkpoint_sha256":reports[0]["owner_checkpoint_sha256"],"model_revision":reports[0]["model_revision"],"panel_size":8,"seed":reports[0]["seed"],"shard_count":n,"rows":len(rows),"generated_tokens":sum(r["generated_tokens"] for r in reports),"max_token_exhausted":sum(r["max_token_exhausted"] for r in reports),"output":str(args.output.resolve()),"output_sha256":d.hexdigest()}; t=args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}"); t.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(t,args.report); return payload
def main():
    p=argparse.ArgumentParser(); p.add_argument("--shard-report",type=Path,action="append",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--report",type=Path,required=True); print(json.dumps(main_merge(p.parse_args()),sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
