#!/usr/bin/env python3
"""Generate one frozen K=8 trained-owner panel per unique DPR1 train source."""

from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from typing import Any

from hf_mpr2_generate_drafts import canonical_sources, sha256_file
from hf_product_reasoning_eval import _generate_completions, _generation_stop_token_ids, _load_model, _render_prompt

SCHEMA="shohin-dpr1-train-panel-row-v1"; REPORT_SCHEMA="shohin-dpr1-train-panel-shard-v1"; SOURCE_SCHEMA="shohin-mpr1-revision-data-report-v1"; OWNER_ARCH="shohin-rme1-moe-revision-v1"
class DPR1TrainPanelError(RuntimeError): pass
def atomic_lines(path,rows):
    if path.exists(): raise DPR1TrainPanelError("DPR1 train panel output exists")
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp.{os.getpid()}"); d=hashlib.sha256()
    with tmp.open("xb") as f:
        for r in rows: b=(json.dumps(r,sort_keys=True)+"\n").encode(); f.write(b); d.update(b)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); return d.hexdigest()
def atomic_json(path,payload):
    if path.exists(): raise DPR1TrainPanelError("DPR1 train panel report exists")
    tmp=path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def run(args):
    import torch
    from transformers import AutoTokenizer
    source_report=json.loads(args.source_report.read_text()); expected=source_report.get("outputs",{}).get("aligned",{})
    if source_report.get("schema")!=SOURCE_SCHEMA or source_report.get("status")!="complete" or expected.get("sha256")!=sha256_file(args.source) or Path(str(expected.get("path", ""))).resolve()!=args.source.resolve(): raise DPR1TrainPanelError("DPR1 train source differs")
    source_rows=[json.loads(x) for x in args.source.read_text().splitlines() if x]; all_rows=canonical_sources(source_rows); start=len(all_rows)*args.shard_index//args.shard_count; end=len(all_rows)*(args.shard_index+1)//args.shard_count; rows=all_rows[start:end]
    tok=AutoTokenizer.from_pretrained(args.model_root,trust_remote_code=True); tok.padding_side="left"; tok.pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model,meta,loader=_load_model(args.model_root,args.owner_checkpoint,args.model_loader)
    if meta.get("architecture")!=OWNER_ARCH or meta.get("rme1_draft_control")!="draft_unavailable" or int(meta.get("update",-1))!=256: raise DPR1TrainPanelError("DPR1 owner differs")
    stop=_generation_stop_token_ids(tok); torch.manual_seed(args.seed+args.shard_index); torch.cuda.manual_seed_all(args.seed+args.shard_index); torch.cuda.reset_peak_memory_stats(); out=[]; tokens=exhausted=0; started=time.monotonic()
    for index,row in enumerate(rows):
        prompt=_render_prompt(tok,row["question"],True,False); completions,usage=_generate_completions(model,tok,[prompt]*8,True,"qwen-thinking",args.max_new_tokens,stop); panel=[]
        for i,(completion,(count,hit)) in enumerate(zip(completions,usage,strict=True)):
            if not completion.strip(): raise DPR1TrainPanelError("empty DPR1 panel member")
            panel.append({"candidate_index":i,"completion":completion,"generated_tokens":count,"max_token_exhausted":hit}); tokens+=count; exhausted+=int(hit)
        out.append({"schema":SCHEMA,"source_identity_sha256":row["source_identity_sha256"],"source_prompt_sha256":hashlib.sha256(row["question"].encode()).hexdigest(),"candidates":panel})
        if (index+1)%16==0 or index+1==len(rows): print(f"[dpr1-train-panel] {index+1}/{len(rows)}",flush=True)
    torch.cuda.synchronize(); elapsed=time.monotonic()-started; output_sha=atomic_lines(args.output,out); report={"schema":REPORT_SCHEMA,"status":"complete","source":str(args.source.resolve()),"source_sha256":sha256_file(args.source),"source_report_sha256":sha256_file(args.source_report),"unique_sources":len(all_rows),"owner_checkpoint_sha256":sha256_file(args.owner_checkpoint),"owner_architecture":meta["architecture"],"owner_update":meta["update"],"owner_draft_control":meta["rme1_draft_control"],"model_revision":args.model_revision,"model_loader":loader,"panel_size":8,"sampling":{"temperature":1.0,"top_p":.95,"top_k":20},"max_new_tokens":args.max_new_tokens,"seed":args.seed,"effective_seed":args.seed+args.shard_index,"shard_index":args.shard_index,"shard_count":args.shard_count,"row_start":start,"row_end":end,"rows":len(out),"generated_tokens":tokens,"max_token_exhausted":exhausted,"elapsed_seconds":elapsed,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),"output":str(args.output.resolve()),"output_sha256":output_sha}; atomic_json(args.report,report); return report
def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--source-report",type=Path,required=True); p.add_argument("--model-root",type=Path,required=True); p.add_argument("--model-revision",required=True); p.add_argument("--model-loader",default="causal"); p.add_argument("--owner-checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--report",type=Path,required=True); p.add_argument("--shard-index",type=int,required=True); p.add_argument("--shard-count",type=int,required=True); p.add_argument("--max-new-tokens",type=int,default=768); p.add_argument("--seed",type=int,default=2026080925); a=p.parse_args(); print(json.dumps(run(a),sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())

