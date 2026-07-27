# R12 Dual-Provenance Carry-Motor Recovery Preregistration

**Status: COMMIT / INSTALL / SIGN / H100 NO-GO pending fresh exact hostile
rereview.** Local CPU tests and static review are the only actions authorized by
this candidate.
No reservation, recovery plan, fit, development evaluation, confirmation
generation, or confirmation evaluation is authorized now. A fresh reviewer
must first approve the exact four-file source sufficiently to permit the
custody-only reservation step. That step creates no fit; it pins the new root
and three output-directory inodes and emits an unsigned review statement. A
holder of the frozen external Ed25519 review key must then sign that exact
statement. Only a valid immutable signed receipt can authorize plan
publication and H100 fit. This document is not a capability claim.

## 1. Purpose and frozen upstream lineage

The sole purpose of this protocol is to recover the already preregistered carry
motor fit from a mechanical Python/JSON representation defect and to add the
restricted calibration nulls that the original protocol omitted. The recovery
does not claim that new executor code produced the upstream plan or feature
tensors. Treatment versus shuffled is not sufficient evidence of a
feature-dependent reader: favorable global and operation/width/cursor/terminal
carry-logit threshold shifts must also be measured and beaten. Exact
within-nuisance balancing and shuffled-label preservation are necessary but do
not replace a direct nuisance-only model-class comparison.

The immutable upstream identities are:

- source commit:
  `a0c258e6709766c643cf127a429a7d6ef4a4211b`
- source-manifest SHA-256:
  `9ae61e1a3e8f672a71a01edc16e6a5f1f8f3c69f49afd5e97f41c6cde15350a9`
- canonical plan SHA-256:
  `1b845d47f6875df571169efb5adb0716dfbc5d266a2499e4a92451351a262b6d`
- confirmation-commitment SHA-256:
  `1ee32e4e2e8f9eb56026b7b8de1fdff207e9fd3694e0ae354f103d58ebb820da`
- fit-row SHA-256:
  `6517b1ff3aa557e449a2eef9c5540c3d5f8699482d933d5c320b606adb4a0f1b`
- canonical board SHA-256:
  `d6282610ba845b23ebe849efe574233bf657a50aea0a7edb901e9e1d95b24391`

The repaired source freezes plan audit
`causal_carry_motor_recovery_plan_v8`, fit audit
`causal_carry_motor_fit_v11_recovery_v7`, and signed-review audit
`causal_carry_motor_recovery_signed_review_v8`. These names are schema
identities, not authorization or result claims.

The eight immutable feature-shard receipts, in shard-index order, are:

1. `4affa12434513ebe9587464ff38656abaaf7e47904d9db6ced252c3adea52a96`
2. `4731c1644703e26c1978ca1ec1ba80af7c173c5d9676ae68fbd04368f3b54c2c`
3. `e81639e68a838bfa6695be92f7c1333d100b2317c48fb2cf0d995f22a6e50a43`
4. `ae86ec1b70dca21d67849fc4be17ffec682472851735c3b9523292836a74e70f`
5. `ce5a151f89e20e774c7d37afc446ea026ec14a587c70fa614414f060f10a2144`
6. `f02d8221bf3a393566c279e27bf888fcbd1ef9ea17bdd33262472c898950ea83`
7. `009b83f0c2a70362654e3e3e4cad27d30f79f93f3bdd32d6ce3064695dd2b9db`
8. `8214d356288c56a116a3de753a8948a35f731d52c520fa906f4e31c1b0f14fb4`

The upstream root
`artifacts/carry_motor/canonical_a0c258e6709766c643cf127a429a7d6ef4a4211b`
is read-only evidence. Recovery must require its root and shard directories to
remain mode `0555`, its plan and shard files mode `0444` and one-link, and its
fit, development, and confirmation directories empty mode `0700`. Recovery
never writes, renames, links, copies, chmods, or publishes inside that root.

## 2. Observed failure and exact normalization proof

Job `692563` successfully replayed all eight shards and completed the frozen
2,000 treatment plus 2,000 shuffled updates. It then failed before publication
because the generated in-memory board contained integer histogram keys while
the JSON-loaded plan contained string keys. The prepublication fit directory
remained empty.

An independent reconstruction from the exact tokenizer and episode bytes
generated 65,536 rows with the frozen row digest. A recursive type-sensitive
comparison found exactly these two differences:

```json
[
  {
    "generated_key_type": "int",
    "generated_keys": [97, 99, 103, 105],
    "path": "board.prompt_length_histogram",
    "sealed_key_type": "str",
    "sealed_keys": ["97", "99", "103", "105"]
  },
  {
    "generated_key_type": "int",
    "generated_keys": [114, 116, 120, 122],
    "path": "board.token_length_histogram",
    "sealed_key_type": "str",
    "sealed_keys": ["114", "116", "120", "122"]
  }
]
```

The ledger SHA-256 is
`b43cb4a6fbfab97c659e8658f63185ae8b3dc1d8cce34089958d3b09df0593b6`.
All non-histogram fields, histogram counts, row order, labels, and values are
recursively key-type-exact and value-type-exact. Strict finite JSON serialization followed by
duplicate-key-rejecting parsing produces the exact sealed plan board and the
canonical board digest above.

The sole allowed transformation is
`strict_json_round_trip_of_complete_generated_fit_board`. It has zero permitted
semantic changes and zero additional transformations. A count change, extra
key, non-histogram difference, bool/int or int/float alias, duplicate JSON key,
JSON `NaN`/`Infinity`, finite-parser overflow such as `1e999` or `-1e999`, or a
third type difference fails closed. Every parsed or canonicalized JSON tree must
be an acyclic tree of exact `dict`, `list`, string-key, `str`, `bool`, `int`,
finite `float`, or null values. It receives a recursive finite-float and key-type
check in addition to a finite `parse_float` hook. Integer mapping keys are
accepted only by the one frozen generated-board normalization entrypoint.

## 3. Dual provenance

The recovery lineage has two noninterchangeable source identities:

1. **Upstream protocol source.** The exact `a0c258e` source contract recorded by
   the sealed plan and every shard. This identity owns the board, features,
   labels, controls, fit mathematics, confirmation commitment, and frozen
   scientific semantics.
2. **Recovery executor source.** A later reviewed Git commit containing exactly
   the following four existing, non-v9 paths and no aliases:
   `R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md`,
   `train/causal_carry_motor_recovery.py`,
   `train/test_causal_carry_motor_recovery.py`, and
   `train/jobs/causal_carry_motor_recovery.sbatch`. The earlier review request's
   `_v9` filenames were an instruction mismatch and have no authority. This
   identity owns only binding, strict board normalization, recovery validation,
   and recovery publication.

Runtime requires `HEAD` to equal the recovery commit, a descriptor-bound index
equal to its tree, a closed-world checkout whose descriptor-read bytes equal the
tree blob IDs, and an exact manifest over the four recovery files. The recovery
commit must have the full `a0c258e` commit above as its sole direct parent. Tree
maps, not diff porcelain, must establish exactly those four path additions.
The source manifest is regenerated from the descriptor-read bytes as a sorted,
compact ASCII JSON mapping from each exact path to its SHA-256; its manifest
SHA-256 is computed from those bytes rather than copied from a prior request. A
modified baseline file, fifth file, rename, merge, grandchild, extra commit,
untracked or ignored leaf, or module shadow fails closed.

No command may interpret executable repository Git configuration, attributes,
hooks, clean/smudge/process filters, fsmonitor, or external diff settings. The
executor descriptor-reads the `.git` pointer, commondir pointer, `HEAD`, direct
ref or packed refs, index, and object directory, then creates a private
mode-`0700` synthetic bare Git directory with no local config and no attached
worktree. Git system and global config are excluded, HOME and XDG config roots
are isolated, replace objects and system attributes are disabled, and the only
permitted plumbing is `cat-file`, `ls-tree`, and `ls-files --stage` against the
synthetic directory and copied index. `status`, `diff`, `show`, `add`, checkout,
and all worktree-filter paths are forbidden. Local/common/worktree config and
`info/attributes`, if present, are descriptor-bound as uninterpreted evidence
only. Checkout closure is established by a dirfd-relative no-symlink walk and
direct Git-blob hashing of each one-link regular leaf. Include-path,
filter-clean, filter-process, info-attributes, and fsmonitor attacks must prove
that no attacker command executes.

Every loaded recovery, upstream, model, controller, protocol, evaluation, and
probe module must resolve to its exact reviewed path. Each of the four recovery
sources must be Git mode `100644` and a one-link, non-symlink mode-`0644` regular
checkout file; hard-link aliases fail. Every imported upstream scientific
dependency must still equal its blob bytes in `a0c258e`. Passing the old commit
for modified code, `PYTHONPATH` substitution, monkeypatching, dirty checkout
execution, or relabelling an old shard as recovery-produced fails.

## 4. Independent review gate

Review authorization is cryptographic, not a caller-path-and-hash convention.
After the custody-only reservation, an external signer publishes one
mode-`0444`, one-link `hostile_review.json` in the exact mode-`0555` directory
`artifacts/carry_motor/recovery_reviews/review_${RECOVERY_COMMIT}`. Its outer
audit is `causal_carry_motor_recovery_signed_review_v8`; it contains only the
algorithm, frozen key ID, signed payload, and canonical-base64 Ed25519 signature.
The canonical signed payload binds all of the following type-strictly:

- decision exactly `GO`, signer algorithm/key ID, and sequence exactly `1`;
- recovery commit, sole parent, exact four additions, all four source hashes,
  complete clean-checkout/Git-control contract, and source manifest;
- pinned isolated Python runtime, dependency and callable contracts;
- upstream plan receipt, normalization proof, sole allowed transformation, and
  exact v11 claim boundary;
- expected static Slurm/H100 resource contract;
- derived output path, untouched upstream root, and the reservation receipt
  containing the exact root/fit/development/confirmation
  device+inode+mode+link-count+owner identities; and
- the explicit review trust boundary.

The frozen production key ID is
`ed25519-sha256:de00c061da12e04939933da597a399448c3cdc7136e25b29f09d3dbc3d0599d9`.
Only its 32-byte public half appears in source. No production private key,
receipt generator, signing helper, or signature fixture exists in this repo.
The test Ed25519 key is different and production verification rejects it. The
receipt transport SHA-256 is still supplied separately, but it has no authority
without the valid signature. A missing, writable, linked, aliased, wrong-key,
wrong-sequence, wrong-commit, stale-layout, `NO-GO`, expanded, or incorrectly
signed receipt fails before plan or fit. The signed payload is verified only in
its sorted compact ASCII JSON encoding. The outer receipt bytes must equal the
same canonical encoding plus exactly one LF. Base64 must decode to exactly 64
signature bytes and re-encode byte-for-byte, rejecting alternate alphabets,
padding, and nonzero pad bits before Ed25519 verification.

Signature verification proves control of the frozen external private key and
exact receipt bytes. It cannot establish reviewer independence, competence,
diligence, or honesty. Those remain explicit human-governance trust assumptions
and are not converted into a cryptographic claim.

The runtime contract fixes the launcher to
`/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python` and Git to the regular,
non-symlink `/usr/bin/git`. It records the resolved interpreter identity and
SHA-256, Python version, ABI, exact startup flags and `sys.path`, Torch and
Tokenizers versions, exact module paths, and a descriptor-verified dependency
manifest. That manifest includes both package entrypoints and the actual files
used by fitting and loading, including `torch.optim.adamw`, `torch._C`,
`torch.nn.functional`, `torch.nn.modules.linear`, `torch.optim.optimizer`,
`torch.serialization`, and the `tokenizers.tokenizers` native extension. It
binds path, every physical ancestor, SHA-256, mode, size, device, inode, owner,
link count, and times, and
has separate portable-content and full-identity manifest hashes. Module name,
import-spec origin, and `__file__` must agree. Caller override of the Python
launcher is impossible. Slurm must clear the submission environment with the
static `#SBATCH --export=NONE` contract, but that directive is not treated as
the initial trust boundary. The wrapper's exact Linux shebang is
`#!/bin/bash -p`. Linux passes `-p` to `/bin/bash` before the script body, so
privileged Bash ignores `BASH_ENV`, exported `SHELLOPTS`, and exported function
imports before line 1 while retaining the Slurm allocation variables needed for
attestation. The first script command verifies privileged mode and proves
`compgen` is a builtin with `builtin type -t compgen` before any enumeration.
It then proves no function was imported and rejects raw `BASH_ENV`, exported
`SHELLOPTS`, `BASH_FUNC_*`, or other forbidden controls before positional
inputs. Tests invoke this production wrapper with `/bin/bash -p` while hostile
startup files and an exported `compgen` function attempt to write markers; both
must be rejected and neither marker may exist. The wrapper then launches
exactly with `-I -S -B`; it does not import `site`, process `.pth` files, import
`sitecustomize`/`usercustomize`, or honor `PYTHONPATH`. Before any repository
path enters `sys.path` or any repository module is imported, stdlib-only startup
code descriptor-validates Git controls, commit topology, index/tree equality,
the complete closed-world checkout, and the four-source manifest. Only then may
bootstrap replace `sys.path` with the reviewed `train`, stdlib, platform
stdlib/native-extension, purelib, and platlib physical paths. Those exact flags
and paths are revalidated. Bytecode writes are disabled;
thread counts and CUBLAS workspace are fixed; and Git global/system
configuration and replace objects are disabled with receipt-bound values.
Ambient `PYTHONPATH`, `GIT_*`, `SBATCH_*`, BLAS, CUBLAS,
CUDNN, MKL, OpenBLAS, NCCL, PyTorch, TorchDynamo, TorchInductor, and Torch
control prefixes fail on presence, including empty values. Explicit numerical
rejects include `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`,
`NVIDIA_TF32_OVERRIDE`, `CUBLAS_FORCE_TF32`, and PyTorch BLAS/CUDNN preference
variables. The receipt also binds default dtype, float32 matmul precision,
deterministic-algorithm state, CUDA matmul TF32/reduced-precision flags, CUDNN
TF32/benchmark/determinism flags, thread counts, and creation umask `0077`.
Python startup injection, `LD_PRELOAD`, `LD_LIBRARY_PATH`, related dynamic-loader
controls, and Torch deserialization override variables are forbidden. The same
runtime is reconstructed and compared type-strictly before publication.

File hashes do not by themselves bind live Python semantics. At import, before
any command, recovery first exercises the actual in-memory optimizer,
state-load/state-save, and Torch serialization paths so lazy wrappers stabilize,
while restoring the process environment exactly. It then captures the actual
`AdamW` class, `torch.optim.adamw.adamw`, `torch.optim.adam.adam`, inherited
optimizer step/zero-grad methods, inherited `Module` call/device/state methods,
tensor backward, `torch.load`, `torch.save`, every live serialization helper and
their globals, `torch.serialization.safe_globals`, TorchVersion, linear and
cross-entropy primitives, Tokenizer loader and native module, `torch._C`, the
upstream `rollout_episode`, its controller alias and live globals, all carry
model/loss/schedule/initial-state functions, shard merger, evidence builder,
complete upstream validator, CUDA runtime validator, and GPT config class.

The runtime contract records module, qualname, callable type, defining file,
class-method hashes, and recursive semantic-code hashes built from stable Python
code-object instructions, constants, names, flags, arguments, closures, and
exception tables rather than mutable interpreter specialization counters. The
semantic closure recursively binds every reviewed live global and closure cell,
including exact scientific constants, protocol regexes, Torch hook registries,
serialization package/TLS state, nested partials, functions, classes, modules,
and native modules in the upstream, optimizer, module, serialization,
controller, protocol, evaluation, model, and probe modules. Replacement is
rejected by identity; mutable container/object state is also recursively
content-bound. The only excluded globals are Torch's `torch.cuda._initialized`
and `_cached_device_count`, which are native runtime caches expected to
transition only after the one-to-one Slurm/cgroup/GPU attestation; their
externally relevant results remain independently bound by that attestation.

Every sensitive call first proves the live export and every transitive
dependency are still the captured objects and then calls captured methods
directly. Independent replay therefore cannot share an unbound semantic
dependency with construction. Every reviewed dependency is monkeypatched in an
exact `-I -S -B` process and must fail before either fit construction or replay
can reach invalid inputs. Mutable registries and object state receive separate
in-place mutation tests under the same entrypoint gates.

## 5. Immutable recovery plan

The exact recovery root is derived, not selected:

```text
artifacts/carry_motor/recoveries/
  upstream_${UPSTREAM_PLAN_SHA256}_executor_${RECOVERY_COMMIT}/
```

The parent `artifacts/carry_motor/recoveries` is mode `0700`, owned by the
executor user, and is either required with or durably installed with a one-link
mode-`0444` `parent_receipt.json`. The receipt binds the parent device/inode,
mode, owner, exact physical path, installer source-contract digest, and required
`0077` umask. Installation first obtains a nonblocking exclusive `flock` on the
one-link mode-`0600` `.parent_receipt.owner` journal. Its canonical typed bytes
bind purpose, parent path and device/inode, target and stage names, UID/GID,
node, boot ID, PID, process-start token, Slurm JobID when present, and the exact
stage inode after creation. A mode-`0600` `.parent_receipt.stage` is created with
descriptor-relative `O_EXCL`, bound into the journal, written and fsynced,
changed to mode `0444`, hard-linked no-replace to `parent_receipt.json`, and
directory-fsynced. Only after proving both names are the same mode-`0444`
two-link inode is the stage unlinked and the directory fsynced, leaving a
one-link immutable final; then the owner journal is removed and fsynced.

Recovery may clean residue only after acquiring the exclusive lock and proving
the journal's boot/process identity is not live. It may unlink only the exact
stage inode named by that journal. If the hard-link commit already happened, it
may unlink only that stage name after proving the final is the same two-link
mode-`0444` inode. A live owner, foreign or unbound stage, substituted inode,
foreign hard link, malformed journal, or partial canonical
`parent_receipt.json` fails closed and is never deleted or adopted. The
resulting parent and receipt identity are bound into the plan and fit and
revalidated before and after publication.

The derived recovery root must not exist before the custody-only reservation.
Reservation descriptor-creates the root and its fit, development, and
confirmation directories at mode `0700` under umask `0077`, records their
device, inode, mode, link count, owner, group, physical path, and complete
physical-ancestor
chains in a one-link mode-`0444` `layout_receipt.json`, and fsyncs every created
object and containing directory. The root remains mode `0700` while the review
statement is signed. An ancestor symlink, lexical/physical alias, retargeted
ancestor, same-byte root or subdirectory replacement, foreign child, wrong
mode, wrong owner, or layout-receipt substitution fails closed.

Only after the signed receipt validates against that exact reservation may the
planner safely load all eight upstream shards, independently regenerate rows,
normalize the board, recompute the shuffled control, batch schedule, initial
motor state, sentinel identities, and merged feature receipts, and publish one
immutable `causal_carry_motor_recovery_plan_v8` document. Before any parent,
reservation, plan, fit, or artifact creation, both wrapper and executor set and
verify process umask `0077`. Plan publication retains the reserved directory
inodes, creates only one one-link mode-`0444` `recovery_plan.json`, and changes
the same root to mode `0555`; fit, development, and confirmation directories
remain empty mode `0700`.

The recovery plan binds:

- both source contracts and the hostile-review receipt;
- upstream plan, commitment, generator, source, frozen-input, and all eight
  shard identities;
- the complete normalization proof;
- exact checkpoint step, dimensions, token IDs, board, row order, control,
  2,000-update schedule, batch 512, rank 8, learning rate 0.003, weight decay
  0.0001, seed, and initial state;
- the constant and 20-cell nuisance full-board solver bounds, checkpoint and
  convergence rules, exact capacity ledger, immutable width extrapolation,
  selected-state evidence, and no-width-8/no-confirmation selection boundary;
- the upstream merged feature and teacher-metric hashes;
- exact new recovery output paths; and
- explicit safe deserialization behavior; and
- the complete pre-fit downstream evaluation contract in Section 9, including
  both frozen rescue-oracle receipts, cases, denominators, preservation controls,
  thresholds, phase order, and decision labels.

It also binds the reservation layout receipt and physical identities, the
exact static Slurm/H100 request, the signed-review statement, and the recovery
parent receipt. Those bindings are revalidated immediately before and after
plan publication and every later publication or recovery action. The root and
all three subdirectory identities may never be recreated merely because their
bytes or names match.

It also binds a complete upstream custody snapshot, not only content receipts. The
snapshot covers the canonical root, plan, all eight shard directories and
files, the empty fit/development/confirmation directories, and the confirmation
commitment directory and file. Each entry records its exact lexical path,
kind, device, inode, mode, link count, owner, group, size, mtime, ctime, closed
world children, and file SHA-256 where applicable. The complete snapshot is
reconstructed and compared type-strictly immediately before and immediately
after artifact publication and again around final directory sealing. A same-byte
inode replacement, mode change, new child in an empty directory, shard mutation,
or directory substitution is fatal.

Any caller-selected alias, output under the old canonical root, changed budget,
changed shard receipt, changed source, or extra transformation fails closed.

## 6. Safe deserialization

Checkpoint and shard tensor files are bound by exact lexical path, no-symlink
open descriptor, inode/stat identity, and SHA-256 before deserialization.
`torch.load` is called explicitly with `weights_only=True` inside a safe-global
scope containing only `torch.torch_version.TorchVersion`, which is required by
the already sealed runtime metadata. There is no unrestricted-pickle fallback.
Both `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` and
`TORCH_FORCE_WEIGHTS_ONLY_LOAD` are forbidden ambient overrides.

## 7. Fit and publication

The reviewed H100 wrapper has no accepted `SBATCH_*` or resource override. Its
static request and live attestation require account exactly `skattel`, partition
`normal`, one node, one task, four CPUs per task, 32 GiB node memory, six hours,
`Requeue=0`, restart count zero, and exactly one typed
`gpu:nvidia_h100_pcie` allocation. At runtime, both `scontrol show job -o` and
pipe-delimited `sacct` must agree one-to-one on account, JobID, partition, nodes,
tasks, CPUs, memory, time, state, requested TRES, allocated TRES, generic GPU
TRES, and typed GPU TRES/GRES. `SLURM_JOB_GPUS`, `SLURM_STEP_GPUS`, and
`CUDA_VISIBLE_DEVICES` must all identify the same sole GPU by its accepted exact
index, UUID, UUID without the `GPU-` prefix, queried long PCI form, or normalized
PCI form. Capture and validation use the same normalization rule: an eight-digit
PCI domain is accepted only when its leading four digits are zero and is reduced
to canonical `dddd:bb:dd.f` form. A production-capture regression drives the
actual `capture_slurm_h100_attestation` path with controlled `scontrol`, `sacct`,
`nvidia-smi`, character-device, cgroup-membership, `/sys/dev/char`, and PCI
fixtures while mixing equivalent long and normalized PCI selectors. A mismatch
at any source fails.

The reviewed wrapper and signed runtime contract exclude exactly
`evc22,evc26,evc31,evc32,evc36,evc37,evc40,evc43,evc44`. Live `scontrol`
must expose both `NodeList` and `ExcNodeList`. The pinned
`scontrol show hostnames` expansion of `NodeList` must be one exact hostname,
the expansion of `ExcNodeList` must equal that ordered nine-node set, and the
sets must be disjoint. The allocation hostname, both raw expressions, both
expanded lists, and receipts over both lists are serialized in the H100
attestation. Landing on `evc43` or any other excluded node is a hard failure
even if every GPU and TRES field is otherwise self-consistent.

The executor must prove that its Slurm job cgroup authorizes exactly one
openable `/dev/nvidiaN` character device, bind that node's device/inode/mode and
major/minor numbers, map it through `/sys/dev/char` to the same normalized PCI
bus, and bind the cgroup membership path to the same JobID. The pinned
`nvidia-smi` query and `nvidia-smi -L` must expose exactly that one PCI bus and
GPU UUID, whose exact name is `NVIDIA H100 PCIe`, compute capability is `9.0`,
full memory lies in the frozen 80,000--82,000 MiB band, and MIG mode is disabled
with no GPU-instance, compute-instance, or MIG-device identity. Missing,
duplicate, abbreviated, contradictory, untyped, or many-to-one records fail
closed. The wrapper also requires an exact clean recovery checkout, exact
spooled wrapper bytes, the sealed hostile-review receipt, and the sealed
recovery plan. The fit exposes no mutable optimization flags and uses only the
plan's frozen values.

The executor replays and validates the eight upstream shards and fits treatment and
shuffled arms from the same initial state and frozen 2,000-update AdamW schedule.
Those reader arms retain the sealed upstream optimization contract. The calibration
nulls do not inherit a noisy minibatch final iterate as their selected state.

The `constant_bias` null has exactly one scalar, `delta = b1 - b0`, deployed
as `[-delta/2, +delta/2]` at the exact treatment grammar gate. It reads no
hidden value. Its exact production objective is mean full-board, full-vocabulary
cross entropy over every fit row using the deployed `base01` dtype before the
float32 log-sum-exp. The continuous pre-quantization objective is convex in
`delta`. The reviewed solver brackets its derivative on the immutable closed
interval `[-64, 64]` and performs exactly 80 bisection checkpoints for an
interior root. It then rescores the bracket states and both adjacent deployed
dtype levels for each signed half-delta with the exact production arithmetic on
the complete board. It selects
minimum full-board CE, then minimum absolute delta, then the lowest float32 delta.
A deployed quantization level is not required to have zero derivative;
convergence is the frozen sign-bracket plus bracket-width rule. A boundary state
is eligible only with the corresponding KKT derivative sign. A
missing bracket, nonconvergence, or malformed checkpoint is a hard failure; the
last iterate is never a fallback.

The `nuisance_only` null is the singleton
`saturated_op_width_position_v1` family. Its capacity ledger is exact:

- 20 fit metadata cells: `2 operations * (4 + 6 positions)`;
- 20 independent trainable float32 cell deltas and zero intercept parameters;
- rank 20 over the ordered 20-row fit-cell design;
- zero additional interpolation or width-extrapolation parameters; and
- treatment capacity reported separately as `rank*d_model + 3*rank + 2`, while
  the constant null has one parameter.

The signed plan and fit artifact also carry the complete deployment ledger,
recomputed from the bound checkpoint configuration rather than accepted from a
caller:

```text
frozen Shohin base, tied head counted once    125,081,664
rank-8 treatment motor                              4,634
constant-bias null head                                 1
saturated nuisance-only null head                      20
                                                 -----------
combined unique trainable parameters              125,086,319
strict exclusive cap                              150,000,000
remaining headroom                                 24,913,681
```

The base count is independently expanded into the token embedding/tied output
head, all 30 attention/MLP/norm blocks, and final norm. The checkpoint must have
the exact frozen parameter-relevant configuration (`32768` vocabulary, 30
layers, 9 query heads, 3 KV heads, model width 576, FFN width 1536, QK norm on,
and tied embeddings). The treatment formula, both null heads, combined total,
and strict comparison are type-strict. Equality with or any value above
`150,000,000` fails; a caller cannot omit a head or substitute a reported total.

For fit widths 4 and 6 the metadata design is exactly one-hot, so every fit cell
has its own scalar and the former rank-9 basis is not admissible. Each cell must
contain both target classes. Every cell is optimized independently by the same
full-board, full-vocabulary convex derivative-bracketing and float32 checkpoint
rule as the constant null. The artifact serializes every cell's row count, target
0/1 counts, selected delta, CE, derivative, bracket, checkpoint candidates,
candidate receipt, and convergence result, plus complete-board CE, fit-payload
receipt, selected-state hashes, and selected-checkpoint-evidence hash. Any missing
cell, nonconverged cell, altered receipt, or selected state other than the replayed
minimum-CE checkpoint fails closed.

Every null-selection receipt is now downstream of raw model-bound evidence. The
binding covers the frozen base checkpoint and tokenizer hashes, upstream plan,
source contract, all shard receipts, feature merge, exact fit-row manifest, raw
feature tensor payload, deployment vocabulary, and complete parameter ledger.
Validation recomputes that raw-feature payload receipt from the supplied
`base01`, `other_lse`, labels, and the complete ordered non-carry token-ID
vector; changing and re-solving a bfloat16 board while retaining the bound
feature receipt is inadmissible.
The carry-logit tensor must be exact `torch.bfloat16`; a float32 carry-logit
board is inadmissible even when all reported hashes and losses agree. Validation
reconstructs the ordered metadata groups from the canonical fit rows, reruns all
80-step brackets and adjacent deployed-bfloat16 candidates, recomputes each
candidate CE and derivative, selects the deterministic minimum, reconstructs
the 20-cell state tensor, and recomputes full-board CE and every enclosing
receipt. It then requires that exact state to be the serialized deployed state.
`CE=999`, a fabricated but self-consistent state/hash tree, an arbitrary
fit-payload hash, or a non-selected deployed checkpoint therefore fails against
raw replay rather than merely failing a formatting check.

For any deployment width `w`, position is first mapped to normalized position
`t = p/(w-1)`. The width-4 and width-6 cell deltas are independently
piecewise-linearly interpolated at `t`, then combined by the immutable rule
`d(w,t) = d4(t) + (w-4)*(d6(t)-d4(t))/2`. This is exact one-hot lookup on fit
cells and fixed linear extrapolation elsewhere. No hidden residual, prompt text,
token history, style, current carry, target, or operand digit enters the design.

Model-family selection is the preregistered singleton above. Checkpoint selection
uses fit-board full-vocabulary CE only. The fit-width development cells at widths
4 and 6 are a post-fit audit whose exact records and receipt are serialized, but
they cannot change family or checkpoint. Public width 8 and every confirmation
case remain sealed until the family, converged checkpoint, state hashes, fit-cell
strata, and fit-width development receipt are frozen. Width 8 and confirmation
are evaluation-only afterward; neither may trigger reselection, refit, fallback,
or a new checkpoint.

The executor recomputes all retained teacher evidence and diagnostics and
passes the complete upstream v8 payload validator in memory for
the unchanged legacy treatment/shuffled subpayload. Before that legacy
validator, a recovery-owned exhaustive
validator checks every legacy payload field with exact Python types. It rejects
`bool`/`int`, `int`/`float`, mapping-subclass, state-container, tensor-subclass,
fit-report, diagnostic, and nested evidence aliases; requires finite float loss
and accuracy fields; recomputes the expected teacher evidence; and asserts that
its field-coverage set equals the complete frozen legacy schema. It never
publishes a v8 object. The sole output is
`causal_carry_motor_fit_v11_recovery_v7`, with top-level recovery status, both
source domains, upstream plan and shard receipts, normalization proof,
deserialization contract, receipted recovery-parent and layout bindings, live
Slurm/H100 attestation, complete reader-trajectory and converged-null-selection
replay proof, a headerless
legacy scientific fit payload, and recovery-owned constant-bias and
nuisance-only payloads. The
constant payload binds its one-parameter state and SHA-256, initial-state
SHA-256, identical training-row receipt, complete-board objective and selection
evidence, raw-margin threshold diagnostic, and narrow claim boundary. The
nuisance payload independently binds its one-tensor 20-cell state and SHA-256,
zero initial state, exact training-feature and metadata receipts, capacity ledger,
allowed source fields, fit-only width set, no-refit OOD policy, complete-board
objective and per-cell convergence evidence, and narrow claim boundary.
A top-level `canonical` field or v8 audit is forbidden.

Self-consistency is not optimization evidence. Before a fresh candidate may be
sealed, the executor independently starts again from the exact frozen initial
states, replays all 2,000 treatment and shuffled-control AdamW updates, and
re-solves every constant and saturated nuisance full-board convex checkpoint in
the frozen row and cell order. It then
recomputes the complete state dictionaries, state hashes, fit reports, linear
diagnostic, teacher evidence, board, control, raw-margin diagnostic, and every
other published field. The full regenerated tree and every tensor must compare type-strictly
equal to the bytes safe-loaded from the published candidate. The same complete
independent replay is mandatory before accepting an existing mode-`0444`
recoverable candidate and before reporting an already sealed candidate as
valid. A finite, hand-authored, internally consistent artifact and its
self-reported schedule or convergence flag therefore cannot become evidence. The mode-`0600`
partial-serialization stage is never accepted: it is recoverable only under the
exclusive stale-owner rules below, after which the trajectory is rebuilt and
replayed from the initial state.

A deterministic CPU smoke uses real finite tensor fixtures and invokes both the
upstream production `fit_motor` and recovery production `_recovery_fit_motor`
callables for treatment and permuted-control arms. Both arms start from the same
copied initialization, use the same four-update batch schedule and resources,
and must produce type-strictly equal reports and tensor-exact states between fit
and replay. The same smoke invokes the production constant-bias solve twice from
the exact zero-scalar initialization and proves tensor-exact state and evidence
replay plus hidden-value invariance. It also invokes the production nuisance-only
solve twice from the exact zero 20-cell initialization, proves tensor-exact
selected state and evidence replay, rejects any fit row outside widths 4 and 6,
rejects a missing or one-class fit cell, and proves that hidden, prompt, style,
current-carry, target, and token changes cannot alter its metadata. The reader
smoke budget is test-only. The signed H100 reader budget remains exactly 2,000
updates per AdamW arm, batch 512, with no production override; null convergence
uses the separately frozen complete-board rule above.

Publication is recovery-owned and descriptor-bound. The signed reservation
carries device, inode, mode, and link count from each opened directory
descriptor. Before every owner acquisition, stage create, hard-link commit,
cleanup, file chmod, or directory chmod, the executor reopens or revalidates the
complete physical ancestor chain, linked name, descriptor identity, signed
reserved identity, signed filesystem link-count policy, mode, and exact allowed
child set. All mutations are dirfd-relative and there is no rename or replace
operation.

Before touching bytes, the executor obtains a nonblocking exclusive `flock` on
the one-link mode-`0600` `.motor.pt.owner` journal. Its canonical typed document
binds purpose, exact fit directory path and device/inode, target/stage names,
UID/GID, node, boot ID, PID, process-start token, optional Slurm JobID, and the
stage inode after creation. With the signed mode-`0700` fit directory otherwise
empty, the owner creates `.motor.pt.stage` by `O_CREAT|O_EXCL|O_NOFOLLOW` at mode
`0600`, binds its descriptor identity into the fsynced journal, serializes and
hashes through that descriptor, fsyncs it, and changes it to mode `0444`. It then
hard-links no-replace to `motor.pt`, proves target and stage are the same
mode-`0444` two-link inode, fsyncs the directory, unlinks only the stage name,
fsyncs again, and proves the final is one-link. Only then is the stage binding
cleared and owner journal removed. The final canonical name is therefore never
a partial serialization.

An interruption during serialization leaves only the owner journal and its
bound mode-`0600` stage. An interruption after hard-link commit may additionally
leave a mode-`0444` two-name inode. Recovery first acquires the exclusive lock,
validates journal scope and directory identity, and proves the recorded owner is
not live by node, boot ID, PID, and process-start token. It may unlink only the
exact bound stage inode. In the committed case it must first prove stage and
`motor.pt` are the same mode-`0444` two-link inode, then unlink the stage and
prove the immutable final settles at one link. A held lock, live owner, malformed
journal, unbound stage, substituted/foreign inode, foreign hard link, or foreign
final is preserved and fails closed; cleanup never targets it.

Externally recognized states are: empty mode `0700`; closed-world publisher
residue under the owner journal in mode `0700`; one immutable mode-`0444`,
one-link `motor.pt` in mode `0700` after commit but before sealing; or the same
sole artifact in sealed mode `0555`. Publisher residue is never fit evidence.
The mode-`0444` states can be accepted only after safe load, complete typed
validation, independent replay of exactly 2,000 treatment and 2,000 shuffled
updates, and re-verification of runtime, callables, Git contract, Slurm/H100
attestation, recovery plan, signed review, upstream plan, confirmation
commitment, frozen inputs, shard bindings, recovery layout, and upstream custody
snapshot. Only then may the fit directory be descriptor-sealed to mode `0555`.

## 8. Threat model

The fail-closed boundary assumes an attacker or accidental operator may supply
an aliased path, symlinked or retargeted ancestor, dirty checkout, wrong commit,
merge or grandchild commit, additional committed or untracked file, shadow
module, `.pth`/`sitecustomize` startup injection, `BASH_ENV`, exported shell
options/functions, alternate interpreter, unsafe environment, live transitive
callable/global/native-module monkeypatch with unchanged dependency bytes,
modified or wrong-key/noncanonical review receipt, malicious local Git include,
filter, attributes, or fsmonitor controls, same-byte root or subdirectory inode
substitution, linked, mode-changed, or path-retargeted artifact,
partial serialization, concurrent live publisher, stale journal, foreign stage
inode or hard link, crash before or after hard-link commit, finite self-consistent
trajectory forgery, type-aliased Python payload, changed budget, changed board,
changed shard, JSON exponent overflow, numerical/TF32 or Git-control environment
injection, dependency-file substitution, hostile inherited umask, unreceipted
recovery parent, Slurm account/JobID/TRES/GRES/cgroup/PCI/GPU/MIG substitution,
additional normalization, old-root output, or replacement confirmation
generator. The executor must detect these before making or sealing a claim.

The protocol does not claim protection against a compromised kernel, root user,
storage firmware, Git or Python binary whose bytes change after their final
descriptor check, malicious CUDA hardware, SHA-256 collision, or a dishonest
independent reviewer who deliberately signs the exact bad source/runtime. Those
are explicit trust roots. The manifest binds Python extension objects but does
not recursively hash every transitive system DSO selected by the kernel dynamic
loader; the pinned environment, extension identities, backend-state receipt,
and frozen H100 runtime are the boundary for that residual platform risk.
Network availability is irrelevant because execution uses no network source.
Recovery code has no authority to regenerate or inspect the confirmation secret
and no authority to reinterpret a fit as capability.

## 9. Downstream boundary

Fit candidate publication is phase one only. The model family is already the
reviewed singleton saturated null. The full-board fit checkpoint, state hashes,
convergence evidence, and every fit metadata cell are frozen next. Fit-width
development records at widths 4 and 6 are then sealed as an audit-only selection
receipt; they cannot alter the family or checkpoint. Only after those choices and
receipts are immutable may public width 8 be opened for development evaluation.
The single secret confirmation reveal is last. Width 8 or confirmation data in a
`fit_width_audit` split record is a hard failure, and there is no post-reveal refit,
fallback, family change, or checkpoint change.

Arms are base, treatment, `constant_bias`, `nuisance_only`, and shuffled.
Every canonical case requires exactly one result from every arm. The gate accepts
no caller-supplied coverage map, denominator, aggregate count, accuracy, stratum,
case receipt, full-vocabulary receipt, serializer receipt, preservation result, or
gate-off receipt. Its scientific inputs are the canonical case manifest, the
exact case-by-arm raw record matrix, and the raw model-bound fit tensors/state
produced under separately reviewed generator and custody code. It derives a
sorted manifest receipt and sorted record-matrix
receipt itself. Duplicate case IDs, duplicate case-arm results, an unknown result,
a missing result, an extra result, malformed identity, impossible token/logit
pair, or missing output field fails before a decision is computed.

The decision also requires the deployment-vocabulary binding sealed by the fit
authority. It binds the frozen checkpoint and tokenizer, exact output width,
carry token IDs, exact column order `column i = tokenizer token id i`, a receipt
over that complete order, and a receipt over the tokenizer's ID-to-token order.
Every ordinary full-logit row and both gate-off rows must have exactly that
width, and every case-arm record carries the same vocabulary receipt. Making
all three matrices narrower together, changing their token order together, or
changing a local width declaration cannot satisfy the separately bound
vocabulary receipt.

Development has exactly the source-order IDs `development-episode-000` through
`-299`, `development-boundary-000` through `-049`, and
`development-direct-000` through `-011`. Confirmation has exactly
`confirmation-episode-000` through `-255` from the single unchanged
`a0c258e` secret-derived generator opening. Matched rescue and terminal IDs,
branches, operations, widths, terminal positions, targets, expected states, and
answers must equal the frozen preregistered oracles. Each phase also has a
nonempty preservation manifest. The recovery executor cannot substitute itself
as development selector or confirmation generator.

Every fit, matched, development, and confirmation case carries its raw canonical
source input, exact prompt (or, for generated fit rows, the exact prompt/prefix
token identity and source prompt hash), reviewed generator binding, source
receipt, prompt receipt, generator receipt, split name, and split-membership
receipt. Every required case identity field must be present in that raw source
and type-strictly equal to the canonical case; deleting or changing a source
field and recomputing all local hashes remains inadmissible. A complete split
receipt is recomputed from those case records. Its
disjointness receipt is derived from the raw source and prompt receipts across
`fit`, `fit_width_audit`, `matched`, `development`, and `confirmation`; no
caller-supplied disjointness or oracle-exclusion boolean is accepted. Frozen
matched rescue and terminal oracle IDs are selected from canonical source, and
their source/prompt intersections with the fit split must both be empty. A
missing receipt, a rehashed but source-inconsistent case, a forged split entry,
or an oracle declaration without zero-overlap evidence fails closed.

Each raw case-arm record serializes actual token IDs, every full-vocabulary logit
row used to choose them, actual `p/c/r/z` transition fields where required,
serializer token IDs, episode output, motor-gate trace, and separate base/arm
gate-off full-logit rows. Token IDs are recomputed by full-vocabulary argmax with
the fixed lowest-token tie break. The decision derives and serializes per-case
case/record receipts, actual outputs, transition field exactness, serializer and
episode exactness, full-vocabulary output hash, gate-off base and arm logit hashes,
gate-off token identity, motor-fire status, and a receipt over the complete derived
record. Booleans or aggregate counts cannot stand in for these raw outputs.

The `constant_bias` and saturated `nuisance_only` controls use the exact
grammar gate and zero-sum carry-logit deployment defined in Section 7. Their
globally selected full-board states are bound before downstream evaluation. The
saturated arm's immutable normalized-position interpolation and width rule is
used unchanged at every development and confirmation width. Both nulls receive
the complete fit board on every solver objective/derivative evaluation, which is
strictly more favorable optimization access than a noisy final AdamW minibatch
iterate. Neither null receives hidden residuals or post-fit labels for selection.

The model-selection receipt is derived from the bound nuisance fit evidence and
the exact width-4/6 `fit_width_audit` development records. Those records must
cover all 20 fit cells with both targets. The fit evidence must contain every
ordered cell, complete-board CE, recomputable CE and stratum receipts, converged
brackets/checkpoints, and selected-state hashes, with explicit false values for
final-iterate use, width-8 access, and confirmation access. A nine-parameter
basis, missing cell, one-class cell, nonconverged checkpoint, arbitrary receipt,
or final-iterate fallback is inadmissible.

The fit artifact must also report the raw carry margin diagnostic. For raw
margin `m = logit(c1) - logit(c0)`, one constant can rescue every positive while
preserving every negative only when the open interval

`-min(m_positive) < delta < -max(m_negative)`

is nonempty. The report binds both endpoints, whether the interval is feasible,
the deterministic binary-accuracy-optimal constant threshold and its exact
correct/denominator count, and whether the fitted delta lies in the feasibility
interval. This is a calibration diagnostic, not feature-reading evidence.

The sole new calibration diagnostic admitted for this repair is the immutable local
file `artifacts/eval_history/drs_carry_nuisance_audit_20260718_v4.json`, SHA-256
`94bf0b4b61b239601a7677f7badca03ac9b507c3aad6616b80d37f11072c7f68`.
It is a boundary, not a result claimed by this preregistration. It reports that an
operation-by-width fit-optimal calibration changes value-OOD sensitivity from
11/16 to 16/16 and that a binary-margin CE diagnostic reaches 14/16. Those
diagnostics are not the production full-vocabulary objective, were not used to
fit or select any candidate in this repair, cannot authorize the former rank-9
basis, and cannot satisfy any matched, development, confirmation, serializer,
preservation, or mechanism gate.

The first frozen rescue set is the immutable replay artifact with SHA-256
`756911f568c12093f3a303a42525a2519c38187c8eac71f5da3ca06ac1ce3b20`.
Its six cases are fixed as follows:

1. `width_ood_w8-00175`, counterfactual add, answer `177453123`;
2. `width_ood_w8-00207`, normal add, answer `176477219`;
3. `width_ood_w8-00209`, normal add, answer `169264069`;
4. `width_ood_w8-00219`, normal add, answer `187969887`;
5. `width_ood_w8-00219`, counterfactual add, answer `187969888`; and
6. `width_ood_w8-00242`, normal add, answer `164377525`.

A targeted carry-commit rescue requires treatment exact on 6/6, shuffled exact
on 0/6, treatment at least two cases better than both `constant_bias` and
`nuisance_only`, and zero new prefix divergences before the terminal failure
location in every arm.

The second frozen oracle is the superseding matched-width sweep
`artifacts/eval_history/drs_terminal_width_sweep_v2_w2_w10_20260718_mps.json`,
SHA-256
`db6056e66310ed7d56509403d40f7549d016294a014c0c4527173b4005210520`.
It supersedes the v1 width sweep, SHA-256
`c9670853040349cce4eb4f89c5d5d8381d7b25494ff4428fd873fc2b7be6098d`,
because strict-parser nulls overcounted field errors. The earlier manual probe
`artifacts/eval_history/manual_drs_carry_serializer_probe_v2_20260718_mps.json`,
SHA-256
`b1cafe345bad726517e4c426596c691bf3ae1133d93619af581927ca7a336806`,
is retained only as historical provenance and has no decision authority. The
superseding sweep holds lower-digit history fixed within each of widths 2-10 and
changes only the terminal operand digits: positive uses LSF digits `9,8` and the
matched negative uses `2,3`.

The frozen parent is 0/9 exact for positive terminal transitions and 0/9 exact
for positive serializers, but the failed fields are not one carry class. At
widths 2-3, `c` and `r` are correct and only `z` fails. Widths 4, 5, and 7 are
`c`-only failures. Widths 6, 8, and 9 fail both `c` and `r`. Width 10 loses
broader `p,c,r,z` control. Negative `c` is correct at all 9/9 widths. On the
matched negative arm, the serializer is exact at widths 2-6 and fails at widths
7-10; transitions are exact at widths 2-5 and 7, and fail at widths 6 and 8-10.

The frozen residual-swap diagnostic is
`artifacts/eval_history/drs_terminal_carry_residual_swap_w2_w10_20260718_mps.json`,
SHA-256
`4183b8c381e559b23c41b88c8c8cc3b3d0e0b41c03b3dea4786df98a7676590f`.
Teacher-forced layer-29 swaps separate positive from matched negative carry in
8/9 widths: widths 2-5 and 7-10; width 6 is inverted. In ordinary unpatched
decoding, the positive `c=1` logit exceeds `c=0` only at widths 2-3. This is a
calibration hypothesis, not autonomous motor, transition, capability, or
reasoning evidence.

The immutable pre-fit pairwise calibration audit is
`artifacts/eval_history/drs_carry_constant_bias_audit_20260718.json`, SHA-256
`7f2eef8843eb686c2b63683ab7f11a248b5e1b8c8a4358c936a6c2d49326b7b3`,
over source probe SHA-256
`c3c2d0b037852cb57d54e1f147d445d27093a8548b965c41466e81bcc1a27778`.
At layer 29, raw pairwise c0/c1 choice is 32/40 (target 0: 13/20;
target 1: 19/20). The favorable state-independent threshold is 35/40; its
frozen representative delta is `-0.7841806411743164` (target 0: 18/20;
target 1: 17/20). No perfect constant exists: the positive-required open lower
bound `0.6561751365661621` exceeds the negative-required open upper bound
`-1.1492173671722412`. This audit binds no probe-code bytes and is pairwise and
grammar-gated only. Its 35/40 score is neither a full-vocabulary nor an
autonomous gate and cannot replace independent constant-arm fit, state,
trajectory, per-stratum, or downstream receipts.

Two unreviewed one-off nuisance calibrations are retained only as confound
diagnostics. Allowing an oracle-selected independent delta at each width uses
OOD labels and reaches 38/40; it is explicitly inadmissible. A fit-only global
delta `-0.7778145075` reaches 15/16 on fit and 35/40 overall. Fit-only op deltas
`add=-0.7778145075` and `sub=-0.4535870552` also reach 35/40 and 7/8 on public
width-8 OOD. Fit-only op-by-width deltas reach 16/16 fit and 15/16 same-width
value OOD; their one-off linear width-8 extrapolator reaches 6/8 but was not
preregistered before that public OOD result and has no decision authority. An
earlier fit-width affine guidance calculation reached 35/40. These values bind
no audit implementation or artifact and must be independently reproduced. They
prove that calibration is a serious confound, not that any null is a residual
reader. Public width-8 OOD is now inspected and cannot serve as an unopened
gate. They cannot select the replacement comparator. The saturated singleton
family, 20-cell capacity ledger, complete-board optimizer/checkpoint rule,
immutable extrapolation, and no-refit rule are frozen by this source before
width-8 evaluation and the still-sealed confirmation reveal.

The separate one-off pairwise reading over the 18 terminal-width residual-swap
cases is deliberately not numerically bound here and has no gate authority.
Development must independently recompute constant and nuisance outputs over all
18 cases, report every width and field under full-vocabulary decoding, and
retain the result as a diagnostic only.

The strongest calibration null is defined pointwise as the better accuracy of
`constant_bias` and `nuisance_only` for the metric and stratum being gated.
Every strongest-null test is also expanded into separate treatment-over-constant
and treatment-over-nuisance checks. These checks are noncompensatory: a failure
against either null in one required stratum cannot be offset by another metric,
phase, width, target, operation, or pooled score.

Matched evidence is partitioned into the six exact rescue episodes and the 18
terminal cases. Positive and negative terminal transition and episode summaries
are derived separately. Positive treatment transitions and episodes must be exact
at widths 2-10; the aggregate transition and episode gain over each null is at
least 2/9. For every positive width, each of `p`, `c`, `r`, and `z` is
reported and treatment must exceed each null by at least 15 points. On every
matched negative width, treatment must be exact for all four fields, carry `c`
must be preserved in all five arms, and the five established negative serializers
at widths 2-6 must remain exact in all five arms. Treatment serializer output must
be exact separately for every positive and negative width. It must exceed each
null at every positive width and at negative transfer widths 7-10. The six rescue
episodes retain the exact 6/6 treatment, 0/6 shuffled, and at-least-2/6 gain over
each null rules. No matched case or denominator is caller supplied.

For development and confirmation, the raw manifest must contain both targets in
every `(operation,width,position)` transition stratum and at least two such
strata. Serializer strata are `(width,target)`, require both targets for each
width, and include at least two widths with a post-freeze unseen width. Transition
fields `p/c/r/z` are gated separately for every
`(operation,width,position,target)` stratum at a 15-point treatment gain over
each null. Episode output is gated for every
`(regime,operation,width,target)` stratum at a 20-point gain. Serializer output
is gated for every `(width,target)` stratum at 15 points, and full-vocabulary
token output is gated for every `(regime,width,target)` stratum at 15 points.
Pooled metrics cannot repair any failed stratum.

The pooled development and confirmation gates remain additional requirements:
treatment next-carry accuracy is at least 95% and at least 15 points over base,
shuffled, and both nulls; one-step state, full-vocabulary output, serializer
transfer, and unseen-width output are at least 15 points over all comparators;
autonomous episode output is at least 20 points over all comparators. Development
additionally requires at least 25/50 boundary-cycle cases with the frozen 8/50
gain over each null, and at least 8/12 direct interactions with the frozen 2/12
gain over each null.

Fit-width regression is recomputed only from the exact `fit_width_audit` width-4
and width-6 records and may not exceed two points relative to base. Each matched,
development, and confirmation phase has explicit preservation records. Every arm
must preserve their expected full-vocabulary output, produce no motor fire, and
have exact base/arm gate-off token and full-logit identity. In addition, every
case-arm record in the complete matrix must have exact gate-off token and logit
identity, with both logit hashes serialized. A missing serializer, transition
field, episode output, full-vocabulary logit row, gate trace, or gate-off side is
a malformed record, not a zero or a changed denominator.

The gate never compares two caller-provided 64-hex strings. It computes the
manifest, record-matrix, selection-record, metadata-stratum, checkpoint, per-case
record, full-vocabulary, serializer, and gate-off hashes from canonical content.
The exact cross product of manifest IDs and five arms defines coverage and every
denominator. A one-stratum confirmation, missing fit cell, duplicate case, missing
arm, extra case, impossible count, or arbitrary receipt therefore cannot reach a
GO decision. Both development and the single confirmation remain mandatory.

Decision labels are frozen: fit results can support only
`fit-mechanics-only`; carry-actuator evidence only
`writer-actuator-repair-only`; independent all-width serializer evidence only
`serializer-length-transfer-only`; the two rescue sets only
`targeted-carry-commit-repair-only`; and a full pass only
`mechanism-go-not-general-reasoning-proof`. No fit loss, teacher-forced metric,
linear diagnostic, or fit accuracy alone may support a capability or reasoning
claim. The 40/40 residual-swap direction is causal directional evidence only;
it is not autonomous reasoning and cannot substitute for the constant-bias
or nuisance-only controls. No fit result, development score, confirmation score, mechanism
conclusion, autonomous capability, or reasoning claim is established by this
preregistration itself.

## 10. Required CPU gates before review

The exact recovery source must pass:

- normalization success with exactly two frozen key-type differences;
- rejection of non-histogram, count, extra-key, duplicate-key, and scalar-type
  rewrites;
- path alias, symlink, receipt, shard, source, and executor substitution tests;
- physical ancestor-chain, reserved-root/subdirectory identity, ancestor
  symlink, retarget, and same-byte root-replacement tests;
- sole-parent/four-addition history, extra-file, grandchild, and shadow-module
  rejection tests;
- malicious local Git include-path, clean filter, process filter,
  `info/attributes`, and fsmonitor regressions with proof none is executed;
- exponent-overflow, duplicate-key, recursive-container, non-string-key,
  nested-key-type, and recursive finite strict-JSON rejection tests;
- pinned-interpreter, actual Python/native dependency-manifest, startup-flag,
  numerical-backend, TF32/BLAS override, and Git-control environment tests;
- exact `#!/bin/bash -p` production boundary, privileged-mode and builtin-identity
  proof before enumeration, hostile `BASH_ENV` and exported-`compgen` marker
  nonexecution, raw startup-control rejection, pre-import source-gate ordering,
  and isolated `-I -S -B` startup-injection tests;
- one exact isolated process per reviewed optimizer functional/method,
  inherited Module state method, serialization helper/global, native module,
  rollout function/global, and scientific validator monkeypatch, proving both
  construction and independent replay fail before execution;
- complete upstream custody snapshot tests, including all empty directories,
  modes, and same-byte inode replacement;
- frozen-budget, old-root output, and extra-transformation rejection tests;
- confirmation-generator substitution rejection;
- explicit weights-only/TorchVersion loading and ambient-override rejection;
- secure-umask and durable staged recovery-parent receipt installation, live
  publisher exclusion, dead owner/stage recovery, foreign-stage preservation,
  partial-canonical refusal, foreign-parent, alias, and identity-substitution
  tests;
- immutable closed-world plan publication tests;
- signed mode/link/device/inode descriptor identity, link/mode/path-retarget
  races, exclusive publisher ownership, staged hard-link no-replace publication,
  concurrent live-writer exclusion, exact stale-stage recovery, foreign-inode
  refusal, and post-commit two-link crash recovery to one immutable final;
- forged-but-self-consistent mode-`0444` candidate rejection and source-order
  proof that fresh, recoverable, and sealed acceptance all require an
  independent exact 2,000-update reader trajectory replay plus complete
  constant and 20-cell null solver/checkpoint replay;
- robust `scontrol`/`sacct` parsing plus a real production-capture path driven by
  controlled cgroup/device/sysfs/long-PCI/normalized-PCI fixtures, exact `skattel`
  account, JobID,
  partition, node, task, CPU, memory, time, requeue, requested/allocated TRES,
  typed GRES, exact `NodeList`/expanded `ExcNodeList`, all nine bad-node
  exclusions including `evc43`, excluded-node allocation rejection, cgroup
  device authorization, PCI bus, UUID, GPU type, capability, memory-band, MIG
  identity, and visible-device substitution tests;
- production-key signed-review binding, signer-sequence strictness, and proof
  that the separate test key has no production authority, plus canonical
  signed-payload/outer-receipt bytes and canonical-base64 zero-pad-bit tests;
- exhaustive legacy payload scalar/container type-alias rejection tests;
- v11-only dual-provenance schema tests, including independently receipted and
  replayed constant-bias and nuisance-only states;
- exact four non-v9 source-path topology and regenerated source-manifest tests;
- a deterministic four-update CPU smoke through real upstream fit and recovery
  replay callables for matched treatment/control initialization and schedules,
  plus exact full-board constant-bias and saturated nuisance selected-state,
  per-cell, objective, convergence, and checkpoint-evidence replay from raw
  model-bound bfloat16 evidence;
- adversarial proof that the constant arm cannot condition on hidden values,
  width, prompt, or token history; exact grammar gating and gate-off full-logit
  identity; raw-margin feasibility and optimal-threshold diagnostics; and a
  decision fixture that fails whenever `constant_bias` matches treatment;
- adversarial proof that nuisance-only state contains exactly 20 saturated
  op/fit-width/position deltas, its 20-cell fit design has rank 20, its
  extrapolation has zero learned parameters, it cannot read residuals, prompt
  text, style, current carry, target, or token history, rejects OOD-width fit or
  selection access, and causes decision failure when it matches treatment
  globally or in any required stratum;
- exact regressions for arbitrary equal receipts, one reported stratum,
  impossible caller counts, a missing fit cell, duplicate manifest and result
  cases, missing and extra case-arm results, width-8 selection leakage, the
  former under-capacity nine-parameter basis, a nonconverged final iterate,
  `CE=999`, fabricated selected states and hashes, float32 fit logits,
  non-deployed selected checkpoints, narrower mutually consistent output and
  gate-off matrices, changed token order, a combined ledger at or above 150M,
  forged/missing source and split receipts, caller oracle declarations, real
  oracle/fit prompt overlap, missing full-vocabulary logits, missing serializer
  output, and incomplete gate-off evidence;
- type-strict plan binding and mutation rejection for the six replay rescues,
  all 18 matched width-sweep cases, nine per-width positive terminal rescues,
  five negative serializer-preservation widths, separately reported width-7+
  serializer readout, all five arms, 300/256 episode denominators, matched
  preservation suite, fit-width regression, per-width `p/c/r/z`, serializer and
  full-vocabulary output, derived gate-off-logit receipts, noncompensatory
  treatment-over-strongest-null gates, complete matched/development/confirmation
  transition and episode strata, and frozen decision labels;
- warning-clean CPU Pytest, Ruff, Python compilation, `bash -n`, and whitespace
  checks.

Passing these local CPU gates does not authorize a commit, installation,
signature, job submission, or H100 execution. COMMIT / INSTALL / SIGN / H100
remain NO-GO. Only a fresh independent exact hostile-review receipt can make the
reviewed recovery commit eligible for the separately controlled production
steps.
