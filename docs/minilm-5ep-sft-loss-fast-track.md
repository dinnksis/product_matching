# MiniLM-5ep SFT loss fast-track v1

This is the explicitly authorized scope reduction for
`minilm_5ep_sft_hparam_search_v1`:

1. finish the already materialized `classifier_dropout` coordinate;
2. reuse its seed-42 BCE winner;
3. do not run `max_grad_norm` challengers and do not claim a metric for that axis;
4. run the four declared non-BCE losses;
5. run the existing conditional balance×focal overlay and loss-LR refinement
   only when their frozen v1 triggers fire;
6. stop after loss-LR refinement. ODS and the original three-seed/runtime
   confirmation are outside this controller. A separate protocol may use one
   additional seed later.

The original plan, schema-v1 locks and schema-v2 implementation are not edited.
The fast-track policy pins their canonical/file SHA-256 values. A separate
reviewed trusted-local authority,
`configs/minilm_5ep_sft_loss_fast_track_v1.freeze.json`, pins the fast-track
policy, support module, all wrappers, controller, tests and this protocol. It
is never generated or updated during execution: any source drift fails before
receipt creation, lock loading or a subprocess.

Before the receipt exists, only the exact schema-v1 history through
`classifier_dropout` is accepted. Any schema-v2, loss, confirmation,
`max_grad_norm`, future, duplicate or undeclared boundary lock fails closed.
Every schema-v1 lock is reloaded with the frozen strict loader, and every
preserved stage summary must bind the final normal/boundary lock for that
stage. After dropout is complete and every output is exactly synced to
`sft_exps`, the controller creates a write-once
`max_grad_norm_skip.receipt.json` plus a canonical immutable snapshot of the
root dropout summary. The receipt freezes the exact selected parent recipe,
source, predictions, completion/config/Sheets artifacts, metrics, source
summary and lock, detached freeze authority, all prior lock/summary hashes and
the reconstructed kernel-slug union. Its skip statement is explicit:
`evaluated=false`, `metric_claim=null`, `new_kernels=0`, inherited
`max_grad_norm=1.0`.

The primary loss screen creates exactly four kernels:

- `balanced_binary_bce`;
- `balanced_category_class_sqrt_bce`;
- `balanced_category_class_bce`;
- `focal_bce_gamma2_scale4`.

The tuned BCE anchor is reused without a kernel. The existing primary family
still reserves Holm `m=5` for one possible overlay. The overlay is materialized
only if both the best balance loss and focal loss have strictly positive raw
IID deltas over tuned BCE. The loss-LR line is materialized only if the best
non-BCE winner improves IID macro AP by strictly more than `0.002`; it tests the
existing `0.5×` and `2×` points around the inherited LR. IID macro AP remains
the only selection metric; hard/OOD remain diagnostic.

Plan the next action without subprocesses:

```bash
.venv/bin/python scripts/continue_minilm_5ep_sft_loss_fast_track.py --plan-only
```

After reviewing the plan, continue sequentially on Kaggle:

```bash
.venv/bin/python scripts/continue_minilm_5ep_sft_loss_fast_track.py --submit
```

`--submit` is the only remote mutation switch. It keeps the existing
generator → dry-run → sequential `--submit --wait` → summarizer order, never
uses force/retry/fan-out flags, and requires exact Google Sheets sync. There is
no ODS command in the fast-track implementation.

The execution wrappers are deliberately narrower than the frozen core CLIs.
Generator and summarizer require `--stage-lock` and reject stage-only mode.
The launcher accepts exactly `--dry-run` or `--submit --wait`; it rejects
status/background/no-wait/force/retry/fan-out modes. All three accept only
schema-v2 `loss_primary`, `loss_overlay` or `loss_lr_refine` locks.

Once a fast-track loss lock exists, use the fast-track wrappers whenever a
core loader must consume it:

```bash
.venv/bin/python scripts/create_minilm_5ep_sft_loss_fast_track_notebooks.py \
  --fast-track-policy configs/minilm_5ep_sft_loss_fast_track_v1.json \
  --fast-track-receipt reports/minilm_5ep_sft_hparam_search_v1/fast_track/max_grad_norm_skip.receipt.json \
  --fast-track-freeze-manifest configs/minilm_5ep_sft_loss_fast_track_v1.freeze.json \
  --plan configs/minilm_5ep_sft_hparam_search_v1.json \
  --stage-lock PATH_TO_LOCK

.venv/bin/python scripts/run_minilm_5ep_sft_loss_fast_track_kaggle.py \
  --fast-track-policy configs/minilm_5ep_sft_loss_fast_track_v1.json \
  --fast-track-receipt reports/minilm_5ep_sft_hparam_search_v1/fast_track/max_grad_norm_skip.receipt.json \
  --fast-track-freeze-manifest configs/minilm_5ep_sft_loss_fast_track_v1.freeze.json \
  --plan configs/minilm_5ep_sft_hparam_search_v1.json \
  --stage-lock PATH_TO_LOCK --dry-run

.venv/bin/python scripts/summarize_minilm_5ep_sft_loss_fast_track.py \
  --fast-track-policy configs/minilm_5ep_sft_loss_fast_track_v1.json \
  --fast-track-receipt reports/minilm_5ep_sft_hparam_search_v1/fast_track/max_grad_norm_skip.receipt.json \
  --fast-track-freeze-manifest configs/minilm_5ep_sft_loss_fast_track_v1.freeze.json \
  --plan configs/minilm_5ep_sft_hparam_search_v1.json \
  --stage-lock PATH_TO_LOCK
```

For another trusted Python workflow, including the separate one-additional-seed
check, the public API is:

```python
from minilm_5ep_sft_loss_fast_track_support import patched_loss_predecessor

with patched_loss_predecessor(
    policy_path=POLICY,
    receipt_path=RECEIPT,
    freeze_manifest_path=FREEZE_MANIFEST,
):
    # All adaptive.materialize/read_lock and generator.load_campaign_lock calls
    # that consume a fast-track lock must remain inside this context.
    ...
```

The exact dispatcher API for subprocess integrations is
`run_forwarded_main(main, argv, policy_path=..., receipt_path=...,
freeze_manifest_path=...)`; wrapper consumers can preflight one lock with
`load_scoped_loss_lock(plan_path=..., stage_lock_path=..., policy_path=...,
receipt_path=..., freeze_manifest_path=...)`. The context validates the exact
policy, detached freeze manifest, receipt, core hashes and archived dropout
authority. It changes the loss-screen predecessor and temporarily adapts the
archived completion's exact seven-field prepared-data provenance to the frozen
core validator's older three-field projection. The adapter accepts only
306,669 full-human pairs, 711,304 items, positive rate
0.26131105524197096, label-source count `unspecified=306669`, prepared-pairs
SHA-256 `001bb234...6169a3`, prepared-items SHA-256
`5491ebbf...4511e0`, and the unchanged no-sampling/unit-weight report. It then
calls the unchanged frozen validator and requires its canonical result. Both
process-local changes are restored on normal and exceptional exit. A fast-track
primary lock intentionally fails the unpatched core loader outside this
context.

## Kaggle short remote identities and one-time recovery

Kaggle rejected the first primary-loss `SaveKernel` request before creating a
remote run because the generated slug/title exceeded 50 characters. The
experiment label and all recipe/family hashes remain unchanged. Inside the
same fast-track context, only the three loss modes now project their remote
slug and title to
`pm-m5-{lp|lo|lr}-{first 24 family-SHA hex}-s{seed}-v1`. Slug and title must
be identical, lowercase-hyphen-safe, unique within the lock, and at most 50
characters. Confirmation identity is deliberately unchanged. Every fast-track
materializer performs a strict post-write lock reload, so a long or aliased
remote identity fails before notebook generation or Kaggle use.

The already-created zero-kernel skip receipt remains immutable and still binds
the exact prior-18 ledger. Its former detached freeze is preserved byte-for-byte
at
`configs/minilm_5ep_sft_loss_fast_track_receipt_freeze_db7165.json`; the current
freeze pins that legacy authority rather than rewriting the receipt.

Fresh read-only Kaggle evidence for the rejected primary identities is stored
at
`reports/minilm_5ep_sft_hparam_search_v1/fast_track/recovery/long_slug_savekernel_absence.receipt.json`.
For all four exact owner/slug identities, authenticated owner-list searches
returned no match. The status calls were inaccessible and the read-only files
endpoint returned 403, which is recorded honestly rather than interpreted as a
remote output count. The same receipt binds the failed `submit_wait` return
code, all four overlength staging metadata files, and zero local output
directories. No mutating Kaggle command appears in the audit.

Before changing local authority, run the strictly local preflight:

```bash
.venv/bin/python scripts/recover_minilm_5ep_sft_loss_fast_track_short_slugs.py \
  --preflight
```

After reviewing its exact move ledger, apply the one-time local recovery:

```bash
.venv/bin/python scripts/recover_minilm_5ep_sft_loss_fast_track_short_slugs.py \
  --apply
```

`--apply` never calls Kaggle. It is a resumable local transaction, serialized by
an exclusive process lock. Before the first move it installs the exact preflight
as an immutable intent. Every move uses an atomic no-replace rename and gets its
own write-once journal receipt. On restart, each source/target pair is reconciled
against the preflight file SHA or complete directory-tree ledger: an exact
target plus its journal is complete, an exact source plus an absent target is
continued, and missing, conflicting, symlinked or ambiguous state fails closed.
An injected crash after any one of the 12 moves is covered by regression tests.

The audit copy, journal records, relocated provenance manifest, replacement
sidecars/lock and final completion use a resumable same-directory pending file
followed by atomic no-replace installation. Thus crashes after archive creation,
during a copy, immediately before or inside rematerialization, after a complete
replacement, and during completion publication are idempotently recoverable by
running the same `--apply` command again. Existing targets are never overwritten.
The frozen materializer itself is only process-locally wrapped for these atomic
writes, and its original functions are restored on normal and exceptional exit.

Every recovery authority file must also be a regular file with exactly one hard
link. Existing pending/final files are opened relative to a no-follow parent
descriptor; their `lstat` and `fstat` device/inode identities and link counts are
matched before reads, appends, permission changes and installation, then checked
again after writes. The same rule is enforced for every preflight, journal and
completion record, both relocated manifests/locks, the active replacement lock
and manifest, and every file in the relocated and active provenance trees. An
empty or exact-prefix pending hard link and an exact-content final hard link all
fail before the linked victim is written or chmodded. The already-published
completion remains immutable and is accepted under its exact historical freeze
payload only when its complete payload hash is the pinned applied-generation
hash; all newly created test/recovery completions must bind the current freeze.
If a crash lands after a complete pending payload has already been chmodded to
its final read-only mode, replay opens and binds that file read-only and proceeds
directly to atomic no-replace installation. Only an incomplete writable prefix,
or a complete payload that still has its initial writable mode, is reopened for
append/chmod. Fault injection at every one of the 25 `pending_complete`
boundaries verifies successful resume followed by an identical idempotent
replay.

The rejected trusted-provenance manifest is preserved byte-for-byte under
`forensic/` and is explicitly evidence only: its historical absolute paths are
not treated as an authority after relocation. Recovery also creates a distinct
canonical manifest next to the archived old lock, rewrites only `archive_dir`
and snapshot paths into the recovery archive, recomputes its payload SHA, and
reloads the old lock with the frozen core under the old long-identity projection.
Completion binds both roles and proves that the relocated old authority does not
resolve through the newly created active sidecar.

After the archive journal is complete, the script rematerializes the same four
experiment/recipe/family identities with 40-character remote slugs and strictly
reloads the new lock. Preflight, per-transition journals and completion are
immutable and revalidatable. Resume with `--plan-only`, inspect the short
identities and dry-run, and only then use the ordinary explicit `--submit`
command.
