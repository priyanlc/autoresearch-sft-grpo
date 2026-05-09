You are running on a fresh RunPod A100 80GB pod inside the autoresearch
SFT-only training pipeline. The human has already cloned the repo, checked
out the main branch, and activated a uv venv (.venv/bin/activate sourced).
data/ folder contains train.csv and test.csv.

If `/workspace` is MooseFS-backed (`df -T /workspace` shows `fuse`), the
venv and uv cache MUST live on local overlay disk (e.g. `/root/`), not on
`/workspace` — wheel extraction is metadata-heavy and wedges indefinitely
on FUSE/MooseFS. See FRICTION.md F-007. Recovery if the venv is on
`/workspace`: `rm -rf /workspace/.../.venv /workspace/.cache/uv`, then
`uv venv /root/venv-autoresearch`, `ln -s /root/venv-autoresearch
/workspace/autoresearch-sft-grpo/.venv`, `export UV_CACHE_DIR=/root/uv-cache`,
then proceed.

This is an autonomous run — you have permission to install, download,
authenticate, and execute training without confirming each step. STOP
only when the prompt explicitly says STOP, when a pre-flight check fails,
or when something novel breaks. If `--dangerously-skip-permissions`
refuses to start (Claude Code will not run as root), see
`docs/autoresearch-handoff.md` for the non-root user setup, then stop
and ask.

Read program.md first, then BRANCH_NOTES.md to confirm the locked floor
is METRIC 0.5333 at commit c1bb0a6 (current best on main is METRIC 0.6000
at T2.8 / c4a9d1c — see STATUS.md and results.tsv). The 0.5333 figure is
the regression bar; do not raise it without explicit instruction. Then:

1. Confirm the pod-requirements table at the top of runpod-setup.md and
   the Pre-flight verification block in program.md (C-1 BF16 support,
   C-2 transformers==4.51.3, C-3 Nemotron transformers_modules cache).
   Verify `$HF_TOKEN` and `$WANDB_API_KEY` are exported in the env. Verify
   the active venv is NOT on a MooseFS/FUSE filesystem (`readlink -f
   $(which python)` should resolve to a local path like `/root/...`, not
   `/workspace/...`, when `/workspace` is FUSE — see top-of-prompt MooseFS
   note + FRICTION F-007). Abort and ask if any check fails. Do NOT try to
   fix preconditions yourself.
2. Install deps: `bash bootstrap.sh` (handles torch + ninja + mamba_ssm
   + causal_conv1d), then `uv pip install -r requirements.txt`.
   `causal_conv1d` IS required at install time — transformers'
   `check_imports` does AST-level static import checking on
   modeling_nemotron_h.py and rejects the conditional `from causal_conv1d
   import ...` even though it sits inside an `if is_causal_conv1d_available()`
   guard at runtime. See FRICTION F-009 / T1.14. The runtime fast path is
   still force-disabled at train.py:398, so causal_conv1d's CUDA kernels
   are present but never called.
3. Authenticate non-interactively: `hf auth login --token $HF_TOKEN` and
   `wandb login $WANDB_API_KEY`. (Note: W&B is currently disabled in
   train.py via `report_to='none'`; the login is harmless and reserved
   for future re-enablement.)
4. Run pre-flight: `python check_install.py` (expect "All checks passed";
   `causal_conv1d` should now be reported with a version, not as
   informational-absent — that wording was T1.9-era and is gone after
   T1.14) plus C-1/C-2/C-3 from program.md § Pre-flight verification.
5. STOP if any pre-flight check fails — log to FRICTION.md and ask me
   before proceeding. If the model fails to load with
   `HybridMambaAttentionDynamicCache` errors, clear
   `~/.cache/huggingface/modules/transformers_modules/` and retry once
   (F-001 cache poisoning); if it still fails, STOP and ask.
6. Otherwise: run `python prepare.py` with Bash `run_in_background=true`
   (downloads ~60 GB BF16 weights, ~10–30 min on a fresh pod) and use
   the Monitor tool to stream stdout until completion. Then run
   `python train.py` the same way (multi-hour). Do NOT run either in
   the foreground — the Bash tool's 10-minute max timeout will kill them.
   GRPO is wrapped in try/except per F-002 and auto-falls-back to
   SFT-only on the known Mamba/MoE tensor mismatch — expected, not a
   regression.
7. Capture validation outputs per program.md § Validation Contract.
   train.py prints METRIC, per-category accuracy (`<qtype>: 0.xxxx`
   lines), Peak VRAM, and Time on stdout — parse those. (W&B URL and
   tokens/sec are not currently emitted; skip them.) Append a STATUS.md
   block per program.md § 2 (append-only — do not overwrite prior
   entries) and a row to results.tsv. Then run the adapter-on-fresh-
   BF16-base sanity check from a *separate* Python process: load the
   adapter from OUTPUT_DIR onto a fresh BF16 base using the same
   MODEL_ID and `torch_dtype=torch.bfloat16, device_map='auto',
   trust_remote_code=True` pattern from train.py:388-394, run a sample
   inference on one val example, and verify the output is non-empty
   and structurally plausible. Log the result. (Note: program.md does
   not include a ready-made snippet — write the script yourself; the
   reference load pattern is in train.py.)
8. Verify METRIC ≥ 0.5333 (the locked floor; current best on main is
   0.6000 at T2.8 — see STATUS.md). If below 0.5333, STOP — this is a
   regression; do not start Tier 2 work. Otherwise stop and report.
   Do NOT start Tier 2 changes until I confirm the result, even if you
   beat 0.5333 — some Tier 2 has already landed (T2.7 reverted, T2.8
   kept), so new sweeps need explicit go-ahead.

Branch hygiene per program.md: one commit per change ID, T2.x prefix in
commit messages, revert-not-fix-forward on regressions. FRICTION.md is
the first place to look and the first place to write when anything novel
breaks.
