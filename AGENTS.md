# GODMAX — working agreement for agents

GODMAX is a JAX halo-model framework for cosmological cross-correlations (tSZ, kSZ, tau, CMB
lensing, galaxy clustering, weak lensing), used for real analyses whose outputs get published.
The expensive failure mode here is **not a crash — it is a plausible wrong number**.
Everything below exists to prevent that.

This file is the Codex entry point. `CLAUDE.md` is its Claude Code counterpart and says the
same things; the knowledge base, invariants, tooling and git hooks are shared and identical.

---

## Codex does not run the Claude lifecycle hooks. Run their checks yourself.

Under Claude Code, three hooks fire automatically: staleness is injected at session start,
each edit is routed to its owning invariants, and unverified documents are flagged at the end.
**None of that happens in Codex.** Nothing will tell you the knowledge base is stale, and
nothing will remind you to update it. So, every session:

```bash
# 1. START HERE — what do I already not trust?
python tools/kb/kb.py status
python tools/kb/kb.py stale

# 2. BEFORE editing anything — who owns this code, which rules apply?
python tools/kb/kb.py which <files you will touch>

# 3. WHILE working — run the executable rules
python tools/kb/kb.py invariants --check

# 4. BEFORE you finish — close the loop, or the push gate will block
python tools/kb/kb.py verify --doc <id> --evidence <ledger>
python tools/kb/kb.py journal "<what changed and why>" --agent <role> --invariants INV-…
```

Skipping step 1 is how an agent acts confidently on a document describing code that someone
changed on the cluster last week.

**What is still mechanically enforced under Codex:** all four git hooks
(`pre-push`, `post-merge`, `post-checkout`, `post-rewrite`). The pre-push gate blocks a push
whose commits touch the scope of an unverified knowledge document, or that has no journal
entry, or that fails a blocker invariant — exactly as under Claude Code. First time on a
machine: `bash tools/kb/install.sh`.

---

## Process

`knowledge/70-validation/VALIDATION_LOOP.md` governs every change: S0 charter → S1 locate →
S2 **pre-register a falsifiable prediction** → S3 invariant self-check → S4 execute →
S5 evidence → S6 refute → S7 gate → S8 record. Maximum three laps, then escalate.

**Keep S6 independent under Codex.** When collaboration tools are available, launch a
fresh-context `godmax-physics-referee` worker and give it only the claim, evidence ledger,
changed paths and routed invariants — not the author's reasoning. The referee is read-only
and never fixes the change. If dispatch is unavailable, use a separate Codex task/session.
Same-session self-review is allowed only for mechanical changes and must be labelled
non-independent in the ledger and report.

## The five rules that override convenience

1. **No number without the command that produced it.** Stored notebook outputs are never
   evidence — this repository holds megabytes of output from unknown code versions.
   Re-execute, or mark the claim UNVERIFIED. (`INV-PROC-EVIDENCE-01`)
2. **Never loosen a tolerance, eigenvalue cut, prior width, or ell range to make a check
   pass.** That converts a detected error into an undetected one. It is a physics change:
   own document, invariant review, explicit user sign-off. (`INV-PROC-NOTOLERANCE-01`)
3. **Report the absolute result, not the improvement.** Goodness of fit is judged against
   `retained rank − n_varied`. Stage-31 `fast1024`: `459 − 31 = 428 ± ~29`. The v1 best fit at
   whitened chi2 = 7346 is **not a good fit** — it is an operational point for map-pasting.
   Quoting the 224× improvement without the absolute number is a blocker violation.
   (`INV-CHI2-HONEST-01`)
4. **Show what did *not* change.** A fix reported without a null control has demonstrated
   nothing. This is the most frequently skipped and most valuable line of evidence.
5. **Escalate rather than lower the bar.** If a result will not come right, say so with the
   evidence and the remaining hypotheses. A confident wrong answer costs far more than an
   honest unresolved one.

## Conventions that fail silently

These have each produced a wrong number with no error message. Full statements:
`python tools/kb/kb.py invariants`.

| Rule | Getting it wrong looks like |
|---|---|
| NaMaster covariance is **band-major**: `cov.reshape(n_band, n_comp_a, n_band, n_comp_b)[:, a, :, b]` | Nothing raises. Matrix stays positive-definite. Covariance attributed to the wrong probe pair. |
| Covariance must be `gaussian_covariance(..., coupled=False)` | Leading dimension `n_ell` instead of `n_band`; wrong whitening rank. |
| `shear_e_to_kappa_sign = -1` on DES spin-2 fields | Pristine shear EE alongside four inverted cross families. |
| kSZ vector is **raw** `C_ell^{pi,T}`; theory maps via `-T_CMB_uK * A_v_bin * C_ell^{g,tau}`; plots show `-D_ell` | Fit prefers negative gas amplitude while the plot looks right. |
| DESI lens kernel uses **calibrated true-z** n(z), never `Z_PHOT_MEDIAN` | All four galaxy families biased the same way; HOD drifts to compensate. |
| Photometric pz bins **overlap** in true z — per-pz HOD needs separate theory blocks | Adjacent-bin HOD parameters unphysically anti-correlated. |
| Theory compared through **saved bandpower windows**, not at `ell_eff` | Smooth ell-dependent residual tilt no parameter can absorb. |
| ACT y and T get the 1.6 arcmin beam **once** | Monotonic high-ell deficit confined to ACT families. |
| `jax_enable_x64` **before any array is created** | Whitening rank drops below 459; chi2 varies run to run. |
| Never concretise a traced value in a constructor | Exactly zero gradient; the parameter never moves; reads as "unconstrained by data". |
| Guards on the **inputs** of a division/log — `jnp.where` evaluates all arms | Divergences with healthy acceptance; posterior truncated inside the prior. |
| Units and h conventions explicit at every boundary | Amplitudes off by ~0.67 / 0.45 / 1.49 with acceptable shape. |

## Roles (Codex skills)

Ten roles split by **failure mode**, not directory — `kb which` gives the answer
mechanically. Each is installed as a Codex skill named `godmax-<role>`. Dispatch bounded
role-specific workers when collaboration tools are available; otherwise adopt the skills
sequentially.

| Symptom | Skill |
|---|---|
| wrong estimator, covariance, mask, sign, or noise policy | `godmax-measurement-namaster` |
| physically wrong model; unphysical fitted parameter | `godmax-halo-model-physicist` |
| zero/NaN gradient, precision, tracing, speed | `godmax-jax-numerics` |
| wrong statistical conclusion; convergence; chi2 | `godmax-inference-statistician` |
| two xDESI stages disagree; measurement vs theory | `godmax-xdesi-lead` |
| pasted maps vs analytic theory | `godmax-abacus-paste-validator` |
| broken API contract across the `src/` chain | `godmax-core` |
| knowledge stale, code unowned, gate blocking | `godmax-kb-curator` |
| **is this actually right?** | `godmax-physics-referee` (refutes; never fixes) |
| needs numbers, reproducibly | `godmax-repro-runner` |

Prompts: `/godmax-kb-status` · `/godmax-kb-sync` · `/godmax-validate` ·
`/godmax-xdesi-status` · `/godmax-invariant-check` · `/godmax-kb-new`

If those skills or prompts are missing, they were never installed on this machine:

```bash
python tools/kb/sync_codex.py           # install/update
python tools/kb/sync_codex.py --check   # detect drift from .claude/ sources
```

The canonical role text lives in the tracked `.claude/agents/` and `.claude/commands/`;
the Codex skills and prompts are **generated** from it. Edit the tracked source, then re-sync
— never hand-edit `~/.codex/skills/godmax-*`, because the next sync overwrites it.

### Multi-agent dispatch under Codex

The root agent remains the coordinator and owns integration. Before dispatch it runs
`kb status`, `kb stale`, and `kb which <exact paths>`, then gives each worker:

- an observable outcome, exact read/write scope, routed documents and invariants;
- the S2 pre-registered prediction and falsifier;
- required evidence, including the null control; and
- explicit prohibited actions such as cluster submission, threshold changes, commits or
  pushes unless the user authorised them.

All workers share one worktree. Parallelise read-only investigations or disjoint write
scopes only; exactly one agent may write a given file. If two failure modes touch the same
file, serialise them or give integration to `godmax-xdesi-lead`. Keep child dispatch under
the root coordinator unless nested dispatch is explicitly useful and capacity is available.

At S5, `godmax-repro-runner` executes and records but does not interpret or fix. At S6,
`godmax-physics-referee` receives fresh context and only the claim plus evidence, so the
refutation remains independent. The root agent disposes findings, runs the S7 gate and
coordinates S8; each document is verified by its declared owner, and the root records the
integrated journal entry.

## Repository map

```text
src/               core library: base_class -> Profiles -> get_Pkz -> get_Cl -> {get_xi, get_cov}
                   dependency injection, all four params dicts threaded through every layer
src/arxiv/         24 SUPERSEDED modules — history only, never import, never cite
src/mcfitjax/      JAX port of mcfit (FFTLog); precision-critical
param_files/       YAML configs, deep-merged: params_default.yaml + project override
notebooks/xDESI/   the active analysis — see notebooks/xDESI/AGENTS.md
run_scripts/       samplers per project: pge/, dtai/, delta/
tests/             ONE file (812 lines), covers the xDESI measurement. src/ is untested.
knowledge/         the knowledge tree — read before acting, update after
tools/kb/kb.py     staleness, routing, invariant checks, the push gate
```

Config paths are cluster-absolute (`/mnt/ceph/users/spandey/...`,
`/mnt/home/spandey/miniconda3/envs/ili-sbi/`). `data/`, `outputs/`, `results/`, `logs/` are
gitignored and never travel between machines — a cluster result reaches the laptop only
through the tracked knowledge tree and journal.

## Practical notes

- **Never submit `sbatch` / `srun` / `salloc` without asking.** State node count, wall time,
  and the evidence the job will produce. Over ~1 node-hour is an escalation.
- **Prefer the cheapest sufficient evidence:** `fast1024` before `midres2048`; `cap600`
  before `cap2400` before fullsky.
- **`tests/test_xdesi_multiprobe_namaster.py` builds its own synthetic HDF5 inputs**, so it
  runs without cluster data. Use it first, and extend it with every fix.
- **Adding a params key?** Add it to `params_default.yaml` too, or every other config breaks
  with a `KeyError` at a random depth.
- **Notebooks:** read source cells for intent; re-execute for numbers.
- `pytest` and `pymaster` may be absent locally — `python tools/kb/kb.py doctor` reports how
  many invariants are actually machine-checkable in the current environment. A green gate on
  a machine without the scientific stack is a weaker statement than one on the cluster.

## Known open threads

1. **`desi_g_auto` chi2 = 6411** of 7346 total, for 40 data points — the dominant misfit.
   Blocks any physical interpretation of the Stage-31 fit. Eliminate in order: shot-noise
   subtraction → lens kernel → scale cuts / 1h–2h transition → HOD flexibility.
2. **v2 chains not diagnosed.** `max_tree_depth: 4` for 31 correlated parameters is low; the
   saturation fraction, r_hat and ESS are unrecorded, so no v2 posterior is quotable.
3. **`midres2048` DESI mask is provisional** — one DR9 random realization.
4. **kSZ at `lmax = 2048`** covers only the low end of the ~1000–7000 reference range.
5. **`src/` has no test coverage.** A construction smoke test and a gradient-flow test are
   the cheapest durable improvements available.

Most seed knowledge documents are `status: draft`, `confidence: medium` **by design**: they
were extracted from prose in `notebooks/xDESI/survey_measure/README.md`,
`BACKLIGHT_PASTE_HANDOFF_SUMMARY.md` and `src/context/codebase_summary.md`, not from
line-level reading. Treat them as good hypotheses, verify at S1, and promote them with an
evidence ledger.
