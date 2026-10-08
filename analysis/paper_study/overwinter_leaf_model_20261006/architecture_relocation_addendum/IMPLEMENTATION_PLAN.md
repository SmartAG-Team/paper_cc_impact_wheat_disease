# Runtime and calibration boundary relocation

> **For agentic workers:** Execute inline under superpowers:executing-plans; root owns whole-task review and separately owned model/wheat_stb changes.

**Goal:** Keep model packages limited to parameterized simulation, with fitting, observations, evaluation, fitted configurations and experimental protocols outside model.

**Architecture:** Calibration packages import runtime kernels. Runtime never imports calibration or analysis. Existing fitted numbers and simulation functions retain their numerical definitions; active callers use the relocated names, and immutable archives retain original identities.

**Tech Stack:** Python, NumPy, Numba, SciPy/sklearn only in calibration.

**Spec:** Root's authorized relocation inventory and user instruction that calibration belongs outside model.

## Global Constraints

- No source snapshot, archived package, frozen receipt or frozen fit mutation.
- No edits to new model/wheat_stb, overwinter physics or leaf-phenology physics.
- No calibration compatibility shim retained inside model.
- Workspace has no Git repository; explicit file ownership and source addendum snapshots replace Git isolation and commits.

## Review Focus

- Mixed imports containing runtime and scoring names must split without alias loss.
- Running climate worker cannot freeze half-relocated source identities.
- Default wrapper paths must resolve relocated JSON without changing file bytes.
- Archived source-path/hash identities inside active verification scripts remain valid archival references.
- Runtime function bodies and forcing outputs must be numerically unchanged.

## Tasks

- [x] Record original source, configuration and active caller identities; add and run failing boundary tests; characterize unchanged runtime outputs.
- [x] Create calibration namespaces and extract whole/mixed modules, configurations and experimental docs; remove old calibration modules.
- [x] Rewrite only active imports and functional paths; split mixed runtime/evaluation imports; verify all relocated packages import before notifying climate.
- [x] Run focused regression and full-suite verification, AST dependency audit, numerical comparison and archived-source identity checks; write relocation addendum.
