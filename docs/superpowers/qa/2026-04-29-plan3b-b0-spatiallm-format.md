# Plan 3b B0 — SpatialLM door/window format verification

**Date:** 2026-04-29
**Scan:** real bedroom scan via the iOS app + RunPod SpatialLM tunnel
**Reference scan:** SpatialLM's own `manycore-research/SpatialLM-Testset` `scene0000_00.ply`

## Findings

### Raw SpatialLM language string

SpatialLM 1.1 (Qwen 0.5B) emits **structured language strings** (one entry per
line) into a single text blob. From the real-scan probe at
`/tmp/decroom_b0_raw_slm.txt`:

```
wall_0=Wall(3.7344, 0.8690, 1.2694, 7.5594, 0.8690, 1.2694, 2.36, 0.0)
wall_1=Wall(3.7344, 0.8690, 1.2694, 3.7344, 1.2690, 1.2694, 2.36, 0.0)
…
wall_7=Wall(1.0594, 7.4940, 1.2694, 7.5594, 7.4940, 1.2694, 2.36, 0.0)
door_0=Door(wall_5, 2.7344, 3.3940, 2.6944, 0.94, 2.82)
window_0=Window(wall_0, 5.5344, 0.8690, 4.2694, 2.36, 0.94)
bbox_0=Bbox(cupboard, 5.5094, 3.0940, 2.0944, -1.5708, 4.6719, 4.6406, 1.625)
```

### Schema confirmation

| Entry | Format |
|---|---|
| `wall_N=Wall(...)` | `(x0, y0, z0, x1, y1, z1, height, _unused_)` — segment endpoints in SpatialLM coords |
| `door_N=Door(...)` | `(wall_id, x, y, z, width, height)` — **wall_id references the attached wall** |
| `window_N=Window(...)` | `(wall_id, x, y, z, width, height)` — same wall_id reference |
| `bbox_N=Bbox(...)` | `(category, x, y, z, rotation_y, dim_x, dim_y, dim_z)` |

### Key insight: doors/windows reference walls by ID

This is **simpler than the design spec assumed**. The spec assumed
`extract_doors_windows` would use proximity matching ("find nearest wall
within 0.5 m"). Not needed — SpatialLM gives the wall id directly.

## Impact on Task 14

The original spec for Task 14 was:

> Append `extract_doors_windows(slm_objects, walls)` to `consensus.py`.
> Matches SLM door/window entries to nearest wall within 0.5 m, sets
> `has_door` / `has_window`.

That approach is now obsolete. **Better approach: extend the existing
`parse_spatiallm_output` parser** in `pipeline/spatiallm_layout.py` to:

1. Match `door_N=Door(wall_X,…)` and `window_N=Window(wall_X,…)` lines via regex.
2. Collect the referenced wall IDs into sets.
3. Set `has_door` / `has_window` on each wall by checking membership.

No new file in `consensus.py` required. The wall-id reference makes this an
o(walls + openings) operation with no spatial math.

## Side findings (out of scope for Task 14, noted for follow-up)

- The existing `parse_spatiallm_output` returns `objects: []` for the test
  bedroom scan even though the raw text contains
  `bbox_0=Bbox(cupboard,...)`. That's because the SpatialLM-1.1 categories
  emitted by the model (cupboard, dressing_table, sofa, painting, etc.)
  aren't covered by the project's `SPATIALLM_TO_PRD` category mapping.
  Update the mapping in a follow-up task if we want furniture detection in
  the response — currently those bboxes are silently dropped.
- The first phone scans of a real room produced
  `Floating point exception (core dumped)` until two install fixes were
  applied: (a) replace `spconv-cu120` with `spconv-cu124` to match
  PyTorch 2.4.1+cu124, (b) supply `code_template.txt` next to `server.py`.
  Both encoded in `decroom_no_lidar/backend/spatiallm_server/setup.sh`.
- The phone caps the cloud at 500K points. SpatialLM's voxel encoder
  crashes around 250K+. Backend now subsamples to 200K before sending
  (overrideable via `DECROOM_SPATIALLM_MAX_POINTS`).
- SpatialLM's `NormalizeColor` z-scores per-batch and crashes on zero
  std (uniform grey). Backend now jitters synthetic colors with std≈4
  before sending (no effect on the model's understanding).

## Task 14 status

Implemented in this session — the parser change in
`pipeline/spatiallm_layout.py:_DOOR_RE / _WINDOW_RE` plus the membership
check in `parse_spatiallm_output` makes Task 14 trivial. No new
`consensus.py` function needed. Plan 3b spec §7 should be amended to
reflect this.
