# Minimal Python Parallel Systems Lab

Performance Evaluation and Optimization in Parallel Computing — scaling laws, shared-memory
cache contention, empirical roofline and dynamic load balancing. Pure standard-library Python 3.

## Setup
1. In `lab_common.py`, set `STUDENT_ID` to your own ID.
2. Plug in the laptop and close other applications for stable measurements.
3. Disconnect Wi-Fi / Ethernet, as required by the lab rules.

## Run (macOS Terminal)
```bash
caffeinate -i python3 run_all.py   # all phases + report, about 8–15 min
```
To restart from a given phase: `python3 run_all.py 2`.

## Files
| File | Purpose |
|---|---|
| `harness.py` | Minimal timing and verification harness (Listing 1) |
| `lab_common.py` | Student ID, shared-memory helpers, result storage |
| `task0_cal.py` | Phase 0 — hardware topology and timer calibration (10⁶ calls) |
| `task1_falsesharing.py` | Phase 1 — false-sharing stride sweep and Karp-Flatt, M = 1.6×10⁷ |
| `task2_scaling.py` | Phase 2 — strong (N = 1.2×10⁶) vs weak (N0 = 2×10⁵) scaling |
| `task3_roofline_gemm.py` | Phase 3 — 2×64 MB bandwidth, FMA ceiling, GEMM N = 256, BS ∈ {8…128} |
| `task4_scheduling.py` | Phase 4 — N = 8000, K(i) = 1 + ⌊(i/40)²⌋, C ∈ {1, 8, 64, 512} |
| `run_all.py` | Runs all phases in order |
| `make_report.py` | Builds `lab_report_<ID>.md` |

## Output
* `lab_report_<ID>.md` — hardware table, five VERIFY_HASH lines, all tables, analysis,
  raw terminal logs and the complete source code.
* `results/*.json` — raw measurements per phase.

Do not set `LAB_QUICK=1` for real runs; it shrinks problem sizes for smoke testing only.
