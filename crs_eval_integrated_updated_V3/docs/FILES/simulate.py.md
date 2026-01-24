# File: simulate.py

## Purpose
Run a single simulated USER↔SYSTEM conversation and persist a log file.

## Inputs
- `N_TURNS` from `config.py`.
- `UserSimulator` and `CRSSystem`.

## Outputs
- Log JSON in `logs/<model>_session_<id>.json`.

## Key functions
- `run_simulation(session_id, model_name, seed)`

## Do-not-break invariants
- Preserve log schema and alternating USER/SYSTEM turns.
- Include `user_meta` on USER turns and `constraints` on SYSTEM turns.
