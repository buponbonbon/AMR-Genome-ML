# FILE19 v1.0.3 — Cross-source temporal AST audit

- Query timestamp (from BigQuery result): 2026-10-22 17:31:46.721501 UTC
- Verdict: **CURRENT_BQ_BIOSAMPLE_CONSISTENT__HISTORICAL_RAW_DISCORDANT__CAUSE_UNRESOLVED**

Historical FILE11 raw and current NCBI BigQuery/BioSample are compared as **source records**, not independent validation of laboratory AST.
No outcome labels, predictions, training files or primary frozen results were modified.

| Drug | Old vs XML (agree/disagree/missing) | Current vs XML (agree/disagree/missing) | Old vs current (agree/disagree/missing) |
|---|---:|---:|---:|
| meropenem | 0/175/0 | 175/0/0 | 0/175/0 |
| imipenem | 27/148/0 | 175/0/0 | 27/148/0 |
| ertapenem | 1/174/0 | 175/0/0 | 1/174/0 |

## Causal interpretation boundary

A matching current BigQuery/BioSample snapshot and a discordant older CSV establish a *between-snapshot/source difference*, not when, why, or by whom labels changed.
The BigQuery table is a mutable public resource; obtaining an historical query job ID or NCBI explanation would increase confidence in the cause.
Do not overwrite E2_FROZEN or substitute these phenotypes into the primary study.
