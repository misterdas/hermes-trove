# Diagnostics

Use read-only product tools before changing configuration or running an apply path.

## Fast path

1. `hermes plugins`: confirm `hermes-trove` is enabled and the selected context engine is `trove`.
2. Send one normal message if the session has not been bound since restart.
3. `trove_status`: inspect runtime identity, database path, context pressure, summary/store counts, filters, and lifecycle state.
4. `trove_inspect`: inspect current-session lineage, frontiers, fresh tail, externalized-ref readability, and skip/no-op reasons without retrieving content.
5. `trove_doctor`: run database, FTS, lifecycle, configuration, and context-pressure diagnostics.

If optional slash commands are enabled, `/trove status` and `/trove doctor` expose the corresponding operator views. To enable: set `TROVE_ENABLE_SLASH_COMMAND=1` in the environment. Without it, the `/trove` slash commands are silently not registered.

### Duplicate message rows

`/trove doctor duplicate` counts re-ingested rows: identical message content
persisted more than once under different `store_id` values. It is read-only
and reports `duplicate_clusters`, `redundant_rows` (the removable count), and
`largest_cluster_extra`.

`/trove doctor duplicate apply` collapses them, keeping the **earliest**
`store_id` of each cluster. It re-points `trove_chunk_meta` rows onto the
survivor *before* deleting, so recall keeps working, and takes a backup first.
Gated by `TROVE_DOCTOR_CLEAN_APPLY_ENABLED=true` (same gate as
`/trove doctor clean apply`).

These rows come from an ingest failure that left `_ingest_cursor` at its
pre-batch position, so the next turn re-sent the whole history. The unique
identity index cannot catch it: `observed_at` is NULL for most rows (the host
supplies no message timestamp) and SQLite treats NULLs as distinct, so the
re-copies never collide. Counting therefore folds NULL `observed_at` together
with `COALESCE(observed_at, -1)`.

Same work offline, without a bound session:
`python3 scripts/dedup_messages.py --db ~/.hermes/trove.db [--apply] [--vacuum]`.

## Safe mutation order

For cleanup, repair, source normalization, or rotate:

1. run the read-only preview;
2. inspect exact candidates and paths;
3. create/confirm a backup;
4. obtain user authorization for the specific apply operation;
5. run one bounded apply and verify integrity afterward.

Cleanup apply is separately feature-gated. Never infer permission to enable it from a diagnosis request.

## Common states

- Unbound status after restart: send a normal message, then check again.
- Database exists but stays empty: verify plugin enablement, `context.engine`, profile, database path, and ignore/stateless patterns.
- Weak exact recall: verify source rows exist, query construction/scope is correct, summary health is sound, and embedding coverage/provenance matches the requested mode.
- Conflicting summary and raw evidence: prefer the newer exact raw evidence and inspect lineage.
- Path B/context-engine schema log: expected on hosts where plugin-registry handlers do not receive active messages; context-engine schemas and dispatch remain the healthy route.
