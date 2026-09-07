# Backend account erasure

Set `SAM3DBODY_ACCOUNT_DELETION_TOKEN` to a dedicated secret. The backend must use
the same secret in `Authorization: Bearer ...` when posting
`{"userId":"synthetic-account","acceptedAt":1800000000000}` to
`/internal/account-erasure` over HTTPS. Missing or invalid credentials return 401.
Do not reuse callback credentials or expose the secret in client apps.

The endpoint persists a hashed account tombstone before checking in-flight work.
Inference and job-file writes hold shared filesystem leases; erasure requires an
exclusive lease. A live inference thread is allowed to finish rather than being
assumed cancelled. New/recovered jobs and callback writes cannot recreate files
after deletion. This works for processes sharing one artifact root; deployments
with several roots/nodes must erase every root before acknowledging global
completion to the backend.

The response is 200 with `{"userId":"...","status":"complete"}` only after
owned job JSON and its exclusive `technique-analysis/<task-id>/` output directories
are erased. Running work, custom/shared output paths, malformed records and file
errors remain 202 `pending` for retry/operator repair. R2 uploads are erased by the
backend after this acknowledgement, including a settling pass for older signed
upload URLs. The job and tombstone directories are not public artifacts.

New job records retain a small `erasureStorage` namespace descriptor even after
terminal jobs discard their original request URLs. Older terminal jobs may have
already discarded their storage descriptor; inventory/repair their ownership
before production acceptance. The endpoint deliberately keeps those pending
rather than guessing where they wrote files. Logs, backups, custom reference
generation and external-provider retention require their own retention policy.

Deterministic Linux checks (inside `sam_3d_body` conda environment):

```sh
./scripts/run_linux_pytest.sh tests/test_account_erasure.py tests/test_api_main.py
```

Run the normal module baseline and the parent repository's shared-contract smoke
before landing on `custom/main`. No model process restart or GPU inference is
needed for these erasure tests.
