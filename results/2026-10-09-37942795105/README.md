# Run 2026-10-09-37942795105

Run 1: the free entrants (no-tool, every-hunk, Rigour 6.10.0's deterministic
checks) on `corpus-2026-10-09-37884111000`, $0. The page is
[summary.md](summary.md); the calibration sample and its verdicts are
`calibration.yaml`, with the two AI judges' inputs in `calibration-claude.yaml`
and `calibration-model.yaml`.

The score summaries the page is generated from are raw run output, so they
stay in the run's release, not here:
[run-2026-10-09-37942795105](https://github.com/rigour-labs/driftbench/releases/tag/run-2026-10-09-37942795105).

| File | SHA-256 |
|---|---|
| `runs-2026-10-09-37942795105.tar.gz` (release asset) | `2256a5f44639e7ece53034c688ec0c9fc14fce431c92bd5c5e46441072fb6669` |
| `2026-10-09-37942795105/scores/immich-app__immich.json` (in the tarball) | `3c9858ea992b6cf03d4c70edc097b1541aeb51c9e135b737a25bed3d13563ec7` |
| `2026-10-09-37942795105/scores/tailscale__tailscale.json` (in the tarball) | `69ce7ef3d5d1282a01e93902ca8e2f061039bfd17c566b65648a4582a36a25e0` |
| `2026-10-09-37942795105/scores/zulip__zulip.json` (in the tarball) | `bac53410a306b42e270924234e6ace60263a648454e9d930548c9405f72b8e0b` |

To regenerate the page, put the score summaries next to it and run the report
against the unpacked run (the points come from the corpus release):

```bash
gh release download run-2026-10-09-37942795105 -R rigour-labs/driftbench -p 'runs-*.tar.gz' -D /tmp/run1
tar -xzf /tmp/run1/runs-2026-10-09-37942795105.tar.gz -C /tmp/run1
cp /tmp/run1/2026-10-09-37942795105/scores/*.json results/2026-10-09-37942795105/
python -m bench report --run /tmp/run1/2026-10-09-37942795105 --results results/2026-10-09-37942795105
```
