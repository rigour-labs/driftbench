# Run 2026-10-10-38028530710

Run 2: Rigour's reviewer (6.12.1, `--blind`) and Claude Code's /code-review
(2.1.285), both on anthropic/claude-sonnet-5.5 through OpenRouter, on a
seeded subsample of immich and tailscale (`subsamples/run-2.yaml`: whole
pull requests, about 32 heads each; zulip not run in this round for budget).
Corpus `corpus-2026-10-09-37884111000`. Spend: $27.59 estimated, $22.23
billed by OpenRouter, under a $50 cap. Calibration and the issue judge
(openai/gpt-6.1-sol) cost $0.53 more, under a $2 cap.

The page is [summary.md](summary.md). The calibration sample and its
verdicts are `calibration.yaml` (merged), `calibration-claude.yaml` and
`calibration-model.yaml`.

## Reading it

- **Location (same spot within 3 lines):** Rigour's reviewer 19% (immich)
  and 14% (tailscale); Claude Code 0% on both. Claude Code's /code-review
  cites no file:line in headless output on 68 of 72 heads, so it can't score
  on location, whatever it says.
- **By meaning (AI judge, non-Claude, blind to entrant):** Claude Code
  raised the same issue as the human point on 31% (immich) and 20%
  (tailscale); Rigour's reviewer on 13% and 14%. Across both repos, Claude
  Code alone raised 17 points' issues, Rigour alone 1, both 9. The judge
  repeated its verdict on 19 of 20 re-asks.
- **Rigour's location matches, read:** of 16 spot-checked, 6 were the same
  issue, 2 partly, 8 not.
- **Status:** the spot-check sample is complete. Two acted-on entries got no
  valid verdict from the model (twice), so the maintainer judged them; a
  human verdict wins. Acted-on decisions agree with their checks on 24 of 26
  (1 disputed and left out).

## Files kept in the run's release

[run-2026-10-10-38028530710](https://github.com/rigour-labs/driftbench/releases/tag/run-2026-10-10-38028530710)

| File | SHA-256 |
|---|---|
| `issue-judgments.yaml` (release asset): every issue verdict and its reason | `92e722df2d293b8bfc8604e62a316930c9daa45ef15519818b25e6ab4d270868` |
| `runs-2026-10-10-38028530710.tar.gz` (release asset): every record, with paid entrants' whole answers | `084c9e6f755e680aaf05ad83250de39edd757d339c451ffb2bf9065eb1ae192a` |
| `2026-10-10-38028530710/scores/immich-app__immich.json` (in the tarball) | `aca94c0a2aacea6d5f23eb3a2424a4c3ca8a5b3a614e1063570e67cd978f1074` |
| `2026-10-10-38028530710/scores/tailscale__tailscale.json` (in the tarball) | `8d6f44d6de3e86343ce8b79cf58d9a1fc02c5111ef14bcd449cd5e2095883b27` |
| `2026-10-10-38028530710/scores/zulip__zulip.json` (in the tarball) | `b40e9af8938187924d33e6172942d77f55a604bf53f5f88732bd6094c378b953` |

To regenerate the page, put the score summaries and `issue-judgments.yaml`
next to it and run `python -m bench report --run <unpacked run> --results
results/2026-10-10-38028530710`.
