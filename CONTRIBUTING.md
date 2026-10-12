# Contributing

Thanks for your interest in improving jumpking-rl! Bug reports, fidelity
findings, new baselines and documentation fixes are all welcome.

## Development setup

```bash
git clone https://github.com/TranBaDat2607/jumpking-rl
cd jumpking-rl
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
pytest
```

The first run JIT-compiles the physics kernel with numba, which takes a few
seconds. Later runs use the on-disk cache.

If Jump King is installed through Steam, `tests/test_original.py` also checks
the clone against the real game. Without the game, those tests are skipped.

## Ground rules

- **Fidelity first.** Any change to `jumpking/physics.py` or
  `jumpking/constants.py` must keep the full test suite passing. If you own the
  game, run `python scripts/validate_original.py --download` too; it must
  still report `42 / 42 sections reproduced; goal reached: True`.
- **Cite sources.** When you change a mechanic, add or update the source in
  `docs/RESEARCH.md`.
- **No game assets.** Never commit files from the game, or images, maps or
  data derived from it. The original map and executable are only ever read at
  runtime from the user's own install.
- **Determinism.** All simulation state lives in the state vector.
  `get_state()` / `set_state()` must restore a run bit-exactly.
- **Style.** Match the surrounding code. Keep the kernel numba-compatible:
  plain floats, ints and NumPy arrays only.

## Pull requests

1. Open an issue first for large changes.
2. Add tests for new behaviour.
3. Update the README or `docs/RL_GUIDE.md` when you change the public API.
4. Describe how you verified the change in the PR description.

## Reporting fidelity differences

If the clone behaves differently from the real game, please include:

- the screen number;
- a `.tas` recording (press F2 in `jumpking play`, or use `evaluate.py --save-tas`);
- a description or video of what the real game does.
