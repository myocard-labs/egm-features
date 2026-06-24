# myocard-egm-features

> Per-trace morphology + spectral + complexity feature extractors for intracardiac bipolar EGM signals.

Part of the [myocard-labs](https://github.com/myocard-labs) cardiac signal-processing toolkit.

> [!NOTE]
> Pre-v0.1.0 — this repo is being built up in [Refactor Step 4](https://github.com/myocard-labs/intracardiac-platform/blob/main/project/refactor_checklist.md) of the polyrepo refactor. The substantive README below replaces this stub once Block 5 of the egm-features build lands.

---

## What it is (one-liner)

A NumPy/SciPy/antropy library that takes a 1D bipolar-EGM trace + a sample rate and returns scalar features: peak-to-peak amplitude, zero-crossings, activation position, secondary-peak count, spectral centroid / entropy / dominant frequency, sample entropy, Shannon entropy, Lempel-Ziv complexity, Higuchi fractal dimension. Plus a `bundle.extract_all` helper that runs all 11 features over an `(N, T)` batch and returns a pandas DataFrame.

See `docs/theory.md` for the math behind each feature once it lands (Block 2 of the build).

---

## License

MIT — see [LICENSE](LICENSE). Third-party software-license acknowledgements are in [NOTICE](NOTICE).
