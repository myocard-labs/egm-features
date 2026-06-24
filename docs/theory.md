# Theory — myocard-egm-features

The math behind every feature this library ships, with derivations, worked
examples on synthetic signals where the answer is known, and the rationale
for each parameter choice baked into the implementation.

This doc lands **before** the feature implementations per the
"theory-first for unfamiliar domains" workflow rule
(`[[feedback-theory-first-for-unfamiliar-domains]]`). The audience is the
person who has to code-review the implementations and verify each
function is computing the right thing — without prior expertise in
entropy / Lempel-Ziv / fractal-dimension math.

For the *what* and *where* of each feature (CLI invocation, return shape,
column names in the bundle DataFrame), see `docs/usage.md` once it
lands in Block 5. For the design rationale behind the library's overall
shape (pure functions, per-trace `(T,)` inputs, no CLI), see
`project/architecture.md`.

## Table of contents

- [Preprocessing assumptions](#preprocessing-assumptions)
- [Notation](#notation)
- [1. Time-domain features](#1-time-domain-features)
  - [1.1 peak_to_peak](#11-peak_to_peak)
  - [1.2 zero_crossings](#12-zero_crossings)
  - [1.3 activation_position](#13-activation_position)
  - [1.4 sec_peak_count](#14-sec_peak_count)
- [2. Frequency-domain features](#2-frequency-domain-features)
  - [2.1 Power spectrum (the shared computation)](#21-power-spectrum-the-shared-computation)
  - [2.2 spectral_centroid](#22-spectral_centroid)
  - [2.3 spectral_entropy](#23-spectral_entropy)
  - [2.4 dominant_frequency](#24-dominant_frequency)
- [3. Complexity features](#3-complexity-features)
  - [3.1 sample_entropy](#31-sample_entropy)
  - [3.2 shannon_entropy](#32-shannon_entropy)
  - [3.3 lempel_ziv_complexity](#33-lempel_ziv_complexity)
  - [3.4 higuchi_fractal_dimension](#34-higuchi_fractal_dimension)
- [4. References](#4-references)

---

## Preprocessing assumptions

This library does not filter, calibrate, or resample. It assumes the
caller has already:

- **Bandpass-filtered** the trace to the clinical band of interest
  (`[30, 250] Hz` for atrial bipolar EGMs — see
  `myocard_egm_signal.filters.bandpass`).
- **Calibrated** amplitudes if absolute mV matters (R-wave anchoring for
  IAFDB; producer-specified gain for synthetic — see
  `myocard_egm_signal.calibration`).
- Optionally **resampled** to a target rate. The features library accepts
  any `fs_hz`; what matters is that all traces in a comparison study use
  the same `fs_hz`.

Because the input is post-bandpass, EGM traces are approximately
**zero-mean** by construction. Several features below rely on this
(zero_crossings counts crossings around zero rather than around the
per-trace mean; lempel_ziv_complexity's "zero" binarization mode also
uses it).

## Notation

- `x ∈ ℝᵀ` — a single per-trace input. `T` is the sample count.
- `T` — number of samples in the trace. Default 512 (matches the
  egm-classifier v1 input length at 1 kHz = 512 ms).
- `fs_hz` — sample rate in Hz. The features that touch the time axis
  (frequency-domain ones, activation-position-as-time variants) need it;
  the dimensionless complexity features don't.
- `t_i = i / fs_hz` — time at sample index `i`.
- `μ(x)`, `σ(x)`, `med(x)` — mean, std (population, ddof=0), median.
- `NDArray` — a `numpy.ndarray` of floating-point type; specifically
  shape `(T,)` for the per-trace functions. Batch processing over
  `(N, T)` is `bundle.extract_all`'s job, not the individual extractors.

---

## 1. Time-domain features

### 1.1 peak_to_peak

**Definition.**

```
peak_to_peak(x) = max(x) - min(x)
```

In mV (or whatever amplitude unit the caller passed in).

**Worked example.** A pure sinusoid `x[i] = A · sin(2π f · t_i)`:

```
max(x) = A,  min(x) = -A  →  peak_to_peak = 2A
```

So a 1 mV-amplitude sine at any frequency has `peak_to_peak = 2.0`. The
test fixture exercises this.

**Parameter choices.** None. The function takes `x` only.

**EGM interpretation.** Peak-to-peak amplitude is the most basic
substrate signature in the clinical literature — the threshold-based
fibrosis maps from Marchlinski (2000) and Sanders (2003) are
peak-to-peak voltage maps. Low peak-to-peak (`< 0.5 mV` sinus rhythm;
`< 0.2 mV` AF-adjusted per Kosiuk) flags low-voltage substrate. The
v1_baseline diagnostic confirmed this is *not* a sufficient feature on
its own (low-density fibrotic and healthy traces can have overlapping
peak-to-peak), which is why we ship 10 more features alongside.

**Reference:** Marchlinski 2000, Sanders 2003 (clinical voltage maps);
see [[white-paper-references]].

### 1.2 zero_crossings

**Definition.** Count of times the signal crosses zero between adjacent
samples. Each crossing counts once regardless of direction.

```
zero_crossings(x) = | { i : sign(x[i]) ≠ sign(x[i-1]),  1 ≤ i < T } |
```

The exact-zero case (`x[i] = 0`) is folded into `sign` via NumPy's
convention (`sign(0) = 0`), so a value of exactly zero is treated as a
"both directions" indicator and counted as a crossing on either side.
For float-valued bandpass-filtered signals this edge case effectively
never fires.

**Worked example.** A pure sinusoid at frequency `f` over duration
`T_seconds`:

```
zero_crossings ≈ 2 · f · T_seconds
```

For `f = 50 Hz`, `T_seconds = 0.512 s` (= 512 samples at 1 kHz): 
`2 · 50 · 0.512 = 51.2`, so ~51 crossings. The test fixture verifies
this within ±1.

**Parameter choices.** None — the count is around zero (not the per-trace
mean), because EGM traces are approximately zero-mean post-bandpass.

**EGM interpretation.** Zero crossings is a coarse frequency-content
proxy. A fragmented fibrotic signal with high-frequency oscillations
has more zero crossings than a smooth biphasic activation. Cheap to
compute; partially redundant with `dominant_frequency` but captures
overall waveform "wobble" rather than spectral peak.

### 1.3 activation_position

**Definition.** Returns the fractional position (in `[0, 1]`) of the
**maximum slope** of the signal:

```
activation_position(x) = argmax(|dx/dt|) / (T - 1)
```

where `dx/dt` is approximated by the discrete first difference
`x[i] - x[i-1]` for `i = 1, …, T-1`.

**Why max-slope and not peak-amplitude.** The clinical activation-time
convention (Marchlinski / Wittkampf school) takes the steepest *slope*
as the moment of activation under the electrode — the wavefront passes
under the bipolar pair, the field flips polarity rapidly, and that
steepest dV/dt point is when activation "happens." Peak amplitude
(argmax of `|x|`) is also commonly used and slightly easier to compute,
but it tracks the post-activation extremum rather than the activation
event itself.

**Worked example.** A simple biphasic activation:

```
x[i] = -A · sin(2π · (i - k) / w)    for k - w/2 < i ≤ k + w/2
     = 0                              elsewhere
```

This is one cycle of a sine, centered at sample `k`, lasting `w`
samples. The steepest slope occurs at the zero-crossing in the middle
of the pulse (sample `i = k`), so `activation_position = k / (T - 1)`.

If the trace is `T = 512` samples and the activation is centered at
sample `k = 256`, we get `activation_position ≈ 0.5`. The test fixture
exercises this with a few `k` values.

**Parameter choices.** None for the default. A `method="abs_peak"` flag
is reserved for the alternate convention (argmax of `|x|`), but the
default is the dV/dt method per the wave-2 design decision.

**EGM interpretation.** For producer-side `activation-peak-anchored`
traces (synthetic-egm-pipeline's default for the v1 single-activation
sims), the activation position is structurally close to 0.5 because
the producer crops a window centered on the detected peak. So this
feature is more informative when applied to *unanchored* traces (e.g.
multi-activation Phase 4 traces or IAFDB-style continuous recordings
with windows extracted at arbitrary offsets).

**Limitations of the simple discrete first-difference algorithm.** The
discrete first difference `x[i] − x[i-1]` is effectively a high-pass
filter — noise components at frequency `f` get amplified by roughly
`2π · f` relative to the signal. The pre-stage bandpass at 30–250 Hz
caps the absolute worst of this (anything above 250 Hz is gone), but
the algorithm is still sensitive to noise within the band. Specifically
the simple algorithm can mis-locate the activation when:

- **Poor electrode contact / low SNR.** The activation's dV/dt becomes
  comparable to noise dV/dt; argmax lands on a noise spike instead of
  the true activation edge.
- **High-amplitude transients** within the trace window (motion
  artifact, pacing-stimulator artifact, far-field ventricular spike).
  These typically have steeper dV/dt than the local atrial activation.
- **Fragmented activations.** When the activation has multiple fast
  deflections, argmax may pick a secondary deflection rather than the
  morphologically-first activation event.

**Alternative algorithms (clinical-mapping literature):**

| Method | What it adds | Cost |
|---|---|---|
| Wittkampf-smoothed dV/dt — low-pass at ~200 Hz before differencing | Suppresses in-band noise without erasing the activation edge | One extra filter call |
| Polynomial fitting — fit a degree-3 polynomial in a window, take its analytic derivative | Smoother slope estimate; less noise-sensitive | More compute; window-size knob |
| Hilbert-transform phase tracking — extract instantaneous phase, find argmax of `dφ/dt` | Robust to amplitude variation | Sign convention + edge artifacts |
| Template matching — convolve with a known activation template, find peak correlation | Best when activation morphology is known | Needs representative template; over-fits if morphology varies across the dataset |

**When to upgrade.** Phase 1.5 is the natural review point. If the
`activation_position` feature shows excessive variance when comparing
synthetic vs IAFDB distributions during the sim-realism work, the
simple discrete first difference is the most likely culprit and the
Wittkampf-smoothed variant is the cheapest upgrade.

**Reference:** Wittkampf et al., *Activation Mapping of Atrial
Fibrillation* (general use of max-dV/dt as activation marker in clinical
mapping systems).

### 1.4 sec_peak_count

**Definition.** Number of secondary peaks (i.e., local maxima of `|x|`)
above a fraction of the primary-peak amplitude.

Algorithm:

1. Compute the rectified signal `|x|`.
2. Find the primary peak amplitude `A_max = max(|x|)`.
3. Find all local maxima of `|x|` with prominence
   `≥ threshold_frac · A_max`.
4. Subtract 1 (to exclude the primary peak itself), with a floor at 0.

In code:

```python
abs_x = np.abs(x)
A_max = float(abs_x.max())
peaks, _ = scipy.signal.find_peaks(abs_x, prominence=threshold_frac * A_max)
sec_peak_count = max(0, len(peaks) - 1)
```

**Worked example 1 — single biphasic activation.** A single
sine-pulse activation (worked example from §1.3) has exactly one local
maximum of `|x|` at the activation peak. So `sec_peak_count = 0`.

**Worked example 2 — fragmented two-component activation.** A primary
activation of amplitude `A` followed by a secondary activation of
amplitude `A/2`:

```
|x|:    ___/\___/\___    (two bumps, second is half the height)
```

With `threshold_frac = 0.3`, the secondary peak's prominence
(`A/2 = 0.5 · A`) exceeds the threshold (`0.3 · A`), so it counts.
`sec_peak_count = 1`.

**Parameter choices.** `threshold_frac = 0.3` (default; 30% of primary
peak amplitude). The fragmentation-index family in the literature
(Nademanee 2004 CFAE definition, Kim 2014 fractionation analyses) uses
similar fractional thresholds, typically 30–50%. Lower values count
more peaks (noise-sensitive); higher values miss legitimate secondary
deflections. 0.3 is the pragmatic middle.

**EGM interpretation.** Fragmented atrial electrograms (the "CFAE"
phenotype Nademanee popularized) have multiple deflections per
activation event. `sec_peak_count` is a direct proxy for the fragmentation
phenotype — higher count means more fragmented. This is one of the
features most expected to differ between healthy and fibrotic synthetic
traces.

**Reference:** Nademanee 2004 (CFAE); Kim 2014 (fractionation indices).

---

## 2. Frequency-domain features

All three frequency features depend on a power spectral density (PSD)
estimate. We compute it once per trace via `scipy.signal.periodogram`
and reuse it.

### 2.1 Power spectrum (the shared computation)

**Definition.** For a discrete-time signal `x[i], i = 0..T-1` sampled at
`fs_hz`, the **periodogram** estimate of the PSD is:

```
P̂(f_k) = (1 / (fs_hz · T)) · |DFT(x)[k]|²
```

where `f_k = k · fs_hz / T` for `k = 0, 1, …, T/2` (Nyquist-limited;
the DFT is symmetric for real-valued `x` so we only keep the positive
half) and `DFT(x)[k]` is the discrete Fourier transform.

The units of `P̂` are `(signal unit)² / Hz`. We don't care about the
absolute scale for the three downstream features (centroid uses
normalized weighting; entropy uses normalized probabilities; dominant
frequency uses argmax which is scale-invariant) — so the scale factor
out front cancels in every consumer.

`scipy.signal.periodogram` returns two arrays: `f` (length `T/2 + 1`)
and `P̂` (same length). We pass both to the consumers below.

**Choice of method — why periodogram, not Welch.** Welch's method
averages periodograms over overlapping segments, which reduces variance
of the estimate at the cost of frequency resolution. For our typical
trace length `T = 512` at 1 kHz, the frequency resolution of a single
periodogram is `fs_hz / T = 1000 / 512 ≈ 2 Hz`, which is fine for
distinguishing the EGM band (30–250 Hz). Welch with `nperseg = 256`
would double the frequency bin width to ~4 Hz without giving us much
variance reduction (only 2× averaging on a single trace). Periodogram
is the simpler, cleaner choice at our trace length.

**No band restriction internally.** The PSD is computed over the full
spectrum from 0 to `fs_hz / 2`. The caller is assumed to have already
bandpassed the input to the clinical band of interest (per the
preprocessing assumptions section above); double-restricting in
egm-features would be redundant.

### 2.2 spectral_centroid

**Definition.** The power-weighted mean frequency:

```
spectral_centroid(x) = ( Σ_k f_k · P̂(f_k) ) / ( Σ_k P̂(f_k) )
```

Units: Hz.

**Worked example.** A pure sinusoid at frequency `f₀`. Its PSD is a
delta at `f₀` (in the continuous limit; in the discrete periodogram,
it's a sharp peak at the nearest bin):

```
spectral_centroid ≈ f₀
```

For a 50 Hz sine over a 512-sample window at 1 kHz, the bin closest to
50 Hz is `k = round(50 · 512 / 1000) = 26`, so `f_26 ≈ 50.78 Hz`. The
centroid is essentially that bin's frequency. Test verifies
`|spectral_centroid - 50| < 2` Hz (one-bin tolerance).

**Worked example 2 — two sinusoids at different frequencies *and*
amplitudes.** Take `x = 1.0 · sin(2π · 50 · t) + 2.0 · sin(2π · 150 · t)`
— a 50 Hz component at amplitude 1, plus a 150 Hz component at
amplitude 2. The PSD has two peaks. Key point: PSD weights by *power*
(amplitude²), not amplitude. So:

- Power at 50 Hz: `∝ 1² = 1`
- Power at 150 Hz: `∝ 2² = 4`

Centroid:

```
spectral_centroid = (50 · 1 + 150 · 4) / (1 + 4) = 650 / 5 = 130 Hz
```

So the centroid sits at **130 Hz**, much closer to the 150 Hz peak
than to 50 Hz, even though both frequencies are present. If we'd used
equal amplitudes (both at 1.0), the centroid would land at the
unweighted average `(50 + 150) / 2 = 100 Hz`. The amplitude-vs-power
distinction is the most common point of confusion with spectral
centroid; the multi-sinusoid example makes it explicit.

**Parameter choices.** None — PSD method is fixed (periodogram).

**EGM interpretation.** Spectral centroid is a "center of mass" of the
signal's power distribution. A higher centroid means more high-frequency
content (faster activation morphology); lower means more low-frequency
content (slower / smoother). For fibrotic substrate the literature
expects either direction depending on the fibrosis pattern (interstitial
fibrosis tends to fragment → higher centroid; dense scar tends to
slow conduction → lower).

### 2.3 spectral_entropy

**Definition.** The Shannon entropy of the **normalized PSD** treated
as a probability distribution over frequency bins, divided by
`log(N_bins)` for normalization to `[0, 1]`:

```
p_k = P̂(f_k) / Σ_j P̂(f_j)                                   # PSD as a pmf
spectral_entropy(x) = -( Σ_k p_k · log(p_k) ) / log(N_bins)   # normalized to [0,1]
```

By convention `0 · log(0) = 0`. Logarithm base is natural log (nats);
the `log(N_bins)` divisor is in the same base so the result is
dimensionless and ∈ `[0, 1]`.

We delegate to `antropy.spectral_entropy(x, sf=fs_hz, method="welch",
normalize=True)`. Note we override antropy's default `method="welch"`
to `method="fft"` (periodogram) for consistency with §2.1 — same PSD
for centroid + entropy + dominant.

**Worked examples.**

*White noise:* PSD is flat across all frequency bins, so all `p_k = 1/N_bins`
and the entropy is at its maximum `log(N_bins)`. Normalized:
`spectral_entropy(white_noise) ≈ 1.0`.

*Pure sinusoid:* PSD is a delta — one bin has `p_k ≈ 1`, all others
≈ 0. Entropy is ~0. Normalized: `spectral_entropy(sine) ≈ 0`.

*Two sinusoids at different powers* (same setup as the §2.2 example —
50 Hz amplitude 1, 150 Hz amplitude 2). After PSD normalization,
essentially all the probability lives in two bins with proportions
`p_50 = 1/5 = 0.2` and `p_150 = 4/5 = 0.8`. Other bins have
`p_k ≈ 0`. Entropy:

```
H = -(0.2 · ln 0.2 + 0.8 · ln 0.8) = -(0.2 · -1.609 + 0.8 · -0.223)
  = 0.322 + 0.179 = 0.500 nats
```

Normalizing by `log(N_bins)` (with `N_bins ≈ 257` for `T = 512` at
the periodogram resolution):

```
spectral_entropy = 0.500 / log(257) ≈ 0.090
```

That's higher than a single sinusoid (≈ 0) but much lower than white
noise (≈ 1.0) — exactly what we'd want from an "energy concentration"
metric. Two well-defined peaks is still "concentrated," but less so
than one peak. The relative weighting matters too: an *equal*-power
two-sinusoid signal (both at amplitude 1) would have
`p_50 = p_150 = 0.5`, giving `H = ln 2 ≈ 0.693` nats and a slightly
higher normalized entropy (~0.125). More-equal power → higher entropy.

**Parameter choices.** None exposed today. `normalize=True` is hardcoded
because the un-normalized value depends on `N_bins` which depends on
`T`, making cross-trace comparison meaningless without it.

**EGM interpretation.** Spectral entropy tells you how *concentrated*
the signal's power is in frequency space. Low entropy = the energy lives
in a narrow band (orderly activation with a dominant frequency); high
entropy = broadband (noisy, fragmented, or just noise-dominated). For
classifying fibrotic substrate, higher spectral entropy correlates with
fragmentation.

**Reference:** Shannon 1948 (the original entropy formula); Inouye 1991
(spectral entropy applied to EEG, the canonical reference for the
biomedical-signal use).

### 2.4 dominant_frequency

**Definition.** The frequency at which the PSD is maximal:

```
dominant_frequency(x) = argmax_k P̂(f_k) → f_{k*}
```

Units: Hz.

**Worked example.** A pure sinusoid at `f₀ = 50 Hz` over 512 samples
at 1 kHz: `dominant_frequency ≈ 50.78 Hz` (the nearest bin). Test
verifies `|dominant_frequency - 50| < 2` Hz.

**Worked example 2 — two sinusoids, different amplitudes.** Same setup
as §2.2's multi-sinusoid example (50 Hz amplitude 1, 150 Hz amplitude
2). The PSD has two peaks; the 150 Hz peak has 4× the power of the
50 Hz peak. `argmax` picks the higher-power bin:

```
dominant_frequency = 150 Hz (nearest bin: ≈ 150.39 Hz)
```

Two implications worth noting:

- **dominant_frequency tracks the highest-power peak, not the
  highest-frequency peak.** If we'd inverted the amplitudes (50 Hz
  amplitude 2, 150 Hz amplitude 1), `dominant_frequency` would have
  landed at 50 Hz instead of 150 Hz.
- **It's a hard maximum — no second-best information.** If two peaks
  have nearly-equal power, the feature picks one of them and gives
  no hint that the other exists. That's where spectral_centroid (which
  averages over all power) and spectral_entropy (which measures
  spread) become useful complements.

**Parameter choices.** None.

**Note on the DC bin.** `scipy.signal.periodogram` includes the `f = 0`
bin (DC component). For a bandpass-filtered EGM the DC bin is near
zero by construction, so it won't win the argmax. We don't explicitly
exclude it.

**EGM interpretation.** For a clean biphasic activation, the dominant
frequency is roughly the inverse of the activation pulse width — a
shorter, sharper activation has a higher dominant frequency. For a
fragmented or multi-component trace, the dominant frequency might land
on a noise peak or a secondary deflection.

---

## 3. Complexity features

The complexity family captures aspects of the signal that frequency
analysis misses: how predictable the time series is (sample_entropy),
how spread out its amplitude distribution is (shannon_entropy), how
"random" the binarized signal looks (lempel_ziv_complexity), and how
fractal-like its trajectory is (higuchi_fractal_dimension).

Three of the four wrap `antropy` functions because the math is
well-established and the antropy implementations are tested. Our job
is to choose parameters correctly, document them here, and write
synthetic-signal tests that catch parameter-passing bugs.

### 3.1 sample_entropy

**Definition (Richman & Moorman 2000).** Given an embedding dimension
`m` and a tolerance `r`, sample entropy is:

```
SampEn(x; m, r) = -log( A / B )
```

where:

- `B` is the number of pairs of length-`m` subsequences in `x` whose
  Chebyshev (max-norm) distance is ≤ `r`, counted with `i ≠ j`.
- `A` is the same count for length-`(m+1)` subsequences.

Concretely, let `X_i^(m) = (x[i], x[i+1], …, x[i+m-1])` for
`i = 0, …, T - m`. Then:

```
B = | { (i, j) : i < j ≤ T-m,    max_k |X_i^(m)[k]  - X_j^(m)[k]|  ≤ r } |
A = | { (i, j) : i < j ≤ T-m-1,  max_k |X_i^(m+1)[k] - X_j^(m+1)[k]| ≤ r } |
```

Sample entropy is the negative log-probability that two `m`-length
patterns that match continue to match when extended by one more
sample. Higher SampEn ⇒ adding a sample is more often informative ⇒
the signal is harder to predict.

**Differences from approximate entropy (ApEn, Pincus 1991):** SampEn
excludes self-matches (`i ≠ j`), making it bias-free in a way ApEn
isn't. SampEn is the modern default.

**Parameter choices:**

- `m = 2` (default). Embedding dimension. Standard choice from
  Pincus 1991 + Richman 2000 + Sanchez 2021. Larger `m` is more
  selective but exponentially more expensive and noise-sensitive.
- `r_frac = 0.2` (default). Tolerance, applied as `r = r_frac * σ(x)`.
  Standard choice (Pincus 1991). Tying `r` to the per-trace std makes
  the feature comparable across traces with different amplitude scales.

**Worked examples.**

*Periodic sinusoid:* highly predictable. Any matching length-`m`
window almost certainly matches at length `m+1` too, so `A/B ≈ 1`
and `SampEn ≈ -log(1) = 0`. Pure sine has SampEn near 0.

*White Gaussian noise:* unpredictable. Length-`m` matches are rare and
extending them rarely matches, so `A/B` is small and `SampEn` is large
(typically ~2-3 for `m=2, r=0.2σ`).

**EGM interpretation.** Sample entropy measures *temporal regularity*.
A clean periodic activation has low SampEn; fragmented atrial EGM has
higher SampEn. This is one of the seven features Sanchez 2021 reported
as discriminating fibrotic from non-fibrotic tissue.

We delegate to `antropy.sample_entropy(x, order=m, metric="chebyshev")`
after computing `r = r_frac * x.std(ddof=0)`. Note antropy's
`sample_entropy` takes the `r` value indirectly via the `metric`
parameter; we'll need to verify the exact call site against the antropy
API at implementation time.

**Reference:** Richman & Moorman 2000 (the original SampEn paper);
Pincus 1991 (ApEn predecessor, parameter guidance still applies).

### 3.2 shannon_entropy

**Definition.** Shannon entropy of the histogram of signal amplitudes,
binned into `n_bins` equal-width bins between `min(x)` and `max(x)`:

```
Let bin_edges = linspace(min(x), max(x), n_bins + 1)
Let c_k = count of samples in bin k                              # histogram
Let p_k = c_k / T                                                # probabilities
shannon_entropy(x; n_bins) = -Σ_k p_k · log(p_k)                 # nats
```

By convention `0 · log(0) = 0`. We return in **nats** (natural log),
not bits (log base 2), matching information-theory convention. Range
is `[0, log(n_bins)]`; not normalized.

**Parameter choices:**

- `n_bins = 10` (default). Sturges' rule for `T = 512` gives
  `ceil(log2(512) + 1) = 10`, which is the bin count statisticians
  recommend for histograms of that sample count. More bins captures
  finer amplitude structure but introduces sampling noise (some bins
  will be empty); fewer bins blurs the distribution.

**Worked examples.**

*Constant signal `x[i] = c` for all i:* all samples land in one bin.
`p_k = 1` for that bin, `p_k = 0` for all others. Entropy = 0.

*Uniform-in-amplitude signal (random samples drawn uniformly from
[min, max]):* `p_k ≈ 1/n_bins` for all k. Entropy = `log(n_bins)`.
For `n_bins = 10`, that's `log(10) ≈ 2.303` nats.

*White Gaussian noise:* Gaussian distribution, so a few bins near the
mean carry most probability. Entropy is between the constant and
uniform extremes — typically ~2.0 nats for `n_bins = 10`.

**EGM interpretation.** Amplitude-distribution entropy. A trace that
spends most of its time near baseline with occasional spikes has low
shannon_entropy (one big bin around zero). A trace that wanders broadly
across amplitudes has high shannon_entropy. This complements
spectral_entropy: a trace can be peaked in frequency space (low
spectral_entropy) but broad in amplitude space (high shannon_entropy)
if it's a clean wave that swings widely.

**Implementation note.** We compute this directly with `numpy.histogram`
+ `scipy.stats.entropy`. No antropy dependency for this one; the math is
short enough that the direct implementation is clearer.

**Reference:** Shannon 1948.

### 3.3 lempel_ziv_complexity

**The most "what is this actually doing" feature in the bundle.**
Spending extra space here so a non-specialist reviewer can verify the
math.

**Step 1: binarize the signal.** Lempel-Ziv complexity is defined on a
**string of symbols from a finite alphabet** — typically binary. So
the first step is to convert the continuous-valued signal `x` to a
sequence of 0s and 1s. We use the **median** method:

```
b[i] = 1 if x[i] > median(x), else 0
```

So `b` is a binary string of length `T`. Median is robust to baseline
drift (zero would also work for bandpass-filtered EGMs since they're
zero-mean by construction, but median works regardless of mean).

**Step 2: parse the binary string using the LZ76 algorithm.** The
Lempel-Ziv 1976 ("LZ76") complexity is the number of **distinct
substrings** encountered when scanning the string left-to-right and
incrementally building a dictionary. The exact procedure:

1. Initialize: `c = 1` (complexity counter), `i = 0` (read pointer),
   `w = b[0]` (current "word" being scanned).
2. While `i < T - 1`:
   - Let `next_char = b[i+1]`.
   - Look at the candidate next word `w + next_char` (the current word
     extended by one character). Has it already appeared as a contiguous
     substring of `b[0..i]`?
     - If **yes**: extend the current word: `w = w + next_char`,
       `i += 1`.
     - If **no**: this is a new distinct substring. Increment `c`, reset
       the current word: `w = next_char`, `i += 1`.
3. After the loop, `c` is the raw LZ76 complexity count.

**Step 3: normalize.** The raw count `c` grows with sequence length.
Lempel & Ziv 1976 proved that for a random binary sequence,
`c ≈ T / log_2(T)` asymptotically. So we normalize:

```
lempel_ziv_complexity(x) = c · log_2(T) / T
```

This gives a value in roughly `(0, 1]`:

- `≈ 0`: highly compressible / periodic.
- `≈ 1`: random-like / incompressible.

Note "true Kolmogorov complexity" is **incomputable** in general — it's
the length of the shortest program that produces the sequence, and
there's no algorithm that finds the shortest program. LZ76 is the
standard *computable proxy* used in the EGM and EEG complexity
literature. We kept the name `lempel_ziv_complexity` (renamed from
`kolmogorov_complexity` in the v1_baseline diagnostic) to be precise
about what's actually being computed.

**Worked examples.**

*Highly periodic sequence:* `b = "01010101..."` (alternating). After
median-binarization of a pure sine wave we get exactly this. Walking
through LZ76:

- `c=1`, `w="0"`, `i=0`.
- `i=0`: candidate `w + b[1] = "01"`. Is "01" a substring of `b[0..0] = "0"`? **No.** Increment: `c=2`, `w="1"`, `i=1`.
- `i=1`: candidate `"10"`. Is "10" a substring of `b[0..1] = "01"`? **No.** Increment: `c=3`, `w="0"`, `i=2`.
- `i=2`: candidate `"01"`. Is "01" a substring of `b[0..2] = "010"`? **Yes.** Extend: `w="01"`, `i=3`.
- `i=3`: candidate `"010"`. Is "010" a substring of `b[0..3] = "0101"`? **Yes** (at position 0). Extend: `w="010"`, `i=4`.
- `i=4`: candidate `"0101"`. Is "0101" a substring of `b[0..4] = "01010"`? **Yes** (at position 0). Extend: `w="0101"`, `i=5`.
- ... continues, never finding new substrings. Loop ends. `c = 3`.

So `lempel_ziv_complexity(periodic_sequence) ≈ 3 · log_2(T) / T → 0`
for large `T`. The test fixture verifies this is small (< 0.1) for a
pure sine.

*Random binary sequence:* every new short substring is novel for a
while, so the dictionary grows fast initially. Asymptotically
`c ≈ T / log_2(T)`, so `lempel_ziv_complexity ≈ 1`. Test fixture
verifies this is high (> 0.7) for `numpy.random.randint(0, 2, T)`.

**Parameter choices:**

- `binarize_method = "median"` (default). 1 if above median, 0 if
  below. Robust to baseline drift; most common in the EGM-LZ literature.
- A `"zero"` mode is reserved for explicit users (1 if `x > 0`, 0
  else). For bandpassed EGMs the two methods give nearly identical
  results since the bandpass forces `median ≈ 0`.

**EGM interpretation.** Lempel-Ziv complexity captures the "how
many distinct patterns are in this signal" intuition. A clean periodic
activation has low LZ. A fragmented signal with multiple deflections at
varying intervals has higher LZ. This complements sample_entropy
(temporal predictability) and spectral_entropy (frequency-domain
spread) — three different angles on the same intuition of "complexity."

**Implementation note.** We binarize ourselves in
`complexity.py`, then call `antropy.lziv_complexity(binarized, normalize=True)`.
antropy's `lziv_complexity` accepts a numpy array of 0/1 values directly,
applies the LZ76 algorithm above, and (with `normalize=True`) returns the
normalized value `c · log_2(T) / T`.

**Reference:** Lempel & Ziv 1976 (the original LZ complexity); Aboy 2006
(survey of LZ applied to physiological signals, including EGM/ECG).

### 3.4 higuchi_fractal_dimension

**Definition (Higuchi 1988).** Treat the time series as a curve in the
plane (sample index on x-axis, signal value on y-axis). The **fractal
dimension** of this curve measures how much detail it has at fine
scales — a straight line has dimension 1, a plane-filling curve has
dimension 2, and real signals lie between.

Higuchi's algorithm estimates the fractal dimension via a length-vs-scale
analysis. The procedure:

1. For each scale `k = 1, 2, …, k_max`:
   - For each starting offset `m = 1, …, k`:
     - Construct the "downsampled" sub-series:
       `X_k^m = (x[m], x[m+k], x[m+2k], …, x[m + ⌊(T-m)/k⌋ · k])`
     - Compute its **length** (sum of absolute differences between
       consecutive samples, normalized):
       `L_m(k) = ( 1/k · Σ_{i=1}^{n_m} | x[m + ik] - x[m + (i-1)k] | ) · (T - 1) / (n_m · k)`
       where `n_m = ⌊(T - m) / k⌋`.
   - Average over starting offsets: `L(k) = (1/k) · Σ_m L_m(k)`.
2. Higuchi shows that for a fractal curve, `L(k) ∝ k^{-D}` where `D`
   is the fractal dimension. Equivalently, `log L(k) = -D · log k + const`.
3. Fit a line to `(log k, log L(k))` for `k = 1, …, k_max`. The slope
   is `-D`; return `D`.

The result is dimensionless. For a 1D signal it's in `[1, 2]`:
- `D = 1`: perfectly smooth (straight line).
- `D = 2`: maximally rough (space-filling, white-noise-like).

**Parameter choices:**

- `k_max = 10` (default). Maximum scale for the length-vs-scale
  analysis. Literature uses anywhere from 8 to 64; Higuchi's original
  paper went up to 100. Smaller `k_max` is faster but the linear fit is
  less stable; larger is slower with diminishing accuracy improvement.
  10 is a pragmatic mid-range that works well for `T = 512`.

**Worked examples.**

*Straight line `x[i] = a · i + b`:* should give `D = 1` (the topological
dimension of a line). Walking through Higuchi's formula:

- Each consecutive difference at scale `k` is `|x[m + ik] - x[m + (i-1)k]| = |a · k| = |a| · k`.
- Sum over `i = 1, …, n_m`: `Σ |diff| = n_m · |a| · k`.
- Multiply by `1/k`: `n_m · |a|`.
- Multiply by `(T-1) / (n_m · k)`: `L_m(k) = |a| · (T-1) / k`.
- Average over starting offsets `m`: same value, so `L(k) = |a| · (T-1) / k`.

Therefore `log L(k) = log(|a|(T-1)) − log(k)`, which is a line with
slope `-1` in the `(log k, log L(k))` plane. Higuchi defines
`L(k) ∝ k^{-D}`, so slope `= -D`, giving **`D = 1`** for a straight
line. ✓

*White Gaussian noise:* the consecutive differences at scale `k` are
samples of `x[m + ik] - x[m + (i-1)k]`, which for independent Gaussian
samples are themselves Gaussian with mean 0 and a constant std
independent of `k`. The expected absolute difference is some constant
`c`. Working through the formula:

- `Σ |diff| ≈ n_m · c`.
- Multiply by `1/k`: `n_m · c / k`.
- Multiply by `(T-1) / (n_m · k)`: `L_m(k) ≈ c · (T-1) / k²`.

So `log L(k) ≈ log(c · (T-1)) − 2 · log(k)`, slope `-2`, giving
**`D = 2`** for white noise. ✓

These two anchors (`D = 1` for a line, `D = 2` for white noise) are
exactly what we'd want from a fractal-dimension measure on 1D signals.
Real EGM traces fall somewhere between — typically `1.3 - 1.7` for a
clean activation, higher for fragmented traces. The test fixture
verifies these boundaries: a perturbed line gives `D < 1.2`, white
noise gives `D > 1.9`.

**EGM interpretation.** Higuchi FD measures how "rough" or "filled" the
signal trajectory is at fine scales. A smooth biphasic activation has
relatively low FD; a fragmented signal with many small deflections has
higher FD. Like sample_entropy and lempel_ziv_complexity, this captures
a "complexity" intuition but from a geometric angle rather than a
predictability or compressibility angle.

We delegate to `antropy.higuchi_fd(x, kmax=k_max)`.

**Reference:** Higuchi 1988 (the original algorithm); Esteller 2001
(comparison of fractal-dimension methods on biomedical signals).

---

## 4. References

### Core feature papers

- **Marchlinski FE, Callans DJ, Gottlieb CD, Zado E.** *Linear ablation
  lesions for control of unmappable ventricular tachycardia in patients
  with ischemic and nonischemic cardiomyopathy.* Circulation 2000.
  [Bipolar voltage threshold convention, used for `peak_to_peak`.]
- **Sanders P et al.** *Spectral analysis identifies sites of
  high-frequency activity maintaining atrial fibrillation in humans.*
  Circulation 2005. [Atrial dominant-frequency analysis.]
- **Kosiuk J et al.** *Validation of voltage mapping during AF: 0.2 mV
  threshold.* (PMID 30873619). [AF-adjusted voltage thresholds.]
- **Nademanee K et al.** *A new approach for catheter ablation of atrial
  fibrillation: mapping of the electrophysiologic substrate (CFAE).*
  J Am Coll Cardiol 2004. [Complex-fractionated EGM definition,
  background for `sec_peak_count`.]

### Entropy + complexity

- **Shannon CE.** *A Mathematical Theory of Communication.* Bell System
  Technical Journal 1948. [Original Shannon entropy.]
- **Pincus SM.** *Approximate entropy as a measure of system
  complexity.* PNAS 1991. [ApEn, predecessor of SampEn; parameter
  guidance.]
- **Richman JS, Moorman JR.** *Physiological time-series analysis using
  approximate entropy and sample entropy.* Am J Physiol Heart Circ
  Physiol 2000. [Sample entropy.]
- **Inouye T et al.** *Quantification of EEG irregularity by use of
  the entropy of the power spectrum.* Electroencephalogr Clin
  Neurophysiol 1991. [Spectral entropy in biomedical signals.]

### Fractal + LZ

- **Higuchi T.** *Approach to an irregular time series on the basis of
  the fractal theory.* Physica D 1988. [The Higuchi fractal dimension
  algorithm.]
- **Lempel A, Ziv J.** *On the complexity of finite sequences.* IEEE
  Trans Inf Theory 1976. [LZ76, the algorithm underneath
  `lempel_ziv_complexity`.]
- **Aboy M, Hornero R, Abásolo D, Álvarez D.** *Interpretation of the
  Lempel-Ziv complexity measure in the context of biomedical signal
  analysis.* IEEE Trans Biomed Eng 2006. [LZ applied to physiological
  signals.]
- **Esteller R, Vachtsevanos G, Echauz J, Litt B.** *A comparison of
  waveform fractal dimension algorithms.* IEEE Trans Circuits Syst I
  2001. [Comparison of Higuchi vs Katz vs Petrosian methods.]

### Project context

- **Sanchez J et al. 2021.** *Using Machine Learning to Characterize
  Atrial Fibrotic Substrate From Intracardiac Signals With a Hybrid in
  silico and in vivo Dataset.* Frontiers in Physiology. The paper this
  feature set most closely tracks — Sanchez used a 7-feature subset
  (peak-to-peak, duration, sample entropy, Shannon entropy, spectral
  entropy, Kolmogorov complexity, fractal dimension) and trained a
  decision tree classifier. The 11 features here are Sanchez's 7 plus
  four more (`zero_crossings`, `activation_position`, `sec_peak_count`,
  `dominant_frequency`) added during the v1_baseline diagnostic.
- See `intracardiac-platform/project/architecture_reading_list.md` for
  the broader literature this project draws on.
