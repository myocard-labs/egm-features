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

For the *what* and *where* of each feature (call signatures, return
shape, column names in the bundle DataFrame), see
[`docs/usage.md`](usage.md). For the design rationale behind the
library's overall shape (pure functions, per-trace `(T,)` inputs, no
CLI), see [`project/architecture.md`](../project/architecture.md).

> **Rendering note.** Equations are written in LaTeX — `$$…$$` for
> display, `$…$` for inline. GitHub and VS Code render these as typeset
> math; in a plain-text viewer they show as LaTeX source. Backticked
> names (`fs_hz`, `bundle.extract_all`) are code identifiers, not math
> symbols.

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
- [4. catch22 features](#4-catch22-features)
  - [4.1 About the catch22 set](#41-about-the-catch22-set)
    - [4.1.1 What catch22 is, and how it differs from §1–§3](#411-what-catch22-is-and-how-it-differs-from-13)
    - [4.1.2 Naming — the hctsa code is the source of truth](#412-naming--the-hctsa-code-is-the-source-of-truth)
    - [4.1.3 Reliability at T = 192, and the usable-now set](#413-reliability-at-t--192-and-the-usable-now-set)
    - [4.1.4 Degenerate input — NaN, and why it must warn](#414-degenerate-input--nan-and-why-it-must-warn)
    - [4.1.5 The anchor signals](#415-the-anchor-signals)
  - [4.2 Distribution shape](#42-distribution-shape)
    - [4.2.1 mode_5](#421-mode_5) · [4.2.2 mode_10](#422-mode_10)
  - [4.3 Extreme-event timing](#43-extreme-event-timing)
    - [4.3.1 outlier_timing_pos](#431-outlier_timing_pos) · [4.3.2 outlier_timing_neg](#432-outlier_timing_neg)
  - [4.4 Linear autocorrelation structure](#44-linear-autocorrelation-structure)
    - [4.4.1 acf_timescale](#441-acf_timescale) · [4.4.2 acf_first_min](#442-acf_first_min) · [4.4.3 periodicity](#443-periodicity)
    - [4.4.4 low_freq_power](#444-low_freq_power) · [4.4.5 centroid_freq](#445-centroid_freq) · [4.4.6 ami_timescale](#446-ami_timescale)
  - [4.5 Nonlinear autocorrelation](#45-nonlinear-autocorrelation)
    - [4.5.1 trev](#451-trev) · [4.5.2 ami2](#452-ami2)
  - [4.6 Simple forecasting](#46-simple-forecasting)
    - [4.6.1 forecast_error](#461-forecast_error)
  - [4.7 Incremental differences](#47-incremental-differences)
    - [4.7.1 high_fluctuation](#471-high_fluctuation) · [4.7.2 whiten_timescale](#472-whiten_timescale)
  - [4.8 Symbolic](#48-symbolic)
    - [4.8.1 stretch_high](#481-stretch_high) · [4.8.2 stretch_decreasing](#482-stretch_decreasing)
    - [4.8.3 entropy_pairs](#483-entropy_pairs) · [4.8.4 transition_variance](#484-transition_variance)
  - [4.9 Self-affine scaling](#49-self-affine-scaling)
    - [4.9.1 rs_range](#491-rs_range) · [4.9.2 dfa](#492-dfa)
  - [4.10 Other](#410-other)
    - [4.10.1 embedding_dist](#4101-embedding_dist)
  - [4.11 The catch24 additions](#411-the-catch24-additions)
    - [4.11.1 mean](#4111-mean) · [4.11.2 std_dev](#4112-std_dev)
- [5. References](#5-references)

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

**One extra assumption for §4.** Every catch22 feature is computed on the
$z$-scored trace, so it is blind to the signal's mean and amplitude by
design — the whole voltage axis that clinical fibrosis mapping rests on
(`peak_to_peak < 0.5 mV`) is invisible to it. That is *why* §4 is an
addition to §1–§3 rather than a replacement: catch22 supplies morphology
and dynamics, §1–§3 supply amplitude and activation. See
[§4.1.1](#411-what-catch22-is-and-how-it-differs-from-13).

Also: several §4 features are indexed in **samples or lags** rather than
Hz (autocorrelation timescales, `whiten_timescale`, `embedding_dist`), and
one — `forecast_error` — has a fixed 3-sample horizon. Their values are
only comparable between two corpora sampled at the **same rate**, and
nothing in the call signature carries `fs_hz` to warn you. Synthetic and
IAFDB are both 1 kHz, and Phase 1.5 fixed that as a constraint: any rate
change is both-sides-or-neither.

## Notation

- $x \in \mathbb{R}^{T}$ — a single per-trace input; $T$ is the sample count.
- $T$ — number of samples in the trace. **Phase 1.5 fixes $T = 192$**
  (192 ms at 1 kHz): the §8.1 study put the single-beat window in the
  150–250 ms range, and the classifier's 1-D MobileViT requires
  $T \equiv 0 \pmod{64}$, so 192 is that range rounded onto the 64-grid.
  Worked examples throughout §4 use $T = 192$.

  **§1–§3 predate this change.** Their worked examples were written at
  $T = 512$ and each states its own window length explicitly, so they
  remain valid as illustrations — but two §1–§3 *parameter choices* were
  justified by $T = 512$ and are worth re-reading at 192:

  - `shannon_entropy`'s `n_bins = 10` (§3.2) came from Sturges' rule,
    $1 + \log_2 T$, which gives $\approx 10$ at 512 but $\approx 9.6$ at
    192. The value is unchanged for now — changing it would move every
    `shannon_entropy` result and break comparability with Phase-1 banks —
    but the stated justification is now approximate rather than exact.
  - Frequency resolution (§2.1) is $f_s/T$, so it coarsens from
    $\approx 2$ Hz at 512 to $\approx 5.2$ Hz at 192. `spectral_centroid`
    and `dominant_frequency` are correspondingly less precise on
    Phase-1.5 traces; `dominant_frequency`, which reports a single bin
    centre, is the more affected of the two.
- $\tilde{x}_i = (x_i - \mu)/\sigma$ — the $z$-scored trace. §1–§3 operate
  on $x$ as given; **every §4 (catch22) feature operates on $\tilde{x}$**,
  internally and unconditionally.
- `fs_hz` — sample rate in Hz. The features that touch the time axis
  (frequency-domain ones, activation-position-as-time variants) need it;
  the dimensionless complexity features don't.
- $t_i = i / f_s$ — time at sample index $i$ (with $f_s$ = `fs_hz`).
- $\mu(x)$, $\sigma(x)$, $\operatorname{med}(x)$ — mean, std (population, ddof=0), median.
- `NDArray` — a `numpy.ndarray` of floating-point type; specifically
  shape `(T,)` for the per-trace functions. Batch processing over
  $(N, T)$ is `bundle.extract_all`'s job, not the individual extractors.

---

## 1. Time-domain features

### 1.1 peak_to_peak

**Definition.**

$$
\text{peak\_to\_peak}(x) = \max(x) - \min(x)
$$

In mV (or whatever amplitude unit the caller passed in).

**Worked example.** A pure sinusoid $x[i] = A \sin(2\pi f \, t_i)$:

$$
\max(x) = A, \quad \min(x) = -A \;\;\Rightarrow\;\; \text{peak\_to\_peak} = 2A
$$

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

$$
\text{zero\_crossings}(x) = \bigl| \{\, i : \operatorname{sign}(x[i]) \ne \operatorname{sign}(x[i-1]),\; 1 \le i < T \,\} \bigr|
$$

The exact-zero case (`x[i] = 0`) is folded into `sign` via NumPy's
convention (`sign(0) = 0`), so a value of exactly zero is treated as a
"both directions" indicator and counted as a crossing on either side.
For float-valued bandpass-filtered signals this edge case effectively
never fires.

**Worked example.** A pure sinusoid at frequency `f` over duration
`T_seconds`:

$$
\text{zero\_crossings} \approx 2 f \, T_{\text{seconds}}
$$

For $f = 50$ Hz, $T_{\text{seconds}} = 0.512$ s (= 512 samples at 1 kHz):
$2 \cdot 50 \cdot 0.512 = 51.2$, so ~51 crossings. The test fixture verifies
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

$$
\text{activation\_position}(x) = \operatorname*{arg\,max}_i \lvert dx/dt \rvert \,/\, (T - 1)
$$

where $dx/dt$ is approximated by the discrete first difference
$x[i] - x[i-1]$ for $i = 1, \dots, T-1$.

**Why max-slope and not peak-amplitude.** The clinical activation-time
convention (Marchlinski / Wittkampf school) takes the steepest *slope*
as the moment of activation under the electrode — the wavefront passes
under the bipolar pair, the field flips polarity rapidly, and that
steepest dV/dt point is when activation "happens." Peak amplitude
(argmax of `|x|`) is also commonly used and slightly easier to compute,
but it tracks the post-activation extremum rather than the activation
event itself.

**Worked example.** A simple biphasic activation:

$$
x[i] =
\begin{cases}
-A \sin\bigl(2\pi (i - k) / w\bigr) & k - w/2 < i \le k + w/2 \\
0 & \text{elsewhere}
\end{cases}
$$

This is one cycle of a sine, centered at sample $k$, lasting $w$
samples. The steepest slope occurs at the zero-crossing in the middle
of the pulse (sample $i = k$), so $\text{activation\_position} = k / (T - 1)$.

If the trace is $T = 512$ samples and the activation is centered at
sample $k = 256$, we get $\text{activation\_position} \approx 0.5$. The test fixture
exercises this with a few $k$ values.

**Parameter choices.** `method` is a **required** parameter (policy
value; no default per the math-vs-policy split in `project/architecture.md`).
Project standard: `method="dvdt_max"` — the clinical convention from
the Marchlinski / Wittkampf school. Used automatically by
`bundle.extract_all`. Alternative: `method="abs_peak"` for the simpler
argmax-of-`|x|` variant; direct callers pick one explicitly.

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

1. Compute the rectified signal $|x|$.
2. Find the primary peak amplitude $A_{\max} = \max(|x|)$.
3. Find all local maxima of $|x|$ with prominence
   $\ge \text{threshold\_frac} \cdot A_{\max}$.
4. Subtract 1 (to exclude the primary peak itself), with a floor at 0.

In code:

```python
abs_x = np.abs(x)
A_max = float(abs_x.max())
peaks, _ = scipy.signal.find_peaks(abs_x, prominence=threshold_frac * A_max)
sec_peak_count = max(0, len(peaks) - 1)
```

**Worked example 1 — single positive bump.** A signal with a single
Gaussian-like positive bump has exactly one local maximum in $|x|$ at
the bump's peak. With `threshold_frac=0.3`, that one peak is the
primary; nothing else clears the threshold. $\text{sec\_peak\_count} = 0$.

> **Note on real biphasic EGM activations.** A *biphasic* activation
> (positive lobe followed by negative lobe, the realistic clinical
> case from §1.3) has TWO lobes in `|x|` of comparable magnitude.
> `find_peaks` returns 2, and `sec_peak_count = 1` for a clean
> biphasic. That's the algorithm's natural baseline on real EGMs —
> the metric is most useful for discriminating clean biphasic (1)
> from fragmented activations (3+). The §1.4 worked examples use
> simplified positive-only bumps to make the math easier to
> verify; the §1.3 biphasic model is the realistic activation
> shape and gives `sec_peak_count = 1` on a clean trace.

**Worked example 2 — fragmented two-component activation.** A primary
activation of amplitude `A` followed by a secondary activation of
amplitude `A/2`:

```
|x|:    ___/\___/\___    (two bumps, second is half the height)
```

With `threshold_frac = 0.3`, the secondary peak's prominence
($A/2 = 0.5A$) exceeds the threshold ($0.3A$), so it counts.
$\text{sec\_peak\_count} = 1$.

**Parameter choices.** `threshold_frac` is a **required** parameter
(policy value; no default per the math-vs-policy split in
`project/architecture.md`). Project standard: `threshold_frac = 0.3`
(30% of primary peak amplitude). Used automatically by
`bundle.extract_all`. Reasoning: the fragmentation-index family in
the literature (Nademanee 2004 CFAE definition, Kim 2014 fractionation
analyses) uses similar fractional thresholds, typically 30–50%. Lower
values count more peaks (noise-sensitive); higher values miss
legitimate secondary deflections. 0.3 is the pragmatic middle.
Direct callers can pass a different fraction (e.g. 0.5 to match the
strict CFAE convention).

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

$$
\hat{P}(f_k) = \frac{1}{f_s \, T} \, \bigl\lvert \operatorname{DFT}(x)[k] \bigr\rvert^2
$$

where $f_k = k \, f_s / T$ for $k = 0, 1, \dots, T/2$ (Nyquist-limited;
the DFT is symmetric for real-valued $x$ so we only keep the positive
half) and $\operatorname{DFT}(x)[k]$ is the discrete Fourier transform.

The units of $\hat{P}$ are $(\text{signal unit})^2 / \text{Hz}$. We don't care about the
absolute scale for the three downstream features (centroid uses
normalized weighting; entropy uses normalized probabilities; dominant
frequency uses argmax which is scale-invariant) — so the scale factor
out front cancels in every consumer.

`scipy.signal.periodogram` returns two arrays: `f` (length `T/2 + 1`)
and the PSD $\hat{P}$ (same length). We pass both to the consumers below.

**PSD reuse across features.** The three frequency features below all
consume the same $(f, \hat{P})$ pair. To avoid recomputing the periodogram
three times per trace when running all three (the bundle.extract_all
hot path), the public `periodogram(signal, fs_hz)` function is exposed
as a building block, and each feature function accepts optional
`freqs` and `psd` keyword arguments — when both are supplied, the
feature uses them directly and skips its internal PSD call.
Single-feature calls stay a one-liner; multi-feature batches compute
the PSD once.

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

$$
\text{spectral\_centroid}(x) = \frac{\sum_k f_k \, \hat{P}(f_k)}{\sum_k \hat{P}(f_k)}
$$

Units: Hz.

**Worked example.** A pure sinusoid at frequency $f_0$. Its PSD is a
delta at $f_0$ (in the continuous limit; in the discrete periodogram,
it's a sharp peak at the nearest bin):

$$
\text{spectral\_centroid} \approx f_0
$$

For a 50 Hz sine over a 512-sample window at 1 kHz, the bin closest to
50 Hz is `k = round(50 · 512 / 1000) = 26`, so `f_26 ≈ 50.78 Hz`. The
centroid is essentially that bin's frequency. Test verifies
`|spectral_centroid - 50| < 2` Hz (one-bin tolerance).

**Worked example 2 — two sinusoids at different frequencies *and*
amplitudes.** Take `x = 1.0 · sin(2π · 50 · t) + 2.0 · sin(2π · 150 · t)`
— a 50 Hz component at amplitude 1, plus a 150 Hz component at
amplitude 2. The PSD has two peaks. Key point: PSD weights by *power*
(amplitude²), not amplitude. So:

- Power at 50 Hz: $\propto 1^2 = 1$
- Power at 150 Hz: $\propto 2^2 = 4$

Centroid:

$$
\text{spectral\_centroid} = \frac{50 \cdot 1 + 150 \cdot 4}{1 + 4} = \frac{650}{5} = 130 \text{ Hz}
$$

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

$$
p_k = \frac{\hat{P}(f_k)}{\sum_j \hat{P}(f_j)}, \qquad
\text{spectral\_entropy}(x) = \frac{-\sum_k p_k \log(p_k)}{\log(N_{\text{bins}})}
$$

The left expression treats the PSD as a probability mass function; the
right normalizes the Shannon entropy to $[0, 1]$.

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

$$
H = -(0.2 \ln 0.2 + 0.8 \ln 0.8) = -\bigl(0.2 \cdot (-1.609) + 0.8 \cdot (-0.223)\bigr) = 0.322 + 0.179 = 0.500 \text{ nats}
$$

Normalizing by `log(N_bins)` (with `N_bins ≈ 257` for `T = 512` at
the periodogram resolution):

$$
\text{spectral\_entropy} = 0.500 / \log(257) \approx 0.090
$$

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

$$
\text{dominant\_frequency}(x) = \operatorname*{arg\,max}_k \hat{P}(f_k) \;\to\; f_{k^*}
$$

Units: Hz.

**Worked example.** A pure sinusoid at $f_0 = 50$ Hz over 512 samples
at 1 kHz: `dominant_frequency ≈ 50.78 Hz` (the nearest bin). Test
verifies `|dominant_frequency - 50| < 2` Hz.

**Worked example 2 — two sinusoids, different amplitudes.** Same setup
as §2.2's multi-sinusoid example (50 Hz amplitude 1, 150 Hz amplitude
2). The PSD has two peaks; the 150 Hz peak has 4× the power of the
50 Hz peak. `argmax` picks the higher-power bin:

$$
\text{dominant\_frequency} = 150 \text{ Hz} \quad (\text{nearest bin} \approx 150.39 \text{ Hz})
$$

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

$$
\mathrm{SampEn}(x; m, r) = -\log\left( \frac{A}{B} \right)
$$

where:

- $B$ is the number of pairs of length-$m$ subsequences in $x$ whose
  Chebyshev (max-norm) distance is $\le r$, counted with $i \ne j$.
- $A$ is the same count for length-$(m+1)$ subsequences.

Concretely, let $X_i^{(m)} = (x[i], x[i+1], \dots, x[i+m-1])$ for
$i = 0, \dots, T - m$. Then:

$$
\begin{aligned}
B &= \bigl| \{\, (i, j) : i < j \le T-m,\; \max_k \lvert X_i^{(m)}[k] - X_j^{(m)}[k] \rvert \le r \,\} \bigr| \\
A &= \bigl| \{\, (i, j) : i < j \le T-m-1,\; \max_k \lvert X_i^{(m+1)}[k] - X_j^{(m+1)}[k] \rvert \le r \,\} \bigr|
\end{aligned}
$$

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

*Periodic sinusoid:* highly predictable. Any matching length-$m$
window almost certainly matches at length $m+1$ too, so $A/B \approx 1$
and $\mathrm{SampEn} \approx -\log(1) = 0$. Pure sine has SampEn near 0.

*White Gaussian noise:* unpredictable. Length-$m$ matches are rare and
extending them rarely matches, so $A/B$ is small and $\mathrm{SampEn}$ is large
(typically ~2-3 for $m=2$, $r=0.2\sigma$).

**EGM interpretation.** Sample entropy measures *temporal regularity*.
A clean periodic activation has low SampEn; fragmented atrial EGM has
higher SampEn. This is one of the seven features Sanchez 2021 reported
as discriminating fibrotic from non-fibrotic tissue.

We delegate to
`antropy.sample_entropy(x, order=m, tolerance=r_frac * x.std(ddof=0), metric="chebyshev")`.
antropy accepts a custom `tolerance` directly (defaulting to
`0.2 * std(x)` when `None`), so the full theory-spec parameter set
(`m`, `r_frac`) maps cleanly onto the antropy call.

**Reference:** Richman & Moorman 2000 (the original SampEn paper);
Pincus 1991 (ApEn predecessor, parameter guidance still applies).

### 3.2 shannon_entropy

**Definition.** Shannon entropy of the histogram of signal amplitudes,
binned into `n_bins` equal-width bins between `min(x)` and `max(x)`:

$$
p_k = \frac{c_k}{T}, \qquad
\text{shannon\_entropy}(x; n_{\text{bins}}) = -\sum_k p_k \log(p_k)
$$

where $c_k$ is the count of samples in bin $k$ (bin edges $=$
`linspace(min(x), max(x), n_bins + 1)`) and $p_k$ the bin probability.

By convention $0 \cdot \log(0) = 0$. We return in **nats** (natural log),
not bits (log base 2), matching information-theory convention. Range
is $[0, \log(n_{\text{bins}})]$; not normalized.

**Parameter choices:**

- `n_bins = 10` (default). Sturges' rule for `T = 512` gives
  `ceil(log2(512) + 1) = 10`, which is the bin count statisticians
  recommend for histograms of that sample count. More bins captures
  finer amplitude structure but introduces sampling noise (some bins
  will be empty); fewer bins blurs the distribution.

**Worked examples.**

*Constant signal $x[i] = c$ for all $i$:* all samples land in one bin.
$p_k = 1$ for that bin, $p_k = 0$ for all others. Entropy = 0.

*Uniform-in-amplitude signal (random samples drawn uniformly from
[min, max]):* $p_k \approx 1/n_{\text{bins}}$ for all $k$. Entropy = $\log(n_{\text{bins}})$.
For `n_bins = 10`, that's $\log(10) \approx 2.303$ nats.

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

$$
b[i] = \begin{cases} 1 & x[i] > \operatorname{med}(x) \\ 0 & \text{otherwise} \end{cases}
$$

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
$c \approx T / \log_2(T)$ asymptotically. So we normalize:

$$
\text{lempel\_ziv\_complexity}(x) = c \cdot \frac{\log_2(T)}{T}
$$

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

So $\text{lempel\_ziv\_complexity}(\text{alternating}) \approx 3 \log_2(T) / T \to 0$
for large $T$. The test fixture verifies this is small (< 0.1) for a
signal constructed to median-binarize to exactly "01010101..." (e.g.
sample-alternating ±1 amplitudes).

**Real sines give a higher value than the alternating-bit anchor.** A
pure 50 Hz sine sampled at 1 kHz binarizes to *chunks* — each half-
cycle is ~10 samples wide, so the binary string is
"0000000000 1111111111 0000000000 …" rather than "01010101…". The
dictionary takes longer to saturate over half-cycle chunks than over
single-bit alternation, so the empirical LZ value sits around `0.26`
for a 50 Hz sine at fs=1 kHz, N=512 — well below random (~1) but
above the alternating-bit asymptote (<0.1). The trend is preserved
(periodic < random) but the absolute value depends on the binary
chunk size, not just on periodicity.

*Random binary sequence:* every new short substring is novel for a
while, so the dictionary grows fast initially. Asymptotically
$c \approx T / \log_2(T)$, so $\text{lempel\_ziv\_complexity} \approx 1$. Test fixture
verifies this is high (> 0.7) for `numpy.random.randint(0, 2, T)`.

**Parameter choices:**

- `binarize_method` is a **required** parameter (policy value; no
  default per the math-vs-policy split in `project/architecture.md`).
  Project standard: `binarize_method="median"` — 1 if above median, 0
  if below; robust to baseline drift; most common in the EGM-LZ
  literature. Used automatically by `bundle.extract_all`.
  Alternative: `"zero"` (1 if `x > 0`, 0 else); for bandpassed EGMs
  the two methods give nearly identical results since the bandpass
  forces `median ≈ 0`. Direct callers pick one explicitly.

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
normalized value $c \cdot \log_2(T) / T$.

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
       $X_k^m = (x[m], x[m+k], x[m+2k], \dots, x[m + \lfloor (T-m)/k \rfloor \, k])$
     - Compute its **length** (sum of absolute differences between
       consecutive samples, normalized):
       $L_m(k) = \left( \frac{1}{k} \sum_{i=1}^{n_m} \lvert x[m + ik] - x[m + (i-1)k] \rvert \right) \frac{T - 1}{n_m \, k}$
       where $n_m = \lfloor (T - m) / k \rfloor$.
   - Average over starting offsets: $L(k) = \frac{1}{k} \sum_m L_m(k)$.
2. Higuchi shows that for a fractal curve, $L(k) \propto k^{-D}$ where $D$
   is the fractal dimension. Equivalently, $\log L(k) = -D \log k + \text{const}$.
3. Fit a line to $(\log k, \log L(k))$ for $k = 1, \dots, k_{\max}$. The slope
   is $-D$; return $D$.

The result is dimensionless. For a 1D signal it's in $[1, 2]$:
- $D = 1$: perfectly smooth (straight line).
- $D = 2$: maximally rough (space-filling, white-noise-like).

**Parameter choices:**

- `k_max = 10` (default). Maximum scale for the length-vs-scale
  analysis. Literature uses anywhere from 8 to 64; Higuchi's original
  paper went up to 100. Smaller `k_max` is faster but the linear fit is
  less stable; larger is slower with diminishing accuracy improvement.
  10 is a pragmatic mid-range that works well for `T = 512`.

**Worked examples.**

*Straight line `x[i] = a · i + b`:* should give `D = 1` (the topological
dimension of a line). Walking through Higuchi's formula:

- Each consecutive difference at scale $k$ is $\lvert x[m + ik] - x[m + (i-1)k] \rvert = \lvert a k \rvert = \lvert a \rvert k$.
- Sum over $i = 1, \dots, n_m$: $\sum \lvert \text{diff} \rvert = n_m \lvert a \rvert k$.
- Multiply by $1/k$: $n_m \lvert a \rvert$.
- Multiply by $(T-1)/(n_m k)$: $L_m(k) = \lvert a \rvert (T-1) / k$.
- Average over starting offsets $m$: same value, so $L(k) = \lvert a \rvert (T-1) / k$.

Therefore $\log L(k) = \log(\lvert a \rvert (T-1)) - \log(k)$, which is a line with
slope $-1$ in the $(\log k, \log L(k))$ plane. Higuchi defines
$L(k) \propto k^{-D}$, so slope $= -D$, giving $D = 1$ for a straight
line. ✓

*White Gaussian noise:* the consecutive differences at scale `k` are
samples of `x[m + ik] - x[m + (i-1)k]`, which for independent Gaussian
samples are themselves Gaussian with mean 0 and a constant std
independent of `k`. The expected absolute difference is some constant
`c`. Working through the formula:

- $\sum \lvert \text{diff} \rvert \approx n_m c$.
- Multiply by $1/k$: $n_m c / k$.
- Multiply by $(T-1)/(n_m k)$: $L_m(k) \approx c (T-1) / k^2$.

So $\log L(k) \approx \log(c (T-1)) - 2 \log(k)$, slope $-2$, giving
$D = 2$ for white noise. ✓

These two anchors ($D = 1$ for a line, $D = 2$ for white noise) are
exactly what we'd want from a fractal-dimension measure on 1D signals.
Real EGM traces fall somewhere between — typically $1.3$–$1.7$ for a
clean activation, higher for fragmented traces. The test fixture
verifies these boundaries: a perturbed line gives $D < 1.2$, white
noise gives $D > 1.9$.

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

## 4. catch22 features

The 22 features of the **catch22** set (Lubba et al. 2019), plus the two
`catch24` additions. These come from a different tradition than §1–§3:
where our first eleven were chosen *because a cardiologist or a paper
named them*, catch22 was chosen **statistically** — ~4800 candidate
time-series features from the `hctsa` library filtered down to 22 that
perform well across many classification tasks while being minimally
redundant with each other. They are a general-purpose fingerprint of a
time series' shape and dynamics, not an EGM-specific instrument.

> **How to read this section.** Each entry gives the definition, what the
> number means, a **worked anchor** computed on a signal where the answer
> is checkable, and a short EGM read. Full derivations, the per-feature
> literature, and the longer EGM-relevance discussion live in
> `intracardiac-platform/project/investigations/catch22_techniques_explained.md`
> — this section deliberately does not duplicate them. What it *does* own
> is the part specific to this library: our naming, our parameter policy
> (there isn't one — see §4.1), the anchors the tests assert, and the
> reliability read at $T = 192$.

### 4.1 About the catch22 set

#### 4.1.1 What catch22 is, and how it differs from §1–§3

Three differences matter for anyone reading a feature DataFrame.

**1 · Everything is $z$-scored, so amplitude is gone.** Each feature is
computed on $\tilde{x}_i = (x_i - \mu)/\sigma$. catch22 measures
*time-ordering* properties and is deliberately insensitive to location and
scale. For EGM work this is a real loss and a real gain: the single most
established substrate marker — bipolar peak-to-peak voltage, the
`< 0.5 mV` scar threshold — is **invisible** to every feature in this
section, while morphology and rhythm are described far more richly than
§1–§3 manage. The two sets are complementary, which is the entire reason
we ship both. `peak_to_peak` (§1.1) remains the amplitude channel.

**2 · There are no parameters to choose.** §1–§3 carry a deliberate
math-constant-vs-project-policy split (`project/architecture.md`), because
`sample_entropy`'s $m$ and $r$ are ours to pick. catch22's parameters are
*part of the feature definition* — the "5" in `mode_5` is the bin count,
the "40" in `ami_timescale` is the lag cap. Changing one would mean the
feature is no longer the published statistic, so **none of the §4
functions take parameters**, and none appear in the bundle's policy
constants. The only choice we make is *which* features to compute, which
is the feature-set registry, not a parameter.

**3 · We delegate to the reference C implementation.** These are thin
wrappers over `pycatch22`, the Fulcher-lab package, for the same reason
§3 wraps `antropy`: it is the canonical implementation, and several of
these features are fiddly enough (the two-regime crossover fit behind
`rs_range` / `dfa`, the exponential fit in `embedding_dist`, Wang's peak
conditions in `periodicity`) that a reimplementation would risk silently
computing something else. `pycatch22` is an **optional extra** —
`pip install "myocard-egm-features[catch22]"` — so the base library keeps
its lean numpy/scipy/pandas/antropy footprint.

#### 4.1.2 Naming — the hctsa code is the source of truth

catch22 features have two names: the original `hctsa` code
(`SP_Summaries_welch_rect_centroid`) and a short name (`centroid_freq`).
We use short names as our column names, and we maintain **our own**
mapping from hctsa code to short name.

**That is not paranoia — the library's own short-name list is wrong for
two features.** `pycatch22.catch22_all(..., short_names=True)` returns a
list in which `centroid_freq` and `low_freq_power` are **crossed**
relative to the hctsa codes they are paired with. Checked against signals
where the answer is unarguable ($T = 192$, $f_s = 1000$ Hz):

| signal | `..._welch_rect_area_5_1` | `..._welch_rect_centroid` |
|---|---|---|
| 3-cycle sine (very low frequency) | 0.998 | 0.037 |
| 200-cycle sine (very high frequency) | 0.000 | 2.454 |
| 50 Hz sine | 0.992 | **0.3191** |

The 50 Hz row settles it: $2\pi \cdot 50 / 1000 = 0.3142$ rad/sample, so
`..._centroid` is returning the **median frequency in rad/sample**, and
`..._area_5_1` is the **fraction of power in the lowest 20% of
frequencies** — the reverse of the labels the library hands back, and
matching the Fulcher-lab documentation.

Taking those labels at face value would have compared a median frequency
on synthetic against a power fraction on IAFDB under one column name — a
wrong answer with no error raised. So: **we key every lookup off the hctsa
code**, never off `short_names`, and a test pins the mapping. Two further
names we set ourselves rather than inherit: `transition_variance`
(`pycatch22` says `transition_matrix`) and `std_dev` (it says `SD`).

#### 4.1.3 Reliability at T = 192, and the usable-now set

Several catch22 features need a long series to mean anything, and a
192-sample window is short. Two groups are **not used** in the Phase-1.5
comparison feature set, though all 22 are implemented and available —
they are expected to earn their place once the project moves to longer,
multi-beat windows:

- **Self-affine scaling** ([§4.9](#49-self-affine-scaling)) — a
  fluctuation curve needs a wide range of window sizes to fit two regimes
  to; at $T = 192$ there isn't one.
- **The linear-autocorrelation family**
  ([§4.4](#44-linear-autocorrelation-structure)) — lag and period features
  saturate once the timescale they measure approaches the window length. A
  period longer than 192 samples simply cannot be seen.

That leaves the **usable-now 14**: distribution shape (2), extreme-event
timing (2), `trev`, `ami2`, `forecast_error`, `high_fluctuation`,
`whiten_timescale`, the four symbolic features, and `embedding_dist`.

**Two empirical caveats on that set**, measured over 400 traces per length
drawn from one generative process (a biphasic deflection at a random
position plus noise), comparing the across-trace spread at $T = 512$ vs
$T = 192$:

1. Six of the 14 widen by more than 1.5× at 192 — `entropy_pairs` 3.2×,
   `transition_variance` 2.4×, `ami2` 2.1×, `high_fluctuation` 1.9×,
   `embedding_dist` 1.6×. Wider spread is not disqualifying on its own,
   but it costs separation.
2. **`whiten_timescale` looks degenerate on single-activation traces** —
   across 400 traces it returned *one* distinct value at $T = 512$ and
   *four* at $T = 192$. A near-constant coordinate contributes nothing to
   a distribution distance. Treat it as the first candidate to drop.

Both are indicative, not decisive: the measurement used a crude surrogate,
not real or simulated EGM. The feature-responsiveness screening on actual
banks is what settles set membership.

#### 4.1.4 Degenerate input — NaN, and why it must warn

On a **constant** trace, 19 of the 22 features are mathematically
undefined and `pycatch22` returns `NaN` (the three exceptions return 0).
Near-degenerate traces — a flat-lined channel, a dropped electrode, an
all-zero window from a failed export — produce the same thing.

**We propagate the `NaN` rather than substituting a sentinel.** A silent
`0.0` would enter a distribution distance as if it were a real coordinate
and quietly bias the result, whereas a `NaN` column is visible and the
consumer can drop it. Note this is a deliberate difference from §1–§3,
where several features return a finite value in the same situation
(`shannon_entropy` returns `0.0` on an empty signal).

**But silence is its own failure mode.** Batch extraction over a large
bank can run for minutes; discovering afterwards that a whole channel
produced `NaN` wastes all of it, and a `NaN` that reaches a feature
distribution unnoticed is worse than a crash. So the implementation
**must warn when it produces `NaN`** — a single aggregated
`RuntimeWarning` at the end of a batch naming how many traces and which
features were affected, not one warning per trace (which would drown the
output and slow the loop). The intent is that someone extracting a
10,000-trace bank sees the problem early enough to cancel, fix the input,
and not pay for the whole run twice.

The `NaN` values themselves still flow through unchanged — the warning is
diagnostic, never a behaviour change.

#### 4.1.5 The anchor signals

Every worked example below uses one of these six signals, all with
$T = 192$ at $f_s = 1000$ Hz. They are defined here once so the numbers
are reproducible and the tests can rebuild them exactly:

| name | definition |
|---|---|
| `sine50` | $x_i = \sin(2\pi \cdot 50 \cdot i / 1000)$ — period exactly 20 samples |
| `white` | `numpy.random.default_rng(0).standard_normal(192)` |
| `ramp` | $x_i = i$ — a pure linear trend |
| `constant` | $x_i = 1$ — the degenerate case |
| `sawtooth` | $x_i = (i \bmod 64)/64$ — slow rise, sharp drop |
| `biphasic` | $-\,\mathrm{d}/\mathrm{d}i$ of a Gaussian centred at $i = 96$ with $\sigma = 6$, scaled to unit peak — an idealised clean activation |

All values quoted below are computed, not estimated; they are the values
the test suite asserts.

### 4.2 Distribution shape

Both features histogram the $z$-scored trace into $B$ equal-width bins
spanning $[\min \tilde{x}, \max \tilde{x}]$ and return the **centre of the
most populated bin**:

$$
\text{mode}_B(x) = \operatorname{centre}\Big(\arg\max_b\, c_b\Big),
\qquad c_b = \big|\{\, i : \tilde{x}_i \in \text{bin}_b \,\}\big| .
$$

The output is in standard deviations from the mean, because the input is
$z$-scored. They differ only in $B$, and are blind to time ordering — two
traces with the same amplitude histogram in any order score identically.

#### 4.2.1 mode_5

`DN_HistogramMode_5` — $B = 5$, a coarse read.

**What it shows.** Where the *most probable* value sits relative to the
mean, at low resolution. With only five bins spanning the full range, this
is a robust, heavily-smoothed skew descriptor: it answers "which fifth of
the amplitude range does this trace spend most of its time in."

**Worked anchors.** `sine50` → $1.0905$: a sinusoid's density piles up at
its turning points, not its centre, so the tallest bin sits well off zero.
`white` → $-0.2179$, near zero, wandering with the finite realisation.
`constant` → `NaN`.

**EGM read.** On a $z$-scored EGM window the histogram is dominated by
**baseline** — the trace sits near zero between activations — with a thin
tail during the deflection. So `mode_5` mostly reports the
baseline-versus-deflection **duty cycle**, and its sign reports the
deflection's **polarity**: a predominantly negative bipolar deflection
drags the mean down, pushing the baseline mode positive. Coarse enough to
be stable on noisy IAFDB windows.

#### 4.2.2 mode_10

`DN_HistogramMode_10` — $B = 10$, a finer read of the same histogram.

**What it shows.** The same quantity at twice the resolution, so it can
resolve structure `mode_5` smooths over — a secondary density peak, or a
baseline that is itself split.

**Worked anchors.** `sine50` → $1.2324$ (vs $1.0905$ at $B=5$ — the finer
grid localises the turning-point pile-up better). `white` → $0.4654$,
where `mode_5` gave $-0.2179$: on unstructured data the mode is unstable
and the two bin counts need not agree.

**EGM read.** Same interpretation as `mode_5`, with more sensitivity to
fine amplitude structure and correspondingly more noise. Its real value is
**relative to `mode_5`**, below.

**Reading the family together.** The pair is more informative than either
alone. When `mode_5` and `mode_10` **agree**, the amplitude histogram has
one clear, broad peak — a clean baseline-plus-deflection trace. When they
**disagree** (as on `white`), the density has no stable mode: either
genuinely multimodal, or noise-dominated with no structure to find. That
disagreement is a cheap unimodality probe no single feature gives you. Both
pair naturally with `shannon_entropy` (§3.2), which bins the *same*
histogram but returns its **spread** rather than its **location** — mode
says where the mass is, entropy says how concentrated.

### 4.3 Extreme-event timing

Both sweep a threshold $\theta$ (upward from 0 for the positive version,
downward for the negative), and at each $\theta$ take the **median index**
of the samples beyond it, rescaled so the window's middle maps to 0, its
end to $+1$, and its start to $-1$:

$$
r(\theta) = \frac{\operatorname{med}\{\, i : \tilde{x}_i \ge \theta \,\}}{T/2} - 1,
\qquad
\text{outlier\_timing} = \operatorname*{med}_{\theta} r(\theta) \in [-1, 1].
$$

Unlike everything in §4.2, these are explicitly **positional** — they ask
*when*, not *what*.

#### 4.3.1 outlier_timing_pos

`DN_OutlierInclude_p_001_mdrmd` — thresholds swept upward, so it tracks
the **positive** excursions.

**What it shows.** Where in time the large *upward* deviations live.
$\approx 0$: spread evenly through the window. Negative: clustered early.
Positive: clustered late.

**Worked anchors.** `sine50` → exactly $0.0$ — extremes recur throughout,
so the median index is dead centre. `ramp` → $+0.7500$: the largest values
are all at the end, which is the sign convention made concrete. `white` →
$-0.0833$.

**EGM read.** For a single-activation window this is essentially **where
the positive limb of the deflection sits** — closely related to
`activation_position` (§1.3), but derived from the amplitude distribution
rather than from $\mathrm{d}V/\mathrm{d}t$, so it degrades differently on
fractionated signal. On a continuous IAFDB window a non-zero value flags
positive-going activity **drifting** through the window.

#### 4.3.2 outlier_timing_neg

`DN_OutlierInclude_n_001_mdrmd` — thresholds swept downward, tracking the
**negative** excursions.

**What it shows.** The mirror statistic: where the large *downward*
deviations live, on the same $[-1, 1]$ scale.

**Worked anchors.** `sine50` → exactly $0.0$. `ramp` → $-0.7396$: the
smallest values are all at the start. `white` → $+0.1198$.

**EGM read.** Where the negative limb sits. On a bipolar activation the
negative limb is usually the sharp one, so this is often the better
positional marker of the two.

**Reading the family together.** The pair is a **stationarity and
asymmetry probe**, and the interesting information is in how the two
compare:

- **Both $\approx 0$** — activity is stationary and centred. This is what
  an activation-centred synthetic trace should look like, and a useful
  sanity check that the splitter did its job.
- **Both the same sign** — the whole window's activity is drifting toward
  one end. On real IAFDB this is the signature of a fibrillatory burst
  starting or stopping mid-window, or a catheter making or losing contact.
  A segmentation-quality flag more than a substrate marker.
- **Opposite signs** (as on `ramp`, $+0.75$ / $-0.74$) — the positive and
  negative excursions sit at *different* times, which is what a monotone
  trend or a single asymmetric deflection produces. The **gap** between
  them is a crude read of the deflection's internal timing.

Together they cover what a single centre-of-mass measure cannot: `sine50`
and a trace with one early positive and one late negative spike both have
their overall energy centred, but only the latter separates the pair.

### 4.4 Linear autocorrelation structure

> **Deferred family** — implemented and available, but **not** in the
> Phase-1.5 comparison set. Every one of these measures a lag, a period,
> or a spectral summary that saturates when the timescale approaches the
> window length, and $T = 192$ is short. They are expected to become the
> most valuable family once the project moves to multi-beat windows. See
> [§4.1.3](#413-reliability-at-t--192-and-the-usable-now-set).

Five read the autocorrelation function of the $z$-scored trace,

$$
\rho(\tau) = \frac{\sum_{i=1}^{T-\tau}(x_i - \mu)(x_{i+\tau} - \mu)}{\sum_{i=1}^{T}(x_i - \mu)^2},
\qquad \rho(0) = 1,
$$

and two read its Fourier partner, the Welch power spectrum $\hat{S}(f)$ —
the same information in two coordinate systems, which is why they share a
family.

#### 4.4.1 acf_timescale

`CO_f1ecac` — the smallest $\tau$ at which $\rho(\tau) \le 1/e \approx
0.3679$, linearly interpolated to a real-valued lag.

**What it shows.** The **memory length** in samples: how far ahead the
signal stays substantially correlated with itself. An AR(1) process with
coefficient $\phi$ has $\rho(\tau) = \phi^\tau$ and so crosses at
$-1/\ln\phi$.

**Worked anchors.** `white` → $0.6717$ (no memory, as it should be);
`sine50` → $3.8047$; `ramp` → $41.7737$, since a trend stays correlated
for a long time.

**EGM read.** How **oscillatory versus noise-like** the local signal is.
Organised, rhythmic activity holds correlation; fractionated or
noise-dominated signal loses it within a few samples. At $T = 192$ the
measurable range is capped well below an atrial cycle length, which is
exactly why the family is deferred.

#### 4.4.2 acf_first_min

`CO_FirstMin_ac` — the smallest $\tau$ with
$\rho(\tau-1) > \rho(\tau) < \rho(\tau+1)$.

**What it shows.** For an oscillation, the first ACF trough sits at
**half the dominant period**, so this is a direct period estimate. Where
`acf_timescale` reads the decay rate, this reads the first
anti-correlation.

**Worked anchors.** `sine50` → exactly $10.0$, against a period of 20
samples — $P/2$, precisely as the theory predicts, and the cleanest
verification anchor in this family. `white` → $1.0$.

**EGM read.** The most direct route to **atrial cycle length** in the
catch22 set — on a multi-beat window it would report half the
activation-to-activation interval. On a 192-sample single-activation
window there is no second beat to correlate against, so the value is
structurally uninformative here.

#### 4.4.3 periodicity

`PD_PeriodicityWang_th0_01` — detrend with a three-knot cubic-regression
spline, then return the lag of the **first ACF peak** meeting Wang's
amplitude and shape conditions; 0 if none qualifies.

**What it shows.** The **dominant repeating period** in samples, made
robust by the detrending and the peak conditions. High for slow, clearly
periodic signals; 0 when nothing qualifies as periodic.

**Worked anchors.** `sine50` → $19.0$, recovering the 20-sample period to
within the peak-condition tolerance. `ramp`, `sawtooth`, and `constant`
all → $0.0$ — nothing passes the conditions.

**EGM read.** A stricter, better-defended rhythm estimate than
`acf_first_min`, because a trace has to *earn* a non-zero value. The 0
return is informative in itself: a hard "no periodicity found here," which
is the expected answer for a single activation and a meaningful one for
disorganised AF.

#### 4.4.4 low_freq_power

`SP_Summaries_welch_rect_area_5_1` — the fraction of Welch spectral power
below $0.2 f_{Ny}$:

$$
\text{low\_freq\_power} = \frac{\sum_{f \le 0.2 f_{Ny}} \hat{S}(f)}{\sum_f \hat{S}(f)} \in [0,1].
$$

**What it shows.** $\to 1$: energy concentrated at low frequencies (a slow
signal); $\to 0$: energy at high frequencies.

**Worked anchors.** `white` → $0.2584$ — a flat spectrum puts ~20% of its
power in the lowest 20% of the band, which is the sanity check on the
definition itself. `sine50` → $0.9920$, since 50 Hz is well below
$0.2 \times 500 = 100$ Hz.

**EGM read.** A coarse **fractionation** proxy from the frequency side: a
smooth activation concentrates power low, while sharp secondary
deflections and fragmentation push energy up. Blunt compared with
`sec_peak_count` (§1.4), but it needs no peak-detection policy to compute.

#### 4.4.5 centroid_freq

`SP_Summaries_welch_rect_centroid` — the Welch spectrum's **median**
frequency, in **radians per sample**: the $f_{\text{med}}$ splitting total
power in half.

**What it shows.** Where the spectral mass sits. High → fast morphology.

**Worked anchors.** `sine50` → $0.3191$ against the exact
$2\pi \cdot 50/1000 = 0.3142$ — this is the anchor that proves the naming
correction in [§4.1.2](#412-naming--the-hctsa-code-is-the-source-of-truth).
`white` → $1.3990$, near the $\pi/2 \approx 1.571$ a flat spectrum implies.

> **Not interchangeable with our `spectral_centroid` (§2.2).** Ours is the
> power-weighted **mean** in Hz; this is the **median** in rad/sample. On a
> skewed EGM spectrum they differ substantially. Two different statistics
> that happen to share a word.

**EGM read.** The catch22 analogue of dominant-frequency mapping's rate
axis. Because it is a *median*, it is markedly more robust than our mean
`spectral_centroid` to a single high-frequency artefact — which on noisy
real recordings is a meaningful advantage.

#### 4.4.6 ami_timescale

`IN_AutoMutualInfoStats_40_gaussian_fmmi` — the first minimum of the
automutual-information function under a Gaussian estimator, capped at lag
40. Under that assumption $I(\tau) = -\tfrac{1}{2}\ln(1-\rho(\tau)^2)$, a
nonlinear transform of the ACF; the first minimum is the classic
Fraser–Swinney choice of **time-delay for phase-space embedding**.

**What it shows.** A mildly nonlinear autocorrelation timescale. High =
long memory; low = noise-like.

**Worked anchors.** `white` → $2.0$; `sine50` → $4.0$; `sawtooth` →
$16.0$.

**EGM read.** Same rate-and-organisation axis as `acf_timescale`, but via
an information-theoretic route that survives monotone nonlinearities in
the recording chain. The lag-40 cap is $40$ ms at 1 kHz — comfortably
shorter than an atrial cycle, another reason this family wants longer
windows.

**Reading the family together.** These six trilaterate **rate and
organisation**, the axis clinical AF analysis cares most about, from three
independent directions — ACF decay (`acf_timescale`, `ami_timescale`), ACF
structure (`acf_first_min`, `periodicity`), and the spectrum
(`low_freq_power`, `centroid_freq`). The combinations carry the signal:

- **All three routes agree** on a timescale → a genuinely organised,
  rhythmic segment. `acf_first_min` $\approx$ half of `periodicity`, and
  `centroid_freq` consistent with both, is the signature.
- **ACF says periodic, spectrum says broadband** → the rhythm is present
  but buried in fractionated high-frequency content — plausibly the most
  substrate-relevant combination in the family.
- **`periodicity` returns 0 while `acf_timescale` stays high** → the signal
  has memory but no repeating period: drift or a single slow event, not a
  rhythm.

The linear/nonlinear pair (`acf_timescale` vs `ami_timescale`) adds a
fourth read: a large gap between them implies dependence a linear ACF
cannot see, which is where §4.5 picks up.

### 4.5 Nonlinear autocorrelation

Two features capturing dependence structure a *linear* ACF misses. Both
are cheap, both are in the usable-now set, and both survive short windows
well — they aggregate over every sample pair rather than estimating a
timescale.

#### 4.5.1 trev

`CO_trev_1_num` — the mean cube of successive differences:

$$
\text{trev} = \frac{1}{T-1}\sum_{i=1}^{T-1}\big(\tilde{x}_{i+1} - \tilde{x}_i\big)^3 .
$$

**What it shows.** A **time-irreversibility** probe. Cubing preserves sign
and amplifies large steps, so the statistic is $\approx 0$ when up-steps
and down-steps have mirror-image distributions, **positive** when the
sharp moves are rises, and **negative** when they are drops. Linear
Gaussian processes are time-reversible, so a non-zero value signals
nonlinearity.

**Worked anchors.** `sawtooth` → $-0.4119$: it ramps up gently over 64
samples and drops in one, so the rare huge negative step dominates the
cube — the sign convention demonstrated on a signal built for it.
`sine50` → $-0.0004$ (symmetric, so $\approx 0$). `biphasic` → $+0.0196$.

**EGM read.** One of the most EGM-appropriate features in catch22. A
bipolar activation is **morphologically asymmetric** — a fast steep limb
and a slower recovery — and `trev` reads exactly that upstroke/downstroke
asymmetry, with the sign naming which limb is sharper. Fractionated
activations distort the asymmetry differently from clean biphasic ones,
and unlike `sec_peak_count` (§1.4) it needs no threshold policy to say so.

#### 4.5.2 ami2

`CO_HistogramAMI_even_2_5` — mutual information between the trace and
itself at lag 2, from a 2-D histogram with 5 equal-width bins per axis:

$$
\text{ami2} = \sum_a \sum_b p(a,b)\,\log \frac{p(a,b)}{p(a)\,p(b)} .
$$

**What it shows.** How much the value now tells you about the value two
samples later, **including nonlinear** structure that $\rho(2)$ would
miss. $\approx 0$ means independence.

**Worked anchors.** `white` → $0.0675$ (independent, as it should be);
`sine50` → $0.7448$ (deterministic short-lag structure); `ramp` →
$1.4392$.

**EGM read.** Short-range **nonlinear predictability**. Organised
activation carries structured short-lag dependence; disorganised
fibrotic signal tends lower. Conceptually adjacent to `sample_entropy`
(§3.1) — both probe predictability — but by an information-theoretic,
fixed-lag route, and at $O(T)$ rather than `sample_entropy`'s $O(T^2)$.

**Reading the family together.** The two are near-orthogonal and most
useful read as a pair, because they answer different questions about the
same 2-sample neighbourhood: `ami2` asks **how much** structure is there,
`trev` asks **what shape** it has.

- **High `ami2`, `trev` $\approx 0$** — strongly predictable and
  symmetric: a smooth, clean, organised waveform.
- **High `ami2`, `trev` far from 0** — predictable but asymmetric: a
  structured activation with a distinct sharp limb, which is what a
  healthy bipolar deflection should look like.
- **Low `ami2`, `trev` $\approx 0$** — noise.

That last case matters practically: it is the combination that says a
window carries no usable morphology at all, and neither feature says it
alone.

### 4.6 Simple forecasting

One feature, and the only one in catch22 that frames the trace as a
**prediction** problem.

#### 4.6.1 forecast_error

`FC_LocalSimple_mean3_stderr` — predict each sample from the mean of the
previous three and return the standard deviation of the residuals:

$$
\hat{x}_t = \tfrac{1}{3}\big(\tilde{x}_{t-1} + \tilde{x}_{t-2} + \tilde{x}_{t-3}\big),
\qquad
\text{forecast\_error} = \operatorname{std}\big(\tilde{x}_t - \hat{x}_t\big).
$$

**What it shows.** Predictability at a 3-sample horizon by a trivial
local-mean model. Because the input is $z$-scored ($\sigma = 1$), a
forecaster doing anything useful yields residual std $< 1$; $\ge 1$ means
the 3-point mean is worse than useless.

**Worked anchors.** `white` → $1.1331$ — above 1, and correctly so: on
unpredictable data the local mean actively hurts. `biphasic` → $0.4007$
and `sine50` → $0.6108$, both smooth enough to be locally predictable.
`ramp` → $0.0000$: a straight line is *exactly* forecast by a local mean.

**EGM read.** A **smoothness** proxy at the sampling scale. A clean
activation is well predicted by a local mean; a fragmented or noisy one is
not. Note it is **sample-rate dependent** — "3 samples" is 3 ms at 1 kHz —
so values are only comparable at fixed $f_s$, the constraint
[Preprocessing assumptions](#preprocessing-assumptions) already imposes.

**Reading it alongside the others.** `forecast_error` is a scalar summary
of the same predictability `ami2` (§4.5.2) and `sample_entropy` (§3.1)
probe, and the three disagree informatively. `forecast_error` is
**linear and local** (three neighbouring samples); `ami2` is **nonlinear
and fixed-lag**; `sample_entropy` is **nonlinear and pattern-matching
across the whole trace**. A trace that is hard to forecast but has high
`ami2` carries nonlinear structure a local mean cannot exploit — which is
precisely the profile of a sharp, well-formed deflection, and distinguishes
it from noise, which scores poorly on both.

### 4.7 Incremental differences

Both read the one-step differences
$\Delta \tilde{x}_i = \tilde{x}_i - \tilde{x}_{i-1}$, and between them they
separate two things that are easy to conflate: **how much** the trace
moves, and **how much of its correlation is slow drift**.

#### 4.7.1 high_fluctuation

`MD_hrv_classic_pnn40` — the proportion of successive differences
exceeding $0.04\sigma$; on a $z$-scored series, simply $0.04$:

$$
\text{high\_fluctuation} = \frac{1}{T-1}\big|\{\, i : |\tilde{x}_{i+1} - \tilde{x}_i| > 0.04 \,\}\big| .
$$

This is **pNN40**, borrowed from heart-rate-variability analysis.

**What it shows.** $\to 0$: the series has long near-flat stretches.
$\to 1$: it moves at nearly every step.

**Worked anchors.** `biphasic` → $0.2094$ — one sharp deflection on a flat
baseline, so ~79% of steps are essentially still. `white` → $0.9791$ and
`sine50` → $1.0000$ (every step moves).

**EGM read.** A **duty-cycle** measure: how much of the window is actively
deflecting versus quiet baseline. A single activation on a clean baseline
reads low; sustained fragmented activity, or continuous AF, reads high.
Complements `sec_peak_count` (§1.4), which counts *deflections* where this
measures *occupancy* — a trace with one long messy activation and a trace
with three crisp ones can share a peak count but not a duty cycle.

#### 4.7.2 whiten_timescale

`FC_LocalSimple_mean1_tauresrat` — the ratio of the first ACF
zero-crossing of the *differenced* series to that of the original:

$$
\text{whiten\_timescale} = \frac{\tau_0(\Delta \tilde{x})}{\tau_0(\tilde{x})},
\qquad \tau_0(\cdot) = \min\{\tau : \rho(\tau) \le 0\}.
$$

**What it shows.** How much a single differencing step **whitens** the
signal. Small → differencing destroyed a strong slow correlation, so the
trace was trend-dominated. Near 1 → it was already white-ish.

**Worked anchors.** `sawtooth` → $0.0588$: mostly slow ramp, and
differencing annihilates it. `sine50` → $0.8333$; `biphasic` → $0.7778$.

**EGM read.** How much of the window's autocorrelation is slow **baseline
wander** rather than morphology — a signal-quality axis rather than a
clinical one, and a candidate flag for windows whose apparent structure is
really drift.

> ⚠️ See [§4.1.3](#413-reliability-at-t--192-and-the-usable-now-set): on
> repeated single-activation traces this feature was **near-constant**, so
> it may carry no usable information at this window length despite being
> in the usable-now 14.

**Reading the family together.** The pair separates activity from drift,
which neither does alone:

- **High `high_fluctuation`, `whiten_timescale` near 1** — genuinely
  active, broadband morphology on a stable baseline. The fragmented-EGM
  profile.
- **High `high_fluctuation`, low `whiten_timescale`** — the trace moves a
  lot, but its correlation is dominated by slow wander. Suspect baseline
  drift or a contact problem rather than substrate.
- **Low `high_fluctuation`, low `whiten_timescale`** — quiet and
  drifting: a near-flat channel that is not actually recording much.

The middle case is the one worth having: `high_fluctuation` alone would
call it interesting, and it usually isn't.

### 4.8 Symbolic

These discretise the trace first, then compute statistics on the symbol
sequence. Quantisation discards fine amplitude detail but is **robust to
noise**, which makes this family unusually well suited to the low-SNR real
side of a synthetic-versus-real comparison.

#### 4.8.1 stretch_high

`SB_BinaryStats_mean_longstretch1` — binarise $b_i = 1$ if
$\tilde{x}_i > \mu$, and return the length of the longest run of 1s:

$$
\text{stretch\_high} = \max\{\, \ell : b_j = \dots = b_{j+\ell-1} = 1 \,\}.
$$

**What it shows.** The longest uninterrupted excursion above the mean, in
samples. For a sinusoid of period $P$ this is $\approx P/2$.

**Worked anchors.** `sine50` → exactly $10.0$ against a 20-sample period.
`white` → $8.0$. `biphasic` → $95.0$ — worth pausing on: the long quiet
baseline sits *above* the mean, because the single large downward
deflection drags the mean below the baseline. "Above the mean" is not
"active."

**EGM read.** Dwell time on one side of the mean. On a trace with one
dominant deflection it inverts into a **baseline-length** measure (as the
`biphasic` anchor shows), which is genuinely useful — it is close to a
measure of how much of the window is *not* activation.

#### 4.8.2 stretch_decreasing

`SB_BinaryStats_diff_longstretch0` — binarise the *differences*
($b_i = 1$ if $\tilde{x}_i > \tilde{x}_{i-1}$) and return the longest run
of 0s: the longest monotone decrease.

**What it shows.** The longest uninterrupted downward run, in samples — a
coarse read of the slowest downstroke in the trace.

**Worked anchors.** `biphasic` → $72.0$, the long smooth decay of the
Gaussian tail. `white` → $5.0$; `sine50` → $11.0$.

**EGM read.** A **slope-duration** proxy, and the closest catch22 comes to
the clinical notion of activation *width*. A fragmented activation is
interrupted by secondary deflections, breaking long monotone runs, so this
shortens as fractionation increases — the same phenomenon
`sec_peak_count` counts, measured as a duration instead.

#### 4.8.3 entropy_pairs

`SB_MotifThree_quantile_hh` — map each sample to one of three symbols by
**equiprobable tertiles**, form all consecutive two-letter words, and
return the Shannon entropy of their distribution:

$$
\text{entropy\_pairs} = -\sum_{s \in \{A,B,C\}^2} p(s) \log p(s)
\;\in\; [0,\ \ln 9 \approx 2.197].
$$

**What it shows.** Predictability of two-step symbolic transitions. Low →
a few pairs dominate (structured); high → all nine near-equiprobable
(unpredictable at the pair scale).

**Worked anchors.** `white` → $2.1803$, close to the $\ln 9$ ceiling —
exactly right for an unpredictable series, and a good verification that
the tertile mapping is equiprobable as specified. `sine50` → $1.6506$;
`biphasic` → $1.2062$ (structured, well below the ceiling).

**EGM read.** A noise-robust **disorder** measure. Because tertiles are
equiprobable by construction, it is invariant to the amplitude
distribution and reads pure sequencing — a fragmented fibrotic EGM should
score higher than an organised activation. The nearest §3 cousin is
`shannon_entropy` (§3.2), but that bins amplitudes and ignores order,
where this ignores amplitude and reads order.

#### 4.8.4 transition_variance

`SB_TransitionMatrix_3ac_sumdiagcov` — tertile-symbolise into 3 states,
set $\tau$ to the first ACF zero-crossing, build the $3\times3$ $\tau$-step
transition-probability matrix $P$, and return the sum of its column
variances:

$$
\text{transition\_variance} = \sum_{k=1}^{3} \operatorname{Var}_j\big(P_{jk}\big).
$$

**What it shows.** How **specific** the state transitions are, evaluated at
the timescale where linear autocorrelation has faded. Near-uniform
transitions (noise) → low; sharply determined transition rules → high.

**Worked anchors.** `white` → $0.0028$ — near-uniform, no rule. `ramp` →
$0.1667$ $(= 1/6)$, the maximally deterministic case. `biphasic` →
$0.0703$.

**EGM read.** Whether the coarse dynamics follow a **rule**. Organised
activation moves through baseline → upstroke → recovery in a determined
order; disorganised signal does not. Because $\tau$ is chosen adaptively
from the ACF, it asks the question at whatever timescale the trace itself
makes relevant.

**Reading the family together.** The four split cleanly into two **shape**
features and two **disorder** features, and the cross-comparisons are
where the family earns its place:

- **`entropy_pairs` high *and* `transition_variance` low** — disordered at
  both the pair scale and the rule scale. The clearest symbolic signature
  of a fragmented, disorganised window, and the pair the fibrosis
  hypothesis predicts.
- **`entropy_pairs` high but `transition_variance` high too** — locally
  varied yet rule-governed. Structured complexity rather than noise; a
  fast but organised rhythm can look like this.
- **`stretch_high` long *and* `stretch_decreasing` short** — a long quiet
  baseline broken by a sharp, interrupted deflection: the fractionation
  profile in shape terms.
- **Both stretches long** — a slow, smooth, single-event trace.

Because all four survive heavy quantisation, they travel onto noisy real
recordings better than the amplitude-sensitive features in §1, which is
the practical argument for keeping the whole family in the comparison set.

### 4.9 Self-affine scaling

> **Deferred pair** — implemented but **not** in the Phase-1.5 comparison
> set; a fluctuation curve needs a wider range of window sizes than 192
> samples affords. Expected to become usable on multi-beat windows. See
> [§4.1.3](#413-reliability-at-t--192-and-the-usable-now-set).

Both ask whether the signal has self-affine (fractal, long-range
correlated) scaling. Form the cumulative profile
$Y_j = \sum_{i \le j} \tilde{x}_i$, split it into windows of size $s$,
remove a local trend in each, measure the residual fluctuation $F(s)$, and
look for a power law $F(s) \propto s^{H}$ — a straight line in
$\log F$ versus $\log s$. They differ only in the detrending step.

> **⚠️ The output is not the Hurst exponent.** catch22 fits **two**
> scaling regimes and returns a value in $[0,1]$ encoding **where the
> crossover between them sits** (`prop_r1` = the proportion of the
> timescale range in the low-scale regime). Low → the scaling law changes
> at short timescales; high → one law persists to long ones.

#### 4.9.1 rs_range

`SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1` — **rescaled-range**
detrending: remove the line joining each window's endpoints and take the
**range** (max − min) of the residuals.

**What it shows.** The crossover position of the rescaled-range
fluctuation curve. Because it uses a range, it is driven by the extremes
within each window.

**Worked anchors.** `white` → $0.4762$; `sine50` → $0.3810$; `sawtooth` →
$0.8333$.

**EGM read.** Roughness reached through extremes, so it is comparatively
sensitive to isolated sharp deflections — the events fibrosis work cares
about, but also the events an artefact produces.

#### 4.9.2 dfa

`SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1` — **detrended fluctuation
analysis**: fit a linear polynomial per window and take the **RMS** of the
residuals, after downsampling the series by 2.

**What it shows.** The same crossover position via an RMS rather than a
range, making it the more stable of the two. DFA is a workhorse in HRV
analysis and has established AF applications.

**Worked anchors.** `white` → $0.8333$; `sine50` → $0.1429$.

**EGM read.** Roughness reached through typical deviations rather than
extremes, so it is the better-behaved substrate descriptor of the pair and
the one to reach for first if only one is affordable.

**Reading the family together.** The two share a method and differ only in
detrending, which makes their **agreement a built-in reliability check** —
unusual, and useful given the family's known instability:

- **`rs_range` and `dfa` agree** — there is genuine scaling structure, and
  the crossover estimate can be trusted.
- **They disagree sharply** (as on `white`: $0.4762$ vs $0.8333$) — the
  fluctuation curve has no clean two-regime fit, and neither value should
  be read as a substrate property. On our 192-sample windows this is the
  expected outcome, which is the concrete reason the pair is deferred.

Both are conceptual cousins of `higuchi_fractal_dimension` (§3.4) — the
same roughness intuition — but reached by fluctuation scaling rather than
curve-length scaling, and reported as a crossover position rather than a
dimension. They are **not** a drop-in substitute for it, and the fact that
§3.4 remains stable at $T = 192$ while these two do not is the reason we
still ship it.

### 4.10 Other

One feature that fits none of the other families — a phase-space
descriptor.

#### 4.10.1 embedding_dist

`CO_Embed2_Dist_tau_d_expfit_meandiff` — build a 2-D time-delay embedding
with $\tau$ = the first ACF zero-crossing, take successive Euclidean step
distances, fit an exponential to their distribution, and return the mean
absolute error of that fit:

$$
\mathbf{v}_t = (\tilde{x}_t,\ \tilde{x}_{t+\tau}), \qquad
d_t = \lVert \mathbf{v}_{t+1} - \mathbf{v}_t \rVert, \qquad
\text{embedding\_dist} = \operatorname{MAE}\big(\mathrm{expfit}(P(d))\big).
$$

Time-delay embedding is the standard Takens attractor reconstruction.

**What it shows.** How closely the reconstructed trajectory's step
distances follow an **exponential** law. Low → near-exponential, so
stochastic-looking. Higher → structured, deterministic geometry.

**Worked anchors.** `white` → $0.0821$ (stochastic, near-exponential);
`biphasic` → $0.7540$ and `sine50` → $0.6411$ (both strongly structured).

**EGM read.** A nonlinear-dynamics descriptor separating stochastic-looking
from structured morphology. More exploratory for EGM than the amplitude
and fractionation features, but cheap once $\tau$ is known, and it is the
only feature here that reads trajectory *geometry* rather than a time or
amplitude statistic. `sample_entropy` (§3.1) shares the delay-embedding
idea but returns a regularity measure rather than an embedding-geometry
fit, so the two are complementary: `sample_entropy` asks whether patterns
repeat, `embedding_dist` asks what shape the trajectory traces.

### 4.11 The catch24 additions

"catch24" is catch22 plus the two statistics $z$-scoring removes. They are
computed on the **raw** trace $x$, not $\tilde{x}$ — the only features in
§4 that see amplitude at all.

#### 4.11.1 mean

`DN_Mean` — the arithmetic mean of the raw trace:

$$
\text{mean} = \frac{1}{T}\sum_i x_i .
$$

**What it shows.** The DC level catch22 discards.

**Worked anchors.** `ramp` → $95.5$, exactly $(0 + 191)/2$. `constant` →
$1.0$ — defined where 19 of the other 22 return `NaN`.

**EGM read.** Near-zero by construction on a bandpassed EGM, so on our
data it is mostly a **check that the bandpass did its job**: a materially
non-zero mean indicates DC offset or baseline wander that should have been
filtered out.

#### 4.11.2 std_dev

`DN_Spread_Std` — the population standard deviation of the raw trace:

$$
\text{std\_dev} = \sqrt{\frac{1}{T}\sum_i (x_i - \mu)^2} .
$$

**What it shows.** The amplitude scale catch22 discards.

**Worked anchors.** `ramp` → $55.5698$. `constant` → $0.0$ — the
degenerate case, and the reason 19 other features return `NaN` on it:
division by this zero.

**EGM read.** An amplitude measure, and therefore the one catch24 feature
that touches the voltage axis fibrosis mapping is built on. But it is
**RMS-like, not peak-to-peak**: it is driven by the whole window,
including the long quiet baseline, so a short sharp activation on a quiet
baseline has a small `std_dev` despite a large deflection.
`peak_to_peak` (§1.1) is the clinically-calibrated statistic and the one
whose thresholds the literature defines.

**Reading the pair together, and why they are not in our default sets.**
Together they restore exactly what $z$-scoring removed, so a caller who
wants canonical catch24 can reconstruct it. But `egm-features` already
carries the amplitude channel in a better form: `peak_to_peak` (§1.1) is
the statistic the voltage-mapping literature actually thresholds, and
`mean` is near-zero by construction post-bandpass. The informative
comparison is **`std_dev` against `peak_to_peak`**: their *ratio* is a
crude duty-cycle read — a trace whose peak-to-peak is large while its
`std_dev` stays small is one brief deflection on a quiet baseline, whereas
a ratio near 1 means the window is deflecting throughout. That ratio is
worth more than either catch24 feature alone, which is why both are
registered but neither is in a default set.

## 5. References

> Links are given as resolvable DOIs where one could be confirmed, and as a
> publisher or PubMed URL otherwise. Three older references
> (Inouye 1991, Esteller 2001, Aboy 2006) are listed without a link because
> their DOIs could not be verified at the time of writing — better no link
> than a wrong one.

### Core feature papers

- **Marchlinski FE, Callans DJ, Gottlieb CD, Zado E.** *Linear ablation
  lesions for control of unmappable ventricular tachycardia in patients
  with ischemic and nonischemic cardiomyopathy.* Circulation 2000;101:1288.
  [doi:10.1161/01.CIR.101.11.1288](https://doi.org/10.1161/01.CIR.101.11.1288)
  — bipolar voltage threshold convention, used for `peak_to_peak` (§1.1).
- **Sanders P et al.** *Spectral analysis identifies sites of
  high-frequency activity maintaining atrial fibrillation in humans.*
  Circulation 2005;112:789.
  [doi:10.1161/CIRCULATIONAHA.104.517011](https://doi.org/10.1161/CIRCULATIONAHA.104.517011)
  — atrial dominant-frequency analysis (§2.4).
- **Kosiuk J et al.** *Validation of voltage mapping during atrial
  fibrillation: the 0.2 mV threshold.*
  [PMID 30873619](https://pubmed.ncbi.nlm.nih.gov/30873619/)
  — AF-adjusted voltage thresholds.
- **Nademanee K et al.** *A new approach for catheter ablation of atrial
  fibrillation: mapping of the electrophysiologic substrate.* J Am Coll
  Cardiol 2004;43:2044.
  [doi:10.1016/j.jacc.2004.03.032](https://doi.org/10.1016/j.jacc.2004.03.032)
  — the CFAE definition, background for `sec_peak_count` (§1.4).

### Entropy + complexity

- **Shannon CE.** *A Mathematical Theory of Communication.* Bell System
  Technical Journal 1948;27:379.
  [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
  — the original entropy, underneath `shannon_entropy` (§3.2) and
  `entropy_pairs` (§4.8.3).
- **Pincus SM.** *Approximate entropy as a measure of system complexity.*
  PNAS 1991;88:2297.
  [doi:10.1073/pnas.88.6.2297](https://doi.org/10.1073/pnas.88.6.2297)
  — ApEn, predecessor of SampEn; the source of our $m$ and $r$ defaults.
- **Richman JS, Moorman JR.** *Physiological time-series analysis using
  approximate entropy and sample entropy.* Am J Physiol Heart Circ Physiol
  2000;278:H2039.
  [doi:10.1152/ajpheart.2000.278.6.H2039](https://doi.org/10.1152/ajpheart.2000.278.6.H2039)
  — sample entropy (§3.1).
- **Inouye T et al.** *Quantification of EEG irregularity by use of the
  entropy of the power spectrum.* Electroencephalogr Clin Neurophysiol
  1991;79:204. *(No verified DOI.)* — spectral entropy in biomedical
  signals (§2.3).

### Fractal + LZ

- **Higuchi T.** *Approach to an irregular time series on the basis of the
  fractal theory.* Physica D 1988;31:277.
  [doi:10.1016/0167-2789(88)90081-4](https://doi.org/10.1016/0167-2789(88)90081-4)
  — the Higuchi fractal-dimension algorithm (§3.4).
- **Lempel A, Ziv J.** *On the complexity of finite sequences.* IEEE Trans
  Inf Theory 1976;22:75.
  [doi:10.1109/TIT.1976.1055501](https://doi.org/10.1109/TIT.1976.1055501)
  — LZ76, underneath `lempel_ziv_complexity` (§3.3).
- **Aboy M, Hornero R, Abásolo D, Álvarez D.** *Interpretation of the
  Lempel-Ziv complexity measure in the context of biomedical signal
  analysis.* IEEE Trans Biomed Eng 2006;53:2282. *(No verified DOI.)* — LZ
  applied to physiological signals.
- **Esteller R, Vachtsevanos G, Echauz J, Litt B.** *A comparison of
  waveform fractal dimension algorithms.* IEEE Trans Circuits Syst I
  2001;48:177. *(No verified DOI.)* — Higuchi vs Katz vs Petrosian.

### catch22 (§4)

- **Lubba CH, Sethi SS, Knaute P, Schultz SR, Fulcher BD, Jones NS.**
  *catch22: CAnonical Time-series CHaracteristics.* Data Mining and
  Knowledge Discovery 2019;33:1821.
  [doi:10.1007/s10618-019-00647-x](https://doi.org/10.1007/s10618-019-00647-x)
  · [arXiv:1901.10200](https://arxiv.org/abs/1901.10200)
  — the feature-selection method and the 22-feature set. All of §4.
- **Fulcher BD, Jones NS.** *hctsa: A Computational Framework for Automated
  Time-Series Phenotyping Using Massive Feature Extraction.* Cell Systems
  2017;5:527.
  [doi:10.1016/j.cels.2017.10.001](https://doi.org/10.1016/j.cels.2017.10.001)
  — the ~7700-feature library catch22 distils.
- **Fulcher BD, Little MA, Jones NS.** *Highly comparative time-series
  analysis: the empirical structure of time series and their methods.*
  J R Soc Interface 2013;10:20130048.
  [doi:10.1098/rsif.2013.0048](https://doi.org/10.1098/rsif.2013.0048)
- **Fulcher-lab feature documentation** —
  [time-series-features.gitbook.io/catch22](https://time-series-features.gitbook.io/catch22/information-about-catch22/feature-descriptions/feature-overview-table)
  — per-feature descriptions and the hctsa↔catch22 name mapping; the
  authority [§4.1.2](#412-naming--the-hctsa-code-is-the-source-of-truth)'s
  naming correction is checked against.
- **Welch PD.** *The use of fast Fourier transform for the estimation of
  power spectra.* IEEE Trans Audio Electroacoust 1967;15:70.
  [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
  — the spectrum behind `low_freq_power` (§4.4.4) and `centroid_freq`
  (§4.4.5).
- **Fraser AM, Swinney HL.** *Independent coordinates for strange
  attractors from mutual information.* Phys Rev A 1986;33:1134.
  [doi:10.1103/PhysRevA.33.1134](https://doi.org/10.1103/PhysRevA.33.1134)
  — first-AMI-minimum embedding delay: `ami_timescale` (§4.4.6), `ami2`
  (§4.5.2).
- **Takens F.** *Detecting strange attractors in turbulence.* Lecture Notes
  in Mathematics 1981;898:366.
  [doi:10.1007/BFb0091924](https://doi.org/10.1007/BFb0091924)
  — time-delay embedding: `embedding_dist` (§4.10.1).
- **Schreiber T, Schmitz A.** *Surrogate time series.* Physica D
  2000;142:346.
  [doi:10.1016/S0167-2789(00)00043-9](https://doi.org/10.1016/S0167-2789(00)00043-9)
  — time-reversibility: `trev` (§4.5.1).
- **Mietus JE, Peng C-K, Henry I, Goldsmith RL, Goldberger AL.** *The pNNx
  files: re-examining a widely used heart rate variability measure.* Heart
  2002;88:378.
  [PMC1767394](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1767394/)
  — pNN40: `high_fluctuation` (§4.7.1).
- **Wang X, Wirth A, Wang L.** *Structure-based Statistical Features and
  Multivariate Time Series Clustering.* Proc IEEE ICDM 2007:351.
  [ieeexplore 4470259](https://ieeexplore.ieee.org/document/4470259/)
  — `periodicity` (§4.4.3).
- **Peng C-K, Buldyrev SV, Havlin S, Simons M, Stanley HE, Goldberger AL.**
  *Mosaic organization of DNA nucleotides.* Phys Rev E 1994;49:1685.
  [doi:10.1103/PhysRevE.49.1685](https://doi.org/10.1103/PhysRevE.49.1685)
  — detrended fluctuation analysis: `dfa` (§4.9.2).
- **Caccia DC, Percival D, Cannon MJ, Raymond G, Bassingthwaighte JB.**
  *Analyzing exact fractal time series: evaluating dispersional analysis
  and rescaled range methods.* Physica A 1997;246:609.
  [doi:10.1016/S0378-4371(97)00363-4](https://doi.org/10.1016/S0378-4371(97)00363-4)
  — `rs_range` (§4.9.1).
- `intracardiac-platform/project/investigations/catch22_techniques_explained.md`
  — the project's full walk-through of all 22, at greater depth than §4,
  with the per-feature EGM-relevance discussion this section condenses.

### Project context

- **Sánchez J et al.** *Using Machine Learning to Characterize Atrial
  Fibrotic Substrate From Intracardiac Signals With a Hybrid in silico and
  in vivo Dataset.* Frontiers in Physiology 2021;12:699291.
  [full text](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.699291/full)
  — the paper this feature set most closely tracks. Sánchez used a
  7-feature subset (peak-to-peak, duration, sample entropy, Shannon
  entropy, spectral entropy, Kolmogorov complexity, fractal dimension) and
  trained a decision-tree classifier. Our §1–§3 eleven are those seven plus
  `zero_crossings`, `activation_position`, `sec_peak_count`, and
  `dominant_frequency`, added during the v1_baseline diagnostic; §4 adds
  the catch22 set on top.
- `intracardiac-platform/project/architecture_reading_list.md` — the
  broader literature this project draws on.
