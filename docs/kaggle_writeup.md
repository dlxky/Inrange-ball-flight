# Kaggle Writeup draft

> **How to use this file.** The table below fills in the Writeup form. Everything under "Project
> Description" is pasted into the Project Description box. Before submitting:
> - Replace `[NOTEBOOK LINK]` with the public Kaggle notebook URL.
> - Upload each image or GIF in the Kaggle editor where its `![...](figures/...)` placeholder sits.
> - Rewrite section 5's AI-use paragraph so it matches how you actually worked.
> - Check the reference details.

## Form fields

| Field | Value |
|---|---|
| Title (60/80) | Seeing past the net: calibrated physics + Gaussian processes |
| Writeup URL | `seeing-past-the-net` |
| Subtitle (113/140) | Spin, apex and landing from 60 m of radar, with every prediction turned into an animated flight, bounce and roll. |
| Card and thumbnail image | `figures/thumbnail.png` (560 × 280) |
| Media gallery | `figures/test_shot.gif`, `figures/test_shots_mix.gif`, `figures/13_case_studies.png`, `figures/11_test_flights.png` |
| Project links | the public Kaggle notebook |
| Files | `submissions/submission.csv` |

---

## Project Description

### Inrange Student Competition Submission

## 1. Submission detail

- **Code:** [NOTEBOOK LINK]. The notebook documents the whole pipeline. It writes every source module,
  calibrates the flight model, runs honest cross-validation, writes `submission.csv`, and renders every
  figure and animation in this writeup.
- **Predictions:** my `submission.csv`, with predictions for `test.csv`, is attached to this writeup.
- **External site:** none. Everything is in this writeup and the notebook.

**Results at a glance.** Honest 5-fold cross-validation on `train.csv`, in which every label-dependent
step (calibration, session wind, spin model, Gaussian process) is refitted inside each fold:

| Landing | Apex | Apex time | Landing time | Spin |
|---|---|---|---|---|
| **4.24 m** (median 3.1 m) | 2.09 m | 0.068 s | 0.118 s | 718 rpm |

A 3-D flight model is calibrated on the training shots and inverted on each shot's four checkpoints;
its prediction becomes a feature for a Gaussian process. That is about 10% better on landing than the
best pure machine-learning model and 30% better than physics alone. Every prediction is then turned
back into a physically consistent flight (launch → net → apex → landing → bounce → roll) and animated.

## 2. Data

**The range geometry can be recovered exactly** (Figure 1). The downrange heading is 28.37° from +x.
The four checkpoint lines are *fixed* planes on the range, at 10.86 + 15k m along that heading, not
distances from each bay. Shots come from four bays, one on an upper deck 4.1 m up. `launch_x/y/z` is
each bay's nominal spot: fitting the checkpoints back to t = 0 shows the ball effectively starts 0.66 m
(about 13 ms) behind it, in every bay and every session. I model each shot in a launch-relative frame
(downrange, lateral, height above the tee) and rotate back at the end; `landing_z` is the tee height.

![Figure 1](figures/01_range_geometry.png)
*Figure 1. Range layout and training landings, coloured by launch spin.*

**The net hides most of the flight** (Figure 2). On average it sees 41% of the carry and 30% of the
flight time; 96% of shots are still climbing when they reach it and 92% peak beyond it. Apex and
landing are therefore extrapolations, not interpolations.

![Figure 2](figures/02_what_the_net_hides.png)
*Figure 2. Reconstructed flights for 28 random training shots: solid up to the net, pale beyond it.*

**Spin leaves a fingerprint in the first 60 m** (Figure 3). Fitting a smooth cubic through the four
checkpoints, anchored at the radar launch velocity, then subtracting gravity and dividing by dynamic
pressure, gives an effective lift and drag coefficient for the visible part of the flight. These
correlate with launch-monitor spin at r = 0.84 and r = 0.82, which is why spin, never measured by the
radar, is recoverable at all.

![Figure 3](figures/04_spin_fingerprint.png)
*Figure 3. Effective lift and drag between the checkpoints against measured launch spin.*

**Shots are not independent** (Figure 4). Players hit runs with one club: consecutive shots' spin
correlates at r = 0.80, and each hitting block shares its weather. I use the second effect through a
physically modelled session wind. I deliberately do *not* copy spin from neighbouring training shots:
it improved spin but worsened positions, and it would be memorising the split rather than modelling the
ball.

![Figure 4](figures/05_same_club_runs.png)
*Figure 4. Spin through one session; test shots are interleaved with training shots.*

## 3. Modelling

### 3.1 Flight model

The ball is a point mass (45.93 g, 42.67 mm) under gravity, drag ½ρAC_D v², Magnus lift ½ρAC_L v²
along ω̂ × v, and spin decay, with air velocity taken relative to a uniform wind for each hitting
block. The spin axis is perpendicular to the launch velocity, tilted to curve the ball. With spin
factor S = rω/v, C_D = c₀ + c₁·S and C_L = C_max·S/(S + S½). All shots are integrated together with
vectorised RK4 (dt = 0.01 s); checkpoint crossings, apex (v_z = 0) and landing are interpolated inside
the step, so thousands of trajectories cost a fraction of a second.

### 3.2 Calibration

`train.csv` includes launch-monitor spin, so the aerodynamics can be calibrated with spin known. One
sparse least-squares problem fits the global coefficients, per-shot nuisances (spin-axis tilt, start
offsets) and a wind vector for each of the 19 hitting blocks against every measured point: four
checkpoints, apex and landing.

The result (Figure 5) is C_D = 0.185 + 0.235·S and C_L = 0.324·S/(S + 0.122); lift saturates at high
spin and calibrated spin decay is negligible. Freeing a drag and lift multiplier per shot fits landing
to 0.85 m RMS, so the model form is right and the residual scatter is ball to ball. The block winds
(Figure 6) share a constant −3.5 m/s offset along the range, which absorbs drag the model does not
capture; the spread between blocks is the weather.

![Figure 5](figures/09_aerodynamics.png)
*Figure 5. Calibrated drag and lift against per-shot fits.*

![Figure 6](figures/10_session_wind.png)
*Figure 6. Effective wind per hitting block.*

### 3.3 From checkpoints to a prediction

For a test shot only the launch and the four checkpoints are known.

1. **Spin.** A LightGBM model on the fingerprint features predicts spin (718 rpm mean error).
2. **Inversion.** A batched Levenberg–Marquardt fit recovers spin-axis tilt and start offsets from the
   checkpoints, with priors from the calibration.
3. **Flight.** The calibrated model flies the shot to apex and landing using its block's wind.
4. **Hybrid layer.** For each target, a Gaussian process with an ARD RBF kernel takes 12 launch and
   checkpoint features plus that target's physics prediction, and learns how far to trust the physics
   for each kind of shot. Physics as a *feature* beat fitting the GP to physics residuals (4.88 m) and
   beat LightGBM with physics features (4.66 m).

### 3.4 Results and ablations

| Approach (5-fold CV) | Landing (m) | Apex (m) | Apex t (s) | Landing t (s) | Spin (rpm) |
|---|---|---|---|---|---|
| Train mean | 42.04 | 31.23 | 0.465 | 0.857 | 2036 |
| Ridge, degree 2 | 5.14 | 2.88 | 0.090 | 0.145 | 1019 |
| LightGBM | 5.37 | 3.11 | 0.084 | 0.138 | 718 |
| Gaussian process | 4.69 | 2.35 | 0.077 | 0.130 | 824 |
| Physics only | 6.06 | 3.30 | 0.094 | 0.201 | 718 |
| **Hybrid (final)** | **4.24** | **2.09** | **0.068** | **0.118** | **718** |

Ablating the final model one ingredient at a time (same honest CV): without session wind 4.51 m;
without the physics feature 4.69 m; without the fingerprint features 4.46 m, with spin error rising
from 718 to 1,009 rpm. Feeding the model the *true* launch spin does not help at all: 4.34 m, no better
than the 4.24 m it achieves with an estimate.

![Figure 7](figures/06_model_comparison.png)
*Figure 7. Landing error by approach and by ablation.*

Downrange and lateral landing errors average 3.0 m and 2.4 m; the 90th-percentile landing error is
8.2 m (Figure 8). Error grows with launch speed, from about 1.5 m at 41 m/s to 6 m at 70 m/s, and is
largest for low-spin shots, which carry furthest beyond the net (Figure 9).

![Figure 8](figures/07_predicted_vs_actual.png)
*Figure 8. Out-of-fold predictions against measurements, and the error distribution.*

![Figure 9](figures/08_error_by_shot_type.png)
*Figure 9. Landing error by launch spin and launch speed.*

### 3.5 Error analysis: what, why and when

**What a miss looks like** (Figure 10). The best-predicted shot lands within 0.1 m despite its spin
being underestimated by 2,375 rpm. The worst misses by 29.8 m: its spin is underestimated by 2,581 rpm
*and* the ball flew with 9% less drag and 10% less lift than the calibrated average, so the model
predicts a flatter, longer flight. The worst decile curves twice as much as the rest (12.6° against
6.1° of spin-axis tilt), has 1.7× more unusual aerodynamics, and carries 128 m past the net against
95 m — yet its spin error is *smaller* (489 against 558 rpm).

![Figure 10](figures/13_case_studies.png)
*Figure 10. Measured and predicted flights for the best, a typical, the 95th-percentile and the worst
out-of-fold shot.*

**Why shots miss** (Figure 11). I regressed the log relative landing error (error ÷ carry) on seven
standardised shot-level factors and also measured each factor's rank correlation, with bootstrap 95%
intervals (n = 491, R² = 0.31). The largest adjusted effects are how unlike the training set the shot
is (+38% per standard deviation), carry beyond the net (+24%), curvature (+16%) and an unusual ball
(+13%). Higher launches (−12%) and the upper-deck bay (−19%) are relatively more accurate. The spin
misestimate has no detectable effect (+2%, interval −6% to +10%), which agrees with the oracle-spin
ablation and with physics-only tests. **The limit is per-ball aerodynamic variation and geometric
extrapolation, not spin.**

![Figure 11](figures/14_error_attribution.png)
*Figure 11. Univariate and adjusted associations with relative landing error.*

**When shots miss** (Figure 12). Relative error differs between sessions (Kruskal–Wallis p < 0.001) and
still does after adjusting for shot type (p = 0.005), so a single wind vector per block does not capture
everything about a day. Two apparent timing effects turn out to be confounded: error falls slightly
through the morning (ρ = −0.15, p < 0.001) and grows through a hitting block (ρ = +0.22, p < 0.001), but
both vanish once shot type is accounted for (ρ = −0.07, p = 0.10; ρ = −0.01, p = 0.86). They were
simply club progression, players moving from wedges to drivers. Finally, residuals of consecutive shots
in a block are independent (permutation p = 0.33 downrange, 0.76 lateral), which indicates the session
wind already absorbs what neighbouring shots share.

![Figure 12](figures/15_error_timing.png)
*Figure 12. Relative error by session, time of day and position in the block, and carry-over between
consecutive shots against a within-block permutation null.*

### 3.6 Why this will work on test.csv

- **Test looks like train** (Figure 13): a LightGBM classifier separates them with AUC 0.56. Both come
  from the same 11 sessions, interleaved in time.
- **The cross-validation is honest:** every label-dependent step is refitted inside each fold.
- **It survives a brand-new day:** holding out whole sessions, where no wind was ever fitted for that
  day, the model gets 5.24 m against 5.53 m for the GP alone. An unseen session uses the mean wind;
  setting it to zero instead would bias the physics (6.77 m).
- **Physics keeps it grounded:** where test shots exceed the training range, predictions still obey drag
  and lift rather than a pure curve fit.

![Figure 13](figures/03_train_vs_test.png)
*Figure 13. Launch conditions, train against test.*

**What is new here:** recovering the fixed checkpoint planes and the hidden start offset from the data;
effective lift and drag measured between checkpoints as machine-learning features; one joint calibration
of aerodynamics, per-shot nuisances and session wind; physics as a GP feature rather than a residual
base; displayed flights that obey the physics *and* match the submitted predictions; and an error
analysis that separates ball-to-ball aerodynamics from spin, geometry and timing.

**Limitations.** A few test shots launch more steeply (up to 37°) or hook harder than any training
shot, so those are extrapolations, exactly the regime the attribution flags as worst. The aerodynamic
anomaly used in that attribution is measured *after the fact* from the full flight: it is a diagnostic,
not a predictor. A new range would need its own wind calibration or a weather feed.

## 4. Trajectory

### 4.1 The full flight

The submission holds point predictions, but a display needs a whole flight that obeys the physics. For
each test shot I **reconcile** the two: a per-shot drag and lift multiplier, spin-axis tilt and start
offsets are fitted so the simulated flight passes through the measured checkpoints *and* the submitted
apex and landing. Across the 559 test shots the flights pass a median 0.95 m from the predicted landing
and 1.2 m from the apex, with multipliers between 0.66 and 1.28, so every flight stays physically
plausible.

Each animation has three linked views (3-D, side, top-down). Colour switches from *tracked* (blue, up to
the net) to *modelled* (orange) to *bounce and roll* (green); the upper-deck bay and net are drawn to
scale; the ball casts a shadow; and time, speed, height and distance update live while apex, carry and
rest are labelled as they happen.

![Figure 14](figures/11_test_flights.png)
*Figure 14. Predicted landings for all 559 test shots, and 40 complete predicted flights.*

### 4.2 Bounce and roll

Impact follows the turf-crater model of Penner (2002): the turf gives way and tilts the effective
contact normal back against the ball's travel by an angle that grows with impact speed and steepness;
the coefficient of restitution depends on normal impact speed; and the tangential impulse is capped by
Coulomb friction or ends in rolling, with backspin carried through the impact so the ball can check or
spin back. Between impacts the ball hops ballistically with drag, then rolls to rest under rolling
resistance plus grass drag. The turf constants are generic, not measured at this range. The behaviour
looks right (Figure 15): a 2,500 rpm shot releases 35 m, a 6,900 rpm iron stops within 2 m, and a
10,300 rpm wedge spins back 0.8 m.

![Figure 15](figures/12_bounce_and_roll.png)
*Figure 15. Bounce and roll for four test shots across the spin range.*

## 5. AI use, reproducibility and references

**How I used AI.** 
I used Claude Code, Anthropic's coding assistant, on a standard subscription, as a pair programmer: exploring the data,
implementing the simulator, calibration, and models and figures. I directed the
approach, reviewed results and chose the final model.

**Reproducing.** Python 3.12 with numpy, pandas, scipy, scikit-learn, lightgbm and matplotlib; no
external data, all libraries open source. Easiest route: run the notebook top to bottom. From the
repository root, with `export PYTHONPATH=src`: `python -m flight.calibration`,
`python -m experiments.physics_studies aerodynamics`, `python -m modelling.pipeline --cv --submit`,
`python -m experiments.ml_studies error_analysis`, `python -m report.figures eda`,
`python -m report.figures results`, `python -m report.animation`.

**References.**
1. A. R. Penner, "The run of a golf ball", *Canadian Journal of Physics* 80 (2002) 931–940.
2. C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006.
3. G. Ke et al., "LightGBM: a highly efficient gradient boosting decision tree", *NeurIPS 30*, 2017.
4. K. Levenberg (1944) and D. W. Marquardt (1963), the least-squares algorithm used for the per-shot fits.
