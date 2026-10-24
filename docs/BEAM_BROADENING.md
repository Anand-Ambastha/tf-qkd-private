# Turbulence-induced beam broadening

## 1. Motivation

Gamma-Gamma and Log-Normal describe how the received irradiance fluctuates around its mean. They keep that
mean at the free-space value, so scintillation moves power around but never removes any. Real turbulence also
spreads the time-averaged beam: eddies larger than the beam push it around, and smaller ones diffract it. The
receiver then collects a smaller share of the power on average. At large Cn2 this is the main loss, and a
mean-one fading model cannot see it. The `*_bb` models add it as a change to the nominal transmittance and
leave the fading statistics alone.

## 2. Long-term beam radius

For a collimated Gaussian beam over a horizontal path of length z (Andrews and Phillips, 2005):

    W_LT^2 = W^2 * (1 + 1.63 * sigma_R^(12/5) * Lambda),      Lambda = 2 z / (k W^2)

- W = w(z) is the free-space beam radius at the receiver, given the transmitter waist w0.
- k = 2 pi / lambda.
- sigma_R^2 is the plane-wave Rytov variance of the arm (`legacy_baseline.rytov_variance`), the same number the
  Gamma-Gamma model uses.
- sigma_R^(12/5) denotes (sigma_R^2)^(6/5), the same convention as in the Gamma-Gamma alpha and beta (see
  GAMMA_GAMMA.md).

With Cn2 = 0 the correction vanishes and W_LT = W. It grows with both Cn2 and path length.

## 3. Effect on the transmittance

A circular aperture of radius a captures `eta_cap(W) = 1 - exp(-2 a^2 / W^2)` of a Gaussian beam of radius W.
The loss caused by turbulence on one arm is the ratio

    f_bb(z) = eta_cap(W_LT) / eta_cap(W)        (0 < f_bb <= 1)

and the arm's nominal transmittance becomes `eta_arm_nominal * f_bb(z)`.

## 4. One factor per arm

As in TWO_LINK_FADING.md, the Alice-Charlie and Bob-Charlie arms each have length z = L/2 and their own
sigma_R^2, W_LT and f_bb. The end-to-end nominal transmittance is

    eta_nom_total = eta_nom(L) * f_bb(L/2)^2

and the independent fading of each arm (Gamma-Gamma, Log-Normal or corrected Gamma-Gamma) multiplies this:
eta = (eta_A_nom f_bb I_A) (eta_B_nom f_bb I_B).

## 5. Factory names

| `model=` | Fading on each arm | Added |
|---|---|---|
| `gamma_gamma_bb` | Gamma-Gamma | beam broadening |
| `lognormal_bb` | Log-Normal | beam broadening |
| `gamma_gamma_corrected_bb` | corrected Gamma-Gamma | beam broadening |

They need `w0` and `receiver_radius`, and they only work per arm (`two_link=True`).

## 6. What it does to the range

Broadening is a loss on the mean, so the maximum range falls as Cn2 rises. Without broadening, the
Gamma-Gamma range is nearly independent of Cn2. In `BEAM_BROADENING_REPORT.txt` the
ranges are 287.9, 198.6 and 81.9 km for weak, moderate and strong turbulence with broadening, against 291.7,
293.2 and 291.7 km without it.

## 7. Assumptions and limits

These should be stated when results are quoted.

1. W_LT already contains beam wander, so pointing jitter (`jitter_std`) should not be added in the same run;
   doing so would count the same effect twice.
2. This is a first-order correction. The Gamma-Gamma alpha and beta stay at their plane-wave values, and the
   scintillation of the broadened beam is not derived again.
3. The 1.63 sigma_R^(12/5) Lambda form comes from weak-to-moderate turbulence. For sigma_R^2 well above 1 it
   keeps growing instead of saturating, so the strong-Cn2 broadening (W_LT of about 16 m at the strong-Cn2 maximum range) is likely
   overestimated and the strong-Cn2 range is likely a lower bound. Gamma-Gamma without broadening gives the
   corresponding upper bound.
4. The path is horizontal, Cn2 is constant, and the transmitter is a collimated Gaussian beam.

## Reference

Andrews, L. C. and Phillips, R. L., *Laser Beam Propagation through Random Media*, 2nd ed., SPIE Press (2005), Ch. 6.
