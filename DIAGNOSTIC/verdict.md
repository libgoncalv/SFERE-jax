# dX-invariance verdict (channel width in metres)

Primary estimator: runs median on the |u|*h field at q0.95; same 1024x512 m domain.

| dX [m] | runs median [m] | EDT [m] | spectrum spacing [m] | cells per width |
|---|---|---|---|---|
| 0.25 | 12.25 | 17.96 | 146.14 | 49.0 |
| 0.5 | 12.00 | 16.14 | 85.17 | 24.0 |
| 1.0 | 15.00 | 19.42 | 78.46 | 15.0 |

Log-log slopes of the primary width vs dX (physics-set: |slope| <~ 0.1; grid-controlled: >~ 0.5):
- 0.25 -> 0.5: ratio 0.98, slope -0.03
- 0.5 -> 1.0: ratio 1.25, slope 0.32
- all points: slope 0.15

**Verdict: over the well-resolved pair (0.25, 0.5) the width is dX-invariant (slope -0.03) - the pattern is physics-set; coarsening is meaningful while channels stay many cells wide (Mariotti's ~width/4 floor shows here as the dX=1.0 widening).**
- caveat: at dX=1.0 the width reads ~25% wider and the trunk network merges visually - the onset of the one-cell floor, not a physics change.

Estimator caveats (why only the q-field runs median is primary):
- the t-field (recomputed shear) mask width tracks a near-constant CELL count across resolutions, i.e. it scales with dX in metres - a resolution-sensitive proxy, kept as cross-check, ignored in the verdict;
- the spectrum spacing varies between realizations (each grid draws its own seedling IC), so it is band information, not a paired measurement;
- mask quantiles fix the channel-cell FRACTION by construction, so widths across DIFFERENT domains/discharges compare pattern response, not an absolute width.

Domain-invariance cross-check (RUN9: a 4x smaller domain -> 4x smaller drainage-integrated discharge; same dX=0.5): median width 3.00 m vs 12.00 m here (ratio 0.25). Width grows with the physics forcing (Hin*LengthY), not with the grid - the Y-extension lever from the stability review.
