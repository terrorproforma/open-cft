# Geometry source and assumption ledger

## Repository sources

1. `FYP/FEMMrun.m`
   - Defines an axisymmetric millimetre FEMM problem.
   - Fixed values: `a=2`, `f=4`, `g=5`, `h=15`, `k=16`, `l=20`, `p=21`.
   - Uses SmCo 27 MGOe, pure iron, aluminium 6061-T6, air, and a BN ceramic
     placeholder.
   - Applies alternating magnetization angles 90°, 270°, 90° to three magnet
     bands.
   - Commented examples `b=3`, `d=9`, and `e=11` are not fixed inputs.

2. `FYP/CFTOpt.m`
   - Labels geometry decision variables IMR, OMR/OMD, inner shield radius,
     outer shield radius, and outer shell radius.
   - Gives independent 2–50 mm numerical bounds. The file does not enforce
     nested radial ordering, so these are optimizer bounds rather than a
     manufacturable geometry specification.

3. `FYP/params.m`
   - Confirms the geometry variable labels IMR, OMD, ISR, OSR, and OER.

4. `modern/spec/magnetics/material-source-ledger-v1.json`
   - Supplies the accepted permanent-magnet authority rule, synthetic SmCo-like
     recoil parameters, axis regularity, and equivalent bound-current
     equations used by adapters.

5. `paper/references.bib`
   - Contains the project bibliography entries `Kornfeld2007`,
     `Muffatti2017`, `Fahey2017`, and `Yeo2020`.
   - The geometry code does not extract dimensions from those bibliographic
     records because the repository contains citations, not redistributable
     full text or dimension tables.

## Authored assumptions

- historical baseline nominal radial gap: 0.40 mm, with a 0.25 mm thermal
  requirement plus two-sided radial tolerance;
- historical yoke outer radius: 13 mm;
- regular pole fill between magnet spans;
- compact and divergent variant dimensions in their entirety;
- linear soft-iron relative permeability of 4000 for handoff screening;
- representative densities: SmCo 8300, BN 2100, iron 7870, aluminium 2700,
  and copper 8960 kg/m³;
- cathode and neutralizer are external and excluded from revolution;
- nominal cusp count is stage count minus one.

These assumptions are machine-labelled in configuration evidence/material
records. They are not measurements, optimization results, or build
recommendations.
