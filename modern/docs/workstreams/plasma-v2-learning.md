# plasma-v2 learning ledger

Lessons from building the sheath-closed four-cell model (2026-09-03). Each
item is a fact checked against the record or the tests, not an impression.

1. **A density-free sheath row cannot identify a potential.** The floating
   condition `(1/4) n v_e_bar exp(-Dphi/T) = n u_B` cancels the density, so
   R28-R30 fix `Dphi_s,k` and nothing else; the rank of the corrected core
   stays 21 on the potentials. The review's expectation that "sheath rows
   restore the count" is right only for the cusp potentials Kornfeld
   cancelled, not for the cell potentials. Check the rank block by block
   before believing a closure identifies anything.
2. **The anode sheath is the one identifying row that costs no new
   inputs.** The ratio of the electron Boltzmann flux to the Bohm ion flux
   cancels density and area; with the cascade's `j_i4 = Ia - j_e4` it pins
   the anode ion fraction (1/K0 = 0.51 % at zero fall) and therefore
   `phi_1`. It reproduces Kornfeld's 14.1 V plume potential at 14.07 V.
3. **Gluing a Maxwellian sheath onto a monoenergetic cascade needs an
   explicit consistency margin.** `dE_k - Dphi_s,k >= 0` has the closed form
   `c_s <= (1 + I_k/Je_k)/CT`, independent of the potential step; xenon
   without emission (5.27) fails below 447 V of gain, the space-charge limit
   (1.02) never fails. Derive the criterion, then run the grid; the grid
   then confirms the criterion (0/96 vs 89/96) instead of surprising you.
4. **Bracket scans must treat "undefined" as a signed limit.** The anode row
   tends to +inf at the edge of the admissible cascade (`j_e4 -> Ia+`), and
   the root can sit in a sliver of 0.07 V just inside it; a scan that resets
   on undefined points misses every such root (first attempt: `no_bracket`
   at the Kornfeld point while the LM found the root from a crude seed).
5. **Rank is a manifold property.** At a least-squares state 1e-12 off the
   manifold the identity row R27 reads one rank higher; evaluate structural
   rank at the exact manifold projection with the state's own potentials.
6. **Published "solutions" of this system are minimum-error points.** Kornfeld
   prints powers that sum to 1005.9 W; his cusp loss matches the v2 (no +EI)
   convention, his excitation loss matches neither; Puca's GA columns violate
   the perveance and interface rows at the 0.5 level once `j_e0` is
   reconstructed from their own rows. Compare row by row, never by "looks
   close".
7. **Feedback of the exit drop on the cascade makes mode B fragile.** Solving
   the anode fall from a declared cathode coupling has two roots below the
   mode-A coupling and only an inadmissible one above it (fall > T4). Report
   the admissible band, do not average over it.
8. **Declared densities dominate CL-4.** With PIC segment means the Goebel
   leak current is 10-100x the discharge current at the PIC point; the
   prefactor sweep is moot once `p_k` saturates. A cusp sheath-edge density
   (PIC density sinks at the cusps) is the input that matters.
9. **Flat interior is a declaration, not an observation, for this PIC.** The
   plateau shows 94 / 55 / 125 V steps; Koch 2011 / Brandt 2016 describe a
   different device and regime. Keep `CL-3-potentials` labelled as declared
   and carry a PIC-declared variant beside it.
10. **Windows tooling.** The Write tool produced LF this session, but every
    new text file was re-checked for `\r` before staging; the Springer table
    page needed a headless Chrome DOM dump (Cloudflare challenge blocks
    plain HTTP); `pdftotext -layout` on the IEPC PDF gave Table 3.1 verbatim.
