# Geometry L1b migration notice

Geometry v1.1 deliberately rejects tapered or frustum permanent magnets. The
accepted magnetics v1 handoff can represent only rectangular meridional
bounds, and silently approximating or dropping an authoritative PM is not
acceptable.

Before L1b removes this restriction, its material-aware axisymmetric solver
contract must provide:

1. exact polygonal or linear-frustum PM material regions;
2. uniform magnetization over those regions without replacing their material
   ID or recoil law;
3. equivalent bound-current construction on sloped inner/outer surfaces with
   correct unit normals and surface measures;
4. recoil-remanence and equivalent-current authority exclusivity using the
   serialized representation plan;
5. endpoint-exact minimum dielectric/thermal clearance over every overlapping
   axial segment;
6. equality between authoritative geometry PM count and L1b PM
   region/source count;
7. reciprocal interface adjacency for segmented sloped boundaries;
8. convergence evidence comparing exact L1b geometry against any optional
   staircase approximation.

Migration acceptance tests must include:

- a taper with `dr/dz=1/6` and outer normal approximately
  `(0.9863939,-0.1643990)`;
- a minimum-clearance violation occurring only at one taper endpoint;
- five geometry PM stages producing exactly five solver PM regions or five
  equivalent-current sources;
- rejection of a handoff that omits or duplicates one PM;
- identical canonical plan/material IDs across geometry and solver artifacts.

Until these gates pass, use rectangular PMs for authoritative solves. The
existing L1a current-equivalent preview remains explicitly non-authoritative
and is not an L1b migration substitute.
