# SCAPIS attribution

`ccl_template.py` and `cfx_def_export.py` reuse code from
`ak-234/SCAPIS-SCAPIS_GUI_VERSION`, commit
`24f36a58b9e7a0adea605a39db6d768cbf600ee4` (24f36a5).
Original files: `windkessel_BC.py`, `cfx_def_export.py`, and the routines under
`cfx_user_fortran`. Original project: Akash Sivananthan; original Windkessel
coupling routines: Kaihong Wang. Reuse was requested by the repository owner.
No new licence is asserted for the upstream source.

Adaptations: explicit outlet RCR inputs; explicit distal and capacitor pressures;
1050 kg/m³ case density; eight-argument WK interface; unpadded Outlet names;
80-character name buffers; frozen lagged flow; end-step commit and checkpoint;
mesh mapping checks; source-matched library preflight; precise floating-point
serialization; repaired zone-name parsing; failed CFX-Pre exit codes remain
failures even when a partial `.def` exists. Original GUI and diameter-based
parameter generation are not used by the exporter.

The Fortran implementation uses the original CFX user-stack/junction-box API
pattern but replaces its pressure-history recurrence with an explicit capacitor
state. It has not been compiled or solver-qualified in this environment.
