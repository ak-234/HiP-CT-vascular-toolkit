# CFX-Pre remesh session template

The immutable `.cfx` physics seed is binary. CFX-Pre must therefore perform the
mesh reload so that named locations are mapped while its Quemada model, inlet
flow, numerics, convergence controls and initialization remain unchanged.

Record one session in CFX-Pre 2025 R2 against a disposable seed copy:

1. Open the seed, choose **File > Reload Mesh Files**, and select a representative
   Simpleware Fluent mesh using millimetres and **Preserve Existing Names**.
2. Import the generated boundary-only CCL with **Replace** for the named outlet
   and opening objects. Do not replace the domain, inlet, wall, solver control or
   initialization.
3. write a solver input `.def`, then export/save the session as
   `cfx_remesh_session_template.pre` beside this file.
4. Replace the literal paths in that recorded ASCII session with these tokens:
   `{{MESH_MSH}}`, `{{BOUNDARY_CCL}}`, `{{OUTPUT_DEF}}`, and (if present)
   `{{SEED_CFX}}`. `{{CASE_ID}}` is also available for object/file labels.

The launcher refuses CFX execution if the template is absent or contains an
unresolved token. This one recorded operation is necessary because the CFX case
format does not expose a supported text API for replacing mesh topology; the
official supported automation is playback of a CFX-Pre session file.
