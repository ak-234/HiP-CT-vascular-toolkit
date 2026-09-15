"""The generated forms are only trustworthy if their argv parses back.

Every widget in the control panel is generated from an argparse action, so the whole
scheme rests on one property: whatever `to_argv` emits, `parse_args` must read back
as the values that produced it. These tests assert that directly, over all nine
subcommands and the viewer's parser, with no Qt and no dataset.

The specific mistake being guarded against is the "unset" case. Twenty-six fields on
the `edit` parser default to None, and their handlers build keyword dicts by skipping
the Nones (`cmd_connect:207-215`), so emitting `--cone-deg=0.0` for an untouched
field would quietly override a library default. `test_defaults_emit_nothing` catches
that for every command at once.
"""

from __future__ import annotations

import inspect

import pytest

from hipct_seg_debug import cliform
from hipct_seg_debug.edit.__main__ import build_parser as edit_parser
from hipct_seg_debug.main import build_parser as viewer_parser

NINE = (
    "report", "gaps", "connect", "flag-interpolation",
    "train-cfc", "evaluate-dpc", "export-dpc-regions",
    "surface", "skeletonise",
    "optimise", "repair-radius", "repair-mask",
    # The Walsh-Berg chain (SKELETONISATION.md), declared together.
    "skeletonise-all", "optimise-skeleton",
    # Diagnostics: they report on a written graph rather than editing one, and are
    # declared beside the pass whose output they explain.
    "segment-diagnosis", "junction-mask", "reformat-radius", "ostium-flare",
    "refine-centreline", "prepare-reconstruction", "radius-perimeter", "crop",
    # Interactive, and declared beside `crop` because it feeds the same `--root-edge`.
    "pick-roots",
    "score",
    "mask-export",
)

SPECS = {s.name: s for s in cliform.describe_parser(edit_parser())}

# A positional has to be supplied before anything can be parsed at all.
# Every command that processes a skeleton now takes several, so the positional is
# always a list of paths.
POSITIONALS = {"graph": ["left.am", "right.am"]}
# ...and `mask-export --out` is the one required optional.
REQUIRED = {
    "mask-export": {"out": "out.am"},
    "train-cfc": {"raw": "raw", "out_model": "model"},
    "evaluate-dpc": {"raw": "raw", "cfc_model": "model"},
    "export-dpc-regions": {"raw": "raw", "output": "review"},
    # The diagnostics report on named segments, so there is no sensible default:
    # asking for "the radius verdict" without saying whose is not a question.
    "segment-diagnosis": {"segment": [265]},
    "junction-mask": {"segment": [265]},
    "reformat-radius": {"segment": [265]},
}


def _minimal(spec) -> dict:
    values = cliform.defaults(spec)
    for f in spec.fields:
        if f.positional:
            values[f.dest] = POSITIONALS[f.dest]
    values.update(REQUIRED.get(spec.name, {}))
    return values


# ------------------------------------------------------------ describe_parser


def test_every_subcommand_is_described_in_order():
    assert tuple(SPECS) == NINE


def test_the_viewer_parser_describes_as_one_unnamed_command():
    specs = cliform.describe_parser(viewer_parser())
    assert len(specs) == 1 and specs[0].name == ""
    assert "raw" in specs[0].dests and "goto_um" in specs[0].dests


def test_help_actions_are_not_fields():
    assert "help" not in SPECS["report"].dests


def test_report_takes_only_the_graph():
    assert SPECS["report"].dests == ("graph",)


def test_the_graph_is_a_positional():
    assert SPECS["gaps"].field("graph").positional


def test_mask_export_out_is_required():
    assert SPECS["mask-export"].field("out").required


def test_choices_survive_in_declaration_order():
    assert SPECS["repair-radius"].field("source").choices == ("outlier", "image", "both")


@pytest.mark.parametrize(
    "command,dest,kind",
    [
        ("connect", "tjunction", cliform.FLAG),
        ("skeletonise", "stride", cliform.INT),
        ("repair-radius", "factor", cliform.FLOAT),
        ("optimise", "reference", cliform.TEXT),
        ("repair-radius", "source", cliform.CHOICE),
    ],
)
def test_action_shapes_map_to_the_right_kind(command, dest, kind):
    assert SPECS[command].field(dest).kind == kind


def test_the_only_multi_value_flag_is_goto_um():
    """If a second one appears, the single-token `--flag=value` rule needs revisiting."""
    spec = cliform.describe_parser(viewer_parser())[0]
    assert [f.dest for f in spec.fields if f.nargs > 1] == ["goto_um"]


# --------------------------------------------------------------- argv round trip


@pytest.mark.parametrize("name", NINE)
def test_defaults_emit_nothing(name):
    """An untouched form must produce the bare command, not every flag at its default.

    This is the test that protects the None-means-omit rule.
    """
    spec = SPECS[name]
    argv = cliform.to_argv(spec, _minimal(spec))
    extra = [tok for tok in argv if tok.startswith("--")]
    required = REQUIRED.get(name, {})
    expected = []
    for field in spec.fields:
        if field.dest in required and not field.positional:
            value = required[field.dest]
            # An `append` field repeats its flag once per value rather than
            # emitting the list, so a required one expands to several tokens.
            if isinstance(value, list):
                expected.extend(f"{field.flag}={v}" for v in value)
            else:
                expected.append(f"{field.flag}={value}")
    assert extra == expected, argv


@pytest.mark.parametrize("name", NINE)
def test_defaults_parse_back_to_the_defaults(name):
    spec = SPECS[name]
    values = _minimal(spec)
    parsed = edit_parser().parse_args(cliform.to_argv(spec, values))
    assert cliform.from_namespace(spec, parsed) == values


@pytest.mark.parametrize("name", NINE)
def test_a_fully_populated_form_round_trips(name):
    """Every field set to something *other* than its default must survive."""
    spec = SPECS[name]
    values = _minimal(spec)
    for f in spec.fields:
        if f.positional or f.dest in REQUIRED.get(name, {}):
            continue
        values[f.dest] = _other_than(f)
    parsed = edit_parser().parse_args(cliform.to_argv(spec, values))
    assert cliform.from_namespace(spec, parsed) == values


def _other_than(f):
    if f.repeatable:
        return [7] if f.kind == cliform.INT else ["value"]
    if f.kind == cliform.FLAG:
        return not f.default
    if f.kind == cliform.CHOICE:
        return next(c for c in f.choices if c != f.default)
    if f.kind == cliform.INT:
        return int(f.default or 0) + 7
    if f.kind == cliform.FLOAT:
        return float(f.default or 0.0) + 1.5
    return f"{f.dest}_value"


@pytest.mark.parametrize("name", NINE)
def test_force_all_also_parses(name):
    """The explicit form is for pasting into a script, so it had better be valid."""
    spec = SPECS[name]
    values = _minimal(spec)
    parsed = edit_parser().parse_args(cliform.to_argv(spec, values, force_all=True))
    assert cliform.from_namespace(spec, parsed) == values


def test_a_float_keeps_its_decimal_representation():
    """A spin box hands back 0.30000000000000004; `str` would round it to 0.3.

    Rounding here would mean the command line the panel echoes does not reproduce
    the run it just performed.
    """
    spec = SPECS["repair-radius"]
    value = 0.1 + 0.2
    argv = cliform.to_argv(spec, {**_minimal(spec), "factor": value})
    assert "--factor=0.30000000000000004" in argv
    assert edit_parser().parse_args(argv).factor == value


def test_a_value_equal_to_the_default_is_still_omitted():
    """0.1 + 0.5 really is 0.6, so `--factor` is unchanged and must not be passed."""
    spec = SPECS["repair-radius"]
    argv = cliform.to_argv(spec, {**_minimal(spec), "factor": 0.1 + 0.5})
    assert not [t for t in argv if t.startswith("--factor")]
    assert edit_parser().parse_args(argv).factor == 0.6


def test_values_are_emitted_as_one_token():
    """Two tokens would make a negative number ambiguous with a flag."""
    spec = SPECS["optimise"]
    argv = cliform.to_argv(spec, {**_minimal(spec), "bb_threshold": -50.0})
    assert "--bb-threshold=-50.0" in argv
    assert edit_parser().parse_args(argv).bb_threshold == -50.0


def test_a_path_with_spaces_survives():
    """One token per path, so a space cannot split a variadic positional in two."""
    spec = SPECS["report"]
    paths = [r"D:\data dir\a b.am", r"C:\x y\c.am"]

    argv = cliform.to_argv(spec, {"graph": paths})

    assert edit_parser().parse_args(argv).graph == paths


def test_a_flag_at_its_default_is_omitted():
    spec = SPECS["connect"]
    assert "--tjunction" not in cliform.to_argv(spec, _minimal(spec))
    assert "--tjunction" in cliform.to_argv(spec, {**_minimal(spec), "tjunction": True})


def test_the_multi_value_flag_round_trips():
    spec = cliform.describe_parser(viewer_parser())[0]
    argv = cliform.to_argv(spec, {**cliform.defaults(spec), "goto_um": [1.0, 2.0, 3.5]})
    assert viewer_parser().parse_args(argv).goto_um == [1.0, 2.0, 3.5]


# ------------------------------------------------------------------ unset rules


def test_the_spin_box_sentinel_reads_as_unset():
    f = SPECS["connect"].field("cone_deg")
    assert cliform.is_unset(f, cliform.UNSET)
    assert not cliform.is_unset(f, 30.0)


def test_an_empty_text_field_reads_as_unset():
    assert cliform.is_unset(SPECS["optimise"].field("reference"), "")


def test_zero_is_not_unset():
    """The whole point: `--cone-deg 0` must reach argparse, `(unset)` must not."""
    spec = SPECS["connect"]
    assert "--cone-deg=0.0" in cliform.to_argv(spec, {**_minimal(spec), "cone_deg": 0.0})
    assert not [t for t in cliform.to_argv(spec, {**_minimal(spec), "cone_deg": None})
                if t.startswith("--cone-deg")]


# ------------------------------------------------------------------ path roles


def test_artifact_producers_write_without_being_asked():
    """A command that spends minutes deriving something must not discard it.

    The dry-run-unless-`--out` convention is for proposers whose printed plan is the
    useful output. `skeletonise-all` decodes the lattice, runs every algorithm and
    scores each -- up to 25 minutes -- so a blank flag costing all of it is not a safe
    default. `surface` is the precedent: it always writes, to a default directory.
    """
    for name, dest in (("surface", "out_dir"), ("skeletonise-all", "out_dir")):
        default = SPECS[name].field(dest).default
        assert default, f"{name} --{dest.replace('_', '-')} must have a writing default"


def test_every_path_flag_has_a_role():
    """A dest that names a file but has no Browse button is the drift this catches."""
    named = {"graph", "seg", "surface", "raw", "out", "out_dir", "edits", "reference",
             "cache", "crop_json"}
    missing = [
        f"{s.name}.{f.dest}"
        for s in [*SPECS.values(), *cliform.describe_parser(viewer_parser())]
        for f in s.fields
        if f.dest in named and f.path_role is None
    ]
    assert missing == []


def test_no_role_names_a_dest_that_does_not_exist():
    dests = {f.dest for s in SPECS.values() for f in s.fields}
    dests |= {f.dest for s in cliform.describe_parser(viewer_parser()) for f in s.fields}
    assert [key for key in cliform.PATH_ROLES if key[1] not in dests] == []


def test_the_command_specific_role_wins_over_the_fallback():
    assert cliform.path_role("repair-mask", "out")[1].startswith("TIFF")
    assert cliform.path_role("gaps", "out")[1].startswith("Amira spatial graph")
    assert "TIFF" in cliform.path_role("mask-export", "out")[1]


def test_the_raw_folder_is_a_directory():
    assert cliform.path_role("", "raw")[0] == "dir"


def test_surface_out_dir_is_a_save_target():
    assert SPECS["surface"].field("out_dir").path_role == "save_dir"


# --------------------------------------------------------------------- errors


def test_a_missing_positional_is_an_error():
    assert "graph is required" in cliform.errors(SPECS["report"], {"graph": None})


def test_mask_export_refuses_without_out():
    values = {**cliform.defaults(SPECS["mask-export"]), "out": None}
    assert "--out is required" in cliform.errors(SPECS["mask-export"], values)


def test_a_nonexistent_input_is_an_error(tmp_path):
    missing = str(tmp_path / "nope.am")
    problems = cliform.errors(SPECS["report"], {"graph": missing})
    assert problems and "does not exist" in problems[0]


def test_an_existing_input_is_fine(tmp_path):
    real = tmp_path / "graph.am"
    real.write_text("x")
    assert cliform.errors(SPECS["report"], {"graph": str(real)}) == []


def test_a_missing_surface_is_not_an_error(tmp_path):
    """`main.py:182-183` continues without one, so the panel must not block on it."""
    spec = cliform.describe_parser(viewer_parser())[0]
    values = {**cliform.defaults(spec), "surface": str(tmp_path / "gone.stl"),
              "raw": str(tmp_path), "graph": str(tmp_path), "seg": str(tmp_path)}
    assert cliform.errors(spec, values) == []


def test_an_output_path_need_not_exist(tmp_path):
    real = tmp_path / "graph.am"
    real.write_text("x")
    values = {**cliform.defaults(SPECS["gaps"]), "graph": str(real),
              "out": str(tmp_path / "new.am")}
    assert cliform.errors(SPECS["gaps"], values) == []


# ------------------------------------------------------------------- rendering


def test_the_command_line_quotes_a_path_with_spaces():
    line = cliform.command_line(
        SPECS["report"], {"graph": r"D:\a b\c.am"}, prog="edit")
    assert r'"D:\a b\c.am"' in line


def test_the_subprocess_command_is_unbuffered():
    """Without -u the child prints nothing until it exits."""
    cmd = cliform.subprocess_command(["report", "g.am"])
    assert cmd[1] == "-u" and cmd[2] == "-m"
    assert cmd[-2:] == ["report", "g.am"]


def test_describe_parser_does_not_import_the_heavy_deps():
    """The panel is unconditional, so building it must not need coronary_sdf."""
    import sys

    for name in [m for m in sys.modules if m.startswith("coronary_sdf")]:
        del sys.modules[name]
    cliform.describe_parser(edit_parser())
    assert not [m for m in sys.modules if m.startswith("coronary_sdf")]


def test_every_optimise_skeleton_flag_reaches_a_real_parameter():
    """`--sweep` sets parameters by flag name, so the two namings must not drift.

    They did: `--sweep "prune-factor=1,2,3"` -- the example in `SKELETONISATION.md` --
    raised `TypeError: unexpected keyword argument 'prune_factor'` because the sweep
    passed the flag name through while `_optimise_kwargs` translated it. That surfaced
    only after the volume had been decoded and the scoring context built, minutes in.
    """
    from hipct_seg_debug.edit import skeleton_optimise as so
    from hipct_seg_debug.edit.__main__ import (
        _OPTIMISE_DIRECT,
        OPTIMISE_ALIASES,
        optimise_parameter,
    )

    known = set(inspect.signature(so.optimise_skeleton).parameters)
    for flag in (*OPTIMISE_ALIASES, *_OPTIMISE_DIRECT):
        assert optimise_parameter(flag) in known, flag

    # ...and the flags really exist on the parser, so a rename on either side is caught.
    dests = {a.dest for a in _subparser("optimise-skeleton")._actions}
    for flag in (*OPTIMISE_ALIASES, *_OPTIMISE_DIRECT):
        assert flag in dests, flag


def test_a_misspelled_sweep_term_is_refused_before_the_expensive_part():
    from hipct_seg_debug.edit.__main__ import _check_sweep_names

    _check_sweep_names({"prune_factor": 2.0, "recentre_blob": "blob4"})
    with pytest.raises(ValueError, match="names nothing that optimise-skeleton sets"):
        _check_sweep_names({"recentre_tangent": 2.0})


def _subparser(name):
    parser = edit_parser()
    return parser._subparsers._group_actions[0].choices[name]


def test_radius_perimeter_branch_aware_controls_parse():
    args = edit_parser().parse_args([
        "radius-perimeter", "graph.am", "--seg", "labels.am",
        "--no-branch-aware", "--root-edge", "7", "--root-edge", "19",
        "--tangent-search-deg", "15", "--carina-tip-factor", "0.08",
    ])
    assert args.branch_aware is False
    assert args.root_edge == [7, 19]
    assert args.tangent_search_deg == 15.0
    assert args.carina_tip_factor == 0.08
