"""Tests for the door/window extraction layer of parse_spatiallm_output.

Format observed from SpatialLM 1.1 (Qwen 0.5B) raw output:
  door_N=Door(wall_X, x, y, z, width, height)
  window_N=Window(wall_X, x, y, z, width, height)

The wall_id is the first arg — that's the only field we need for
has_door / has_window. Documented in
docs/superpowers/qa/2026-04-29-plan3b-b0-spatiallm-format.md.
"""
import numpy as np

from pipeline.spatiallm_layout import parse_spatiallm_output


def _coord_offset():
    """Identity coord offset for tests (Z-up <-> Y-up doesn't matter here)."""
    return np.array([0.0, 0.0, 0.0])


def test_parser_sets_has_door_on_referenced_wall():
    text = (
        "wall_0=Wall(0.0,0.0,0.0,3.0,0.0,0.0,2.4,0.0)\n"
        "wall_1=Wall(3.0,0.0,0.0,3.0,3.0,0.0,2.4,0.0)\n"
        "door_0=Door(wall_1,3.0,1.5,1.0,0.9,2.1)\n"
    )
    layout, _objects = parse_spatiallm_output(text, _coord_offset())
    walls = layout["walls"]
    assert walls[0]["has_door"] is False
    assert walls[1]["has_door"] is True


def test_parser_sets_has_window_on_referenced_wall():
    text = (
        "wall_0=Wall(0.0,0.0,0.0,3.0,0.0,0.0,2.4,0.0)\n"
        "window_0=Window(wall_0,1.5,0.0,1.5,1.2,0.8)\n"
    )
    layout, _ = parse_spatiallm_output(text, _coord_offset())
    assert layout["walls"][0]["has_window"] is True


def test_parser_handles_multiple_doors_on_different_walls():
    text = (
        "wall_0=Wall(0.0,0.0,0.0,3.0,0.0,0.0,2.4,0.0)\n"
        "wall_1=Wall(3.0,0.0,0.0,3.0,3.0,0.0,2.4,0.0)\n"
        "wall_2=Wall(0.0,0.0,0.0,0.0,3.0,0.0,2.4,0.0)\n"
        "door_0=Door(wall_0,1.0,0.0,1.0,0.9,2.1)\n"
        "door_1=Door(wall_2,0.0,2.0,1.0,0.9,2.1)\n"
    )
    layout, _ = parse_spatiallm_output(text, _coord_offset())
    flags = [w["has_door"] for w in layout["walls"]]
    assert flags == [True, False, True]


def test_parser_handles_door_and_window_on_same_wall():
    text = (
        "wall_0=Wall(0.0,0.0,0.0,3.0,0.0,0.0,2.4,0.0)\n"
        "door_0=Door(wall_0,0.5,0.0,1.0,0.9,2.1)\n"
        "window_0=Window(wall_0,2.0,0.0,1.5,1.2,0.8)\n"
    )
    layout, _ = parse_spatiallm_output(text, _coord_offset())
    assert layout["walls"][0]["has_door"] is True
    assert layout["walls"][0]["has_window"] is True


def test_parser_no_doors_no_windows_yields_all_false():
    text = "wall_0=Wall(0.0,0.0,0.0,3.0,0.0,0.0,2.4,0.0)\n"
    layout, _ = parse_spatiallm_output(text, _coord_offset())
    w = layout["walls"][0]
    assert w["has_door"] is False
    assert w["has_window"] is False


def test_parser_door_referencing_unknown_wall_is_ignored():
    """Defensive: SLM occasionally emits door_N=Door(wall_99, ...) for a
    wall index that doesn't exist. The parser should not crash; that
    door simply doesn't get attached anywhere."""
    text = (
        "wall_0=Wall(0.0,0.0,0.0,3.0,0.0,0.0,2.4,0.0)\n"
        "door_0=Door(wall_99,1.0,0.0,1.0,0.9,2.1)\n"
    )
    layout, _ = parse_spatiallm_output(text, _coord_offset())
    assert layout["walls"][0]["has_door"] is False
