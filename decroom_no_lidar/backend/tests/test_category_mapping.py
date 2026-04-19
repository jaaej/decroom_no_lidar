"""
Every SpatialLM class the backend accepts must map to a known Decroom category.
If SpatialLM emits a new class, it must either be added to SPATIALLM_TO_PRD or
explicitly logged as "not mapped" (which the pipeline does at DEBUG).
"""
from pipeline.spatiallm_layout import SPATIALLM_TO_PRD

# The Decroom categories the iOS app knows how to render.
DECROOM_14 = {
    "bed", "sofa", "chair", "desk", "table", "nightstand",
    "dresser", "bookshelf", "monitor", "lamp_floor", "lamp_desk",
    "plant", "rug", "poster",
}


def test_all_mapped_categories_are_decroom_14():
    for src, dst in SPATIALLM_TO_PRD.items():
        assert dst in DECROOM_14, f"{src!r} -> {dst!r} is not in the Decroom 14"


def test_bed_variants_all_map_to_bed():
    for variant in ("bed", "single_bed", "double_bed", "bunk_bed"):
        assert SPATIALLM_TO_PRD.get(variant) == "bed"


def test_sofa_variants_all_map_to_sofa():
    for variant in ("sofa", "l_shaped_sofa"):
        assert SPATIALLM_TO_PRD.get(variant) == "sofa"


def test_dining_table_maps_to_desk():
    # Preserved from pre-3a backend category mapping.
    assert SPATIALLM_TO_PRD["dining_table"] == "desk"
