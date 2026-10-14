from app.normalize import (
    GEMSTONE_SYNONYMS,
    METAL_SYNONYMS,
    normalise_specifications,
)
from app.schemas import ExtractedSpecifications


def specs(**kwargs) -> ExtractedSpecifications:
    base = dict(
        item_type="unknown", metal="unknown", purity="unknown",
        gemstone="unknown", target_weight_grams=0.0,
    )
    base.update(kwargs)
    return ExtractedSpecifications(**base)


def test_purity_is_uppercased_to_catalog_spelling():
    assert normalise_specifications(specs(purity="18k")).purity == "18K"
    assert normalise_specifications(specs(purity=" 18 K ")).purity == "18K"
    assert normalise_specifications(specs(purity="925")).purity == "925"
    assert normalise_specifications(specs(purity="9 25")).purity == "925"


def test_metal_synonyms_map_onto_catalog_values():
    assert normalise_specifications(specs(metal="yellow gold")).metal == "gold"
    assert normalise_specifications(specs(metal="white gold")).metal == "gold"
    assert normalise_specifications(specs(metal="sterling silver")).metal == "silver"
    assert normalise_specifications(specs(metal="pink gold")).metal == "rose gold"


def test_metals_absent_from_the_catalog_become_unknown():
    """A confident-looking value that can never match must not masquerade as an answer."""
    assert normalise_specifications(specs(metal="titanium")).metal == "unknown"
    assert normalise_specifications(specs(metal="stainless steel")).metal == "unknown"


def test_free_form_gemstones_map_onto_catalog_names():
    assert normalise_specifications(specs(gemstone="purple stone")).gemstone == "amethyst"
    assert normalise_specifications(specs(gemstone="red stone")).gemstone == "ruby"
    assert normalise_specifications(specs(gemstone="black onyx")).gemstone == "black onyx"
    assert normalise_specifications(specs(gemstone="plain")).gemstone == "none"


def test_unknown_gemstone_falls_back_to_the_brief():
    assert normalise_specifications(specs(gemstone="unknown"), "a plain gold ring").gemstone == "none"
    assert normalise_specifications(specs(gemstone="unknown"), "a gold ring").gemstone == "unknown"


def test_absurd_weight_is_discarded_but_an_odd_one_is_kept():
    assert normalise_specifications(specs(target_weight_grams=6000)).target_weight_grams == 0.0
    # A merely implausible weight is preserved: the brief said it, and the review engine
    # flags it. Discarding it here would hide the rep's typo instead of surfacing it.
    assert normalise_specifications(specs(target_weight_grams=900)).target_weight_grams == 900.0
    assert normalise_specifications(specs(target_weight_grams=3.2)).target_weight_grams == 3.2


def test_normalisation_is_idempotent():
    once = normalise_specifications(
        specs(item_type="earring", metal="white gold", purity="18k", gemstone="purple stone"),
        "a purple stone ring in white gold",
    )
    twice = normalise_specifications(once, "a purple stone ring in white gold")
    assert once.model_dump() == twice.model_dump()


def test_every_mapped_value_actually_exists_in_the_catalog():
    """The synonym tables must not invent vocabulary the catalog does not use."""
    from app.normalize import catalog_values

    for mapped in METAL_SYNONYMS.values():
        if mapped != "unknown":
            assert mapped in catalog_values("metal"), mapped
    for mapped in GEMSTONE_SYNONYMS.values():
        if mapped != "unknown":
            assert mapped in catalog_values("gemstone"), mapped