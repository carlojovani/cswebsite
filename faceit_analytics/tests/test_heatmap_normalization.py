from faceit_analytics.services.heatmaps import normalize_side


def test_side_normalization_ct_t_all():
    assert normalize_side("CT") == "ct"
    assert normalize_side("T") == "t"
    assert normalize_side("ALL") == "all"
    assert normalize_side("ct") == "ct"
    assert normalize_side("unknown") == "all"
