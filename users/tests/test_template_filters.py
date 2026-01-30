from users.templatetags.dict_extras import get_item


def test_get_item_handles_missing():
    assert get_item({"0-15": 42}, "0-15") == 42
    assert get_item({"0-15": None}, "0-30") is None
    assert get_item(None, "0-15") is None
