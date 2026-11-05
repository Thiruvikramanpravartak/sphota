from sphota.build.label_filter import check, filter_labels, offensive


def test_accepts_good_labels():
    assert check("en", "Corporate Executive") is None
    assert check("hi", "ऐतिहासिक शादी") is None  # vowel signs are marks, not junk
    assert check("ta", "ஐரிஷ் விளையாட்டுப் படங்கள்") is None
    assert check("zh", "YouTube 内容上传") is None
    assert check("es", "Ejecutivo corporativo") is None


def test_rejects_garbage():
    assert check("ar", "Y Y Y Y Y Y") == "repeated tokens"
    assert check("hi", "Carcutiover और हेवोस") == "wrong script for hi"
    assert check("ru", "Corporate Executive") == "wrong script for ru"
    assert check("zh", "1987 1988 1989 年") is not None
    assert check("en", "x" * 80) == "too long"


def test_offensive_cells_held_for_review():
    assert offensive("Sex Work and Rights Issues")
    assert not offensive("Corporate Executive")
    kept, review = filter_labels({"1": {"en": "Corporate Executive", "ru": "Corporate Executive"},
                                  "2": {"en": "Porn Websites"}})
    assert kept == {"1": {"en": "Corporate Executive"}}
    assert "reason" in review["2"] and review["1"]["rejected"]["ru"]
