import pytest

from sphota import unsupported_script
from sphota.api import _warn_unsupported


def test_unsupported_scripts_detected():
    assert unsupported_script("ᱥᱟᱱᱛᱟᱲᱤ ᱯᱟᱹᱨᱥᱤ") == "Ol Chiki (Santali)"
    assert unsupported_script("ꯃꯤꯇꯩꯂꯣꯟ") == "Meetei Mayek (Manipuri)"
    assert unsupported_script("मेरा ऑर्डर नहीं आया") is None
    assert unsupported_script("hello") is None
    assert unsupported_script("123 !!") is None


def test_warning_only_for_unsupported():
    with pytest.warns(UserWarning, match="Ol Chiki"):
        _warn_unsupported(["ok", "ᱥᱟᱱᱛᱟᱲᱤ"])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_unsupported(["ok", "ठीक है"])
