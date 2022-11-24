from AD_structure import utils
def test_get_config():
    cfg = utils.get_config()
    assert cfg is not None
    