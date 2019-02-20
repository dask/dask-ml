import pytest

try:
    import distributed

    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False


@pytest.mark.skipif(HAS_DISTRIBUTED, reason="has package `distribtued`")
def test_incremental_no_distributed():
    with pytest.warns(ImportWarning, match="foo"):
        import dask_ml.model_selection
