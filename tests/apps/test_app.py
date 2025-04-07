def test_importing_app():
    # this will raise an exception if pydantic model validation fails for th app
    from nomad_auto_xrd.apps import models_app_entry_point

    assert models_app_entry_point.app.label == 'Auto XRD Models'
