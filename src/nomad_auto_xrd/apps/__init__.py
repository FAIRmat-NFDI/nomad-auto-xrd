from nomad.config.models.plugins import AppEntryPoint

from nomad_auto_xrd.apps.models_app import models_app

models_app_entry_point = AppEntryPoint(
    name='Auto XRD Models',
    description='Search auto XRD models',
    app=models_app,
)
