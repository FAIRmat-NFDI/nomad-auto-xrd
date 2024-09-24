from nomad.config.models.plugins import ExampleUploadEntryPoint

example_upload_entry_point = ExampleUploadEntryPoint(
    title='Auto XRD example upload',
    category='Examples',
    description='This example upload contains a notebook and sinstructions on how to train an [XRD Auto Analyzer](https://github.com/njszym/XRD-AutoAnalyzer) model in NORTH and save it as an entry.',  # noqa: E501
    path='example_uploads/getting_started',
)
