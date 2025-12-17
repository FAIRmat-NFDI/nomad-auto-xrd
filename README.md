<img width="2981" height="911" alt="image" src="https://github.com/user-attachments/assets/8a8a5818-0777-41ed-965f-06180ce0162c" />

A NOMAD plugin providing schemas, [Actions](https://nomad-lab.eu/prod/v1/docs/howto/plugins/types/actions.html),
and a search app for supporting automated XRD analysis using ML models.
Combined with
[nomad-measurements](https://github.com/FAIRmat-NFDI/nomad-measurements) plugin, it provides a comprehensive
solution for your NOMAD Oasis to create XRD measurement entries using raw files from the instrument and
perform automated phase identification on them.

## Availability

This plugin is hosted on [NOMAD's example Oasis](https://nomad-lab.eu/prod/v1/oasis/gui/). 
You can create an entry with the built-in schemas `Auto XRD Analysis Action`, add an XRD
measurement entry and an Auto XRD model entry to it, and trigger a phase identification
analysis. You can also use a dedicated search app to look up available Auto XRD model
entries [here](https://nomad-lab.eu/prod/v1/oasis/gui/search/auto-xrd-models).

## Adding this plugin to NOMAD Oasis

The plugin can be added to a
[NOMAD Oasis](https://nomad-lab.eu/prod/v1/docs/reference/glossary.html#deployment-nomad-oasis)
instance in a few steps, provided that you have access to the repository hosting the Oasis.

Make sure to add [nomad-measurements](https://github.com/FAIRmat-NFDI/nomad-measurements) plugin
that contains relevant schemas and parsers for creating XRD measurement entries.

Read the [NOMAD plugin documentation](https://nomad-lab.eu/prod/v1/docs/howto/oasis/configure.html#plugins)
for all details on how to deploy the plugin on your NOMAD instance. If you wanna get started with a new Oasis,
start [here](https://nomad-lab.eu/prod/v1/docs/howto/oasis/install.html#how-to-install-a-nomad-oasis).

## Development

If you want to develop locally this plugin, clone the project and in the plugin folder, create a virtual environment (you can use Python 3.9, 3.10, or 3.11):
```sh
git clone https://github.com/foo/nomad-auto-xrd.git
cd nomad-auto-xrd
python3.11 -m venv .pyenv
. .pyenv/bin/activate
```

Make sure to have `pip` upgraded:
```sh
pip install --upgrade pip
```

We recommend installing `uv` for fast pip installation of the packages:
```sh
pip install uv
```

Install the `nomad-lab` package:
```sh
uv pip install '.[dev]' --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```

**Note!**
Until we have an official pypi NOMAD release with the plugins functionality make
sure to include NOMAD's internal package registry (via `--index-url` in the above command).

The plugin is still under development. If you would like to contribute, install the package in editable mode (with the added `-e` flag):
```sh
uv pip install -e '.[dev]' --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```


### Run the tests

You can run locally the tests:
```sh
python -m pytest -sv tests
```

where the `-s` and `-v` options toggle the output verbosity.

Our CI/CD pipeline produces a more comprehensive test report using the `pytest-cov` package. You can generate a local coverage report:
```sh
uv pip install pytest-cov
python -m pytest --cov=src tests
```

By default, the tests related to training and inference of the models are
skipped. If you want to execute them, set the environment variable
`RUN_PIPELINE_TESTS` before running the tests.
```sh
export RUN_PIPELINE_TESTS=true
```

### Run linting and auto-formatting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting the code. Ruff auto-formatting is also a part of the GitHub workflow actions. You can run locally:
```sh
ruff check .
ruff format . --check
```


## Main contributors
| Name | E-mail     |
|------|------------|
| Pepe Márquez | [jose.marquez@physik.hu-berlin.de](mailto:jose.marquez@physik.hu-berlin.de)
| Sarthak Kapoor | [sarthak.kapoor@physik.hu-berlin.de](mailto:sarthak.kapoor@physik.hu-berlin.de)
