<img width="2981" height="911" alt="image" src="https://github.com/user-attachments/assets/8a8a5818-0777-41ed-965f-06180ce0162c" />

A NOMAD plugin providing schemas, [Actions](https://nomad-lab.eu/prod/v1/docs/howto/plugins/types/actions.html),
and a search app for supporting automated XRD analysis using ML models.
It is compatible with
[nomad-measurements](https://github.com/FAIRmat-NFDI/nomad-measurements) plugin and allows
to perform automated phase identification on XRD measurement entries.

## How it works

NOMAD schemas can be used to define interactive and reactive ELNs. One can attach functions
that are triggered from the ELNs. On top of that, NOMAD Actions allows to setup 
workflows that can run robustly on specialized workers. We use these capabilities to build
a collaborative tool for training ML models and using them for automated phase identification
from XRD patterns.

The model training and analysis modules are built on top of [XRD-AutoAnalyser](https://github.com/njszym/XRD-AutoAnalyzer).

Some of the key components of the `nomad-auto-xrd` plugin are:

- Schema package: It contains schemas for triggering actions and saving artifacts like analysis
  results, trained Auto XRD models, etc.
  - `Auto XRD Model`: A schema for trained ML models that can contain metadata like training
    settings and material composition for which the model is trained.
  - `Auto XRD Training Action`: An ELN schema that takes CIF structure files, simulation, and training
    settings as input. A trigger button can be used to trigger the training workflow, which gives
    the trained ML model as output.
  - `Auto XRD Analysis Action`: An ELN schema that takes XRD measurement entry, Auto XRD model entry,
    and analysis settings as input. A trigger button can be used to trigger the analysis workflow,
    which predicts the most probable phases based on the XRD pattern. 
  - `Auto XRD Training` and `Auto XRD Analysis`: Instead of using NOMAD Actions, these ELN schemas
    provide similar functionality as above, using custom Jupyter notebooks with predefined
    code cells containing corresponding code. This is useful when a higher degree of
    customization (to the processing code, for example) is needed.
- Actions:
  - 
- Search App:

## Availability

This plugin is hosted on [NOMAD's example Oasis](https://nomad-lab.eu/prod/v1/oasis/gui/). 
You can create an entry with the built-in schemas `Auto XRD Analysis Action`, add an XRD
measurement entry and an Auto XRD model entry to it, and trigger a phase identification
analysis. You can also use a dedicated search app to look up available Auto XRD model
entries [here](https://nomad-lab.eu/prod/v1/oasis/gui/search/auto-xrd-models).

## Adding this plugin to NOMAD Oasis

The plugin can be added to a
[NOMAD Oasis](https://nomad-lab.eu/prod/v1/docs/reference/glossary.html#deployment-nomad-oasis)
instance by making it a dependency in the `pyproject.toml` of the Oasis repository:

```yaml
[project.optional-dependencies]
plugins = [
  ...
  "nomad-auto-xrd",
]
cpu-action = [
  ...
  "nomad-auto-xrd[cpu-action]",
]

[tool.uv.sources]
nomad-auto-xrd = { git = "https://github.com/FAIRmat-NFDI/nomad-auto-xrd.git", rev = "v0.3.9" }
```

Read the [NOMAD plugin documentation](https://nomad-lab.eu/prod/v1/docs/howto/oasis/configure.html#plugins)
for all details on how to deploy the plugin on your NOMAD instance. If you wanna get started with a new Oasis,
start [here](https://nomad-lab.eu/prod/v1/docs/howto/oasis/install.html#how-to-install-a-nomad-oasis).

## Development

We recommend using the dedicated `nomad-distro-dev` repository to simplify the development process.
It allows running a local NOMAD installation and testing the behavior of the plugins in real-time. 
Please refer to that [repository](https://github.com/FAIRmat-NFDI/nomad-distro-dev) for detailed instructions.


## Main contributors
| Name | E-mail     |
|------|------------|
| Pepe Márquez | [jose.marquez@physik.hu-berlin.de](mailto:jose.marquez@physik.hu-berlin.de)
| Sarthak Kapoor | [sarthak.kapoor@physik.hu-berlin.de](mailto:sarthak.kapoor@physik.hu-berlin.de)

## License
Distributed under the terms of the Apache-2.0 license, the `nomad-auto-xrd` plugin is free and open source software.
