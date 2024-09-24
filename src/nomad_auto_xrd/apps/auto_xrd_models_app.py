import yaml

auto_xrd_models_app = yaml.safe_load(
    """
label: Auto XRD Models
path: auto-xrd-models
category: Use Cases
description: Search auto XRD models
readme: 'This page allows you to search **auto XRD models entries** based on model parameters
  and reference structures used to train the model.'
filters:
  include:
  - '*#nomad_auto_xrd.schema_packages.auto_xrd.AutoXRDModel'
  exclude:
  - mainfile
  - entry_name
  - combine
filters_locked:
  sections: nomad_auto_xrd.schema_packages.auto_xrd.AutoXRDModel
pagination:
  order_by: results.properties.optoelectronic.solar_cell.efficiency
search_syntaxes:
  exclude:
  - free_text
columns:
  selected:
  - results.material.chemical_formula_descriptive
  - entry_name
  options:
    results.material.chemical_formula_descriptive: {label: Descriptive formula}
    references: {}
    results.material.chemical_formula_hill:
      label: Formula
    results.material.structural_type: {}
    results.eln.sections: {}
    results.eln.methods: {}
    entry_name: {label: Name}
    entry_type: {}
    mainfile: {}
    upload_create_time: {label: Upload time}
    authors: {}
    comment: {}
    datasets: {}
    published: {label: Access}
filter_menus:
  options:
    material:
      label: Material
    elements:
      label: Elements / Formula
      level: 1
      size: xl
    structure:
      label: Structure / Symmetry
      level: 1
    custom_quantities:
      label: User Defined Quantities
      size: l
    author:
      label: Author / Origin / Dataset
      size: m
    metadata:
      label: Visibility / IDs / Schema
    optimade:
      label: Optimade
      size: m
dashboard:
  widgets:
  - layout:
      lg: {h: 8, minH: 8, minW: 12, w: 12, x: 0, y: 0}
      md: {h: 8, minH: 8, minW: 12, w: 12, x: 0, y: 0}
      sm: {h: 8, minH: 8, minW: 12, w: 12, x: 0, y: 16}
      xl: {h: 8, minH: 8, minW: 12, w: 12, x: 0, y: 0}
      xxl: {h: 8, minH: 8, minW: 12, w: 13, x: 0, y: 0}
    quantity: results.material.elements
    scale: linear
    type: periodictable
"""
)
