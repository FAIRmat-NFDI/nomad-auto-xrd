#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from nomad.config.models.ui import (
    App,
    Axis,
    AxisScale,
    BreakpointEnum,
    Column,
    Dashboard,
    Filters,
    Layout,
    Menu,
    MenuItemHistogram,
    MenuItemTerms,
    MenuSizeEnum,
    Pagination,
    ScaleEnum,
    SearchSyntaxes,
    WidgetHistogram,
    WidgetPeriodicTable,
    WidgetTerms,
)

SCHEMA_PATH = 'nomad_auto_xrd.schema_packages.schema.AutoXRDModel'

models_app = App(
    label='Auto XRD Models',
    path='auto-xrd-models',
    category='Use Cases',
    description='Search auto XRD models',
    readme=(
        'This page allows you to search **auto XRD models entries** based '
        'on model parameters and reference structures used to train the model.'
    ),
    filters=Filters(
        include=[
            f'*#{SCHEMA_PATH}',
        ],
    ),
    filters_locked={
        'entry_type': 'AutoXRDModel',
    },
    pagination=Pagination(
        order_by=f'data.evaluation_metrics.accuracy#{SCHEMA_PATH}',
        order='desc',
        page_size=20,
    ),
    search_syntaxes=SearchSyntaxes(exclude=['free_text']),
    columns=[
        Column(
            search_quantity='results.material.elements',
            label='Elements',
            selected=True,
        ),
        Column(
            search_quantity=f'data.simulation_settings.min_angle#{SCHEMA_PATH}',
            label='Minimum 2 theta',
            selected=True,
        ),
        Column(
            search_quantity=f'data.simulation_settings.max_angle#{SCHEMA_PATH}',
            label='Maximum 2 theta',
            selected=True,
        ),
        Column(
            search_quantity=f'data.training_settings.num_epochs#{SCHEMA_PATH}',
            label='Number of epochs',
            selected=True,
        ),
        Column(
            search_quantity=f'data.evaluation_metrics.accuracy#{SCHEMA_PATH}',
            label='Test Accuracy',
            selected=True,
        ),
    ],
    filter_menus=None,
    menu=Menu(
        size=MenuSizeEnum.SM,
        items=[
            Menu(
                title='Simulation Settings',
                size=MenuSizeEnum.XXL,
                items=[
                    MenuItemHistogram(
                        title='Minimum 2 Theta',
                        x=Axis(
                            search_quantity=(
                                f'data.simulation_settings.min_angle#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                            unit='deg',
                        ),
                    ),
                    MenuItemHistogram(
                        title='Maximum 2 Theta',
                        x=Axis(
                            search_quantity=(
                                f'data.simulation_settings.max_angle#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                            unit='deg',
                        ),
                    ),
                    MenuItemHistogram(
                        title='Min Domain Size',
                        x=Axis(
                            search_quantity=(
                                'data.simulation_settings.min_domain_size'
                                f'#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                            unit='nm',
                        ),
                    ),
                    MenuItemHistogram(
                        title='Max Domain Size',
                        x=Axis(
                            search_quantity=(
                                'data.simulation_settings.max_domain_size'
                                f'#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                            unit='nm',
                        ),
                    ),
                    MenuItemHistogram(
                        title='Max Strain',
                        x=Axis(
                            search_quantity=(
                                f'data.simulation_settings.max_strain#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                        ),
                    ),
                    MenuItemHistogram(
                        title='Max Texture',
                        x=Axis(
                            search_quantity=(
                                f'data.simulation_settings.max_texture#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                        ),
                    ),
                    MenuItemHistogram(
                        title='Num Patterns per Phase',
                        x=Axis(
                            search_quantity=(
                                f'data.simulation_settings.num_patterns#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                        ),
                    ),
                ],
            ),
            Menu(
                title='Training Settings',
                size=MenuSizeEnum.XXL,
                items=[
                    MenuItemHistogram(
                        title='Number of Epochs',
                        x=Axis(
                            search_quantity=(
                                f'data.training_settings.num_epochs#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                        ),
                    ),
                    MenuItemHistogram(
                        title='Batch Size',
                        x=Axis(
                            search_quantity=(
                                f'data.training_settings.batch_size#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                        ),
                    ),
                    MenuItemHistogram(
                        title='Learning Rate',
                        x=Axis(
                            search_quantity=(
                                f'data.training_settings.learning_rate#{SCHEMA_PATH}'
                            ),
                            scale='log',
                        ),
                    ),
                    MenuItemHistogram(
                        title='Test Fraction',
                        x=Axis(
                            search_quantity=(
                                f'data.training_settings.test_fraction#{SCHEMA_PATH}'
                            ),
                            scale='linear',
                        ),
                    ),
                ],
            ),
            Menu(
                title='Material',
                size=MenuSizeEnum.XXL,
                items=[
                    MenuItemTerms(
                        title='Elements',
                        search_quantity='results.material.elements',
                        options=10,
                        show_input=True,
                    ),
                    MenuItemTerms(
                        title='Crystal System',
                        search_quantity=(
                            'results.material.topology.symmetry.crystal_system'
                        ),
                        options=10,
                        show_input=True,
                    ),
                    MenuItemTerms(
                        title='Formula',
                        search_quantity=(
                            'results.material.topology.chemical_formula_iupac'
                        ),
                        options=10,
                        show_input=True,
                    ),
                ],
            ),
        ],
    ),
    dashboard=Dashboard(
        widgets=[
            WidgetPeriodicTable(
                search_quantity='results.material.elements',
                scale=ScaleEnum.LINEAR,
                layout={
                    BreakpointEnum.XXL: Layout(h=9, w=13, y=0, x=0, minH=8, minW=12),
                    BreakpointEnum.XL: Layout(h=9, w=12, y=0, x=0, minH=8, minW=12),
                    BreakpointEnum.LG: Layout(h=8, w=12, y=0, x=0, minH=8, minW=12),
                    BreakpointEnum.MD: Layout(h=8, w=12, y=0, x=0, minH=8, minW=12),
                    BreakpointEnum.SM: Layout(h=8, w=12, y=0, x=0, minH=8, minW=12),
                },
            ),
            WidgetTerms(
                search_quantity='results.material.topology.chemical_formula_iupac',
                scale=ScaleEnum.LINEAR,
                show_input=True,
                layout={
                    BreakpointEnum.XXL: Layout(h=9, w=6, y=0, x=13),
                    BreakpointEnum.XL: Layout(h=9, w=5, y=0, x=12),
                    BreakpointEnum.LG: Layout(h=8, w=6, y=0, x=12),
                    BreakpointEnum.MD: Layout(h=8, w=3, y=0, x=12),
                    BreakpointEnum.SM: Layout(h=7, w=4, y=11, x=8),
                },
            ),
            WidgetHistogram(
                title='Number of epochs',
                autorange=True,
                n_bins=30,
                x=Axis(
                    search_quantity=f'data.training_settings.num_epochs#{SCHEMA_PATH}',
                    title='Number of epochs',
                ),
                y=AxisScale(scale=ScaleEnum.LINEAR),
                layout={
                    BreakpointEnum.XXL: Layout(h=3, w=8, y=6, x=25),
                    BreakpointEnum.XL: Layout(h=3, w=8, y=6, x=22),
                    BreakpointEnum.LG: Layout(h=3, w=8, y=8, x=16),
                    BreakpointEnum.MD: Layout(h=3, w=4, y=8, x=8),
                    BreakpointEnum.SM: Layout(h=3, w=8, y=11, x=0),
                },
            ),
            WidgetHistogram(
                title='Maximum 2 theta',
                autorange=False,
                n_bins=30,
                x=Axis(
                    search_quantity=f'data.simulation_settings.max_angle#{SCHEMA_PATH}',
                    title='Maximum 2 theta',
                ),
                y=AxisScale(scale=ScaleEnum.LINEAR),
                layout={
                    BreakpointEnum.XXL: Layout(h=3, w=8, y=0, x=25),
                    BreakpointEnum.XL: Layout(h=3, w=8, y=3, x=22),
                    BreakpointEnum.LG: Layout(h=3, w=8, y=8, x=8),
                    BreakpointEnum.MD: Layout(h=3, w=4, y=8, x=4),
                    BreakpointEnum.SM: Layout(h=3, w=6, y=8, x=6),
                },
            ),
            WidgetHistogram(
                title='Minimum 2 theta',
                autorange=False,
                n_bins=30,
                x=Axis(
                    search_quantity=f'data.simulation_settings.min_angle#{SCHEMA_PATH}',
                    title='Minimum 2 theta',
                ),
                y=AxisScale(scale=ScaleEnum.LINEAR),
                layout={
                    BreakpointEnum.XXL: Layout(h=3, w=8, y=3, x=25),
                    BreakpointEnum.XL: Layout(h=3, w=8, y=0, x=22),
                    BreakpointEnum.LG: Layout(h=3, w=8, y=8, x=0),
                    BreakpointEnum.MD: Layout(h=3, w=4, y=8, x=0),
                    BreakpointEnum.SM: Layout(h=3, w=6, y=8, x=0),
                },
            ),
            WidgetTerms(
                search_quantity='results.material.topology.symmetry.crystal_system',
                scale=ScaleEnum.LINEAR,
                layout={
                    BreakpointEnum.XXL: Layout(h=9, w=6, y=0, x=15),
                    BreakpointEnum.XL: Layout(h=9, w=6, y=0, x=15),
                    BreakpointEnum.LG: Layout(h=9, w=6, y=0, x=15),
                    BreakpointEnum.MD: Layout(h=8, w=3, y=0, x=15),
                    BreakpointEnum.SM: Layout(h=4, w=4, y=14, x=0),
                },
            ),
        ]
    ),
)
