import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np

# Import necessary modules from autoXRD
from autoXRD import (
    spectrum_analysis,
    visualizer,
)  # Assume these can work with in-memory data


# Function to convert non-serializable objects to serializable formats
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


@dataclass
class AnalysisConfig:
    # Analysis parameters
    max_phases: int = 3
    cutoff_intensity: float = 1.0
    min_conf: float = 40.0
    wavelength: str = 'CuKa'
    unknown_threshold: float = 25.0
    show_reduced: bool = False
    inc_pdf: bool = False
    parallel: bool = False
    raw: bool = True
    show_indiv: bool = False
    min_angle: float = 25.0
    max_angle: float = 80.0


def run_analysis(
    reference_cifs: List[str],
    spectra_data: List[np.ndarray],
    model_data: Dict[str, bytes],
    config: AnalysisConfig,
):
    """
    Runs the analysis on the provided spectra using the reference CIFs and models.

    Args:
        reference_cifs: List of CIF file contents as strings.
        spectra_data: List of spectra to analyze, each as a numpy array.
        model_data: Dictionary with keys 'XRD' and 'PDF' (if inc_pdf=True) containing model data as bytes.
        config: AnalysisConfig object containing analysis parameters.

    Returns:
        A dictionary containing the analysis results.
    """
    start = time.time()

    if not spectra_data or len(spectra_data) == 0:
        print('Please provide at least one spectrum in spectra_data.')
        return

    results = {'XRD': {}, 'PDF': {}}

    # XRD/PDF ensemble requires all predictions
    if config.inc_pdf:
        final_conf = config.min_conf
        min_conf = 10.0
    else:
        min_conf = config.min_conf

    # Prepare models
    xrd_model_bytes = model_data.get('XRD')
    pdf_model_bytes = model_data.get('PDF') if config.inc_pdf else None

    # Run XRD analysis
    xrd_results = spectrum_analysis.run_analysis(
        spectra_data=spectra_data,
        reference_cifs=reference_cifs,
        model_bytes=xrd_model_bytes,
        max_phases=config.max_phases,
        cutoff_intensity=config.cutoff_intensity,
        min_conf=min_conf,
        wavelength=config.wavelength,
        min_angle=config.min_angle,
        max_angle=config.max_angle,
        parallel=config.parallel,
        is_pdf=False,
    )
    results['XRD'] = xrd_results

    if config.inc_pdf:
        # Run PDF analysis
        pdf_results = spectrum_analysis.run_analysis(
            spectra_data=spectra_data,
            reference_cifs=reference_cifs,
            model_bytes=pdf_model_bytes,
            max_phases=config.max_phases,
            cutoff_intensity=config.cutoff_intensity,
            min_conf=min_conf,
            wavelength=config.wavelength,
            min_angle=config.min_angle,
            max_angle=config.max_angle,
            parallel=config.parallel,
            is_pdf=True,
        )
        results['PDF'] = pdf_results

        # Merge results
        results['Merged'] = spectrum_analysis.merge_results(
            results, final_conf, config.max_phases
        )
    else:
        results['Merged'] = results['XRD']

    # Process results
    for idx, (
        spectrum,
        phase_set,
        confidence,
        backup_set,
        heights,
        final_spectrum,
    ) in enumerate(
        zip(
            spectra_data,
            results['Merged']['phases'],
            results['Merged']['confs'],
            results['Merged']['backup_phases'],
            results['Merged']['scale_factors'],
            results['Merged']['reduced_spectra'],
        )
    ):
        # Display phase ID info
        print(f'Spectrum index: {idx}')
        print(f'Predicted phases: {phase_set}')
        print(f'Confidence: {confidence}')

        # Check for unknown peaks
        if len(phase_set) > 0 and 'None' not in phase_set:
            remaining_I = max(final_spectrum)
            if remaining_I > config.unknown_threshold:
                print(
                    f'WARNING: some peaks (I ~ {int(remaining_I)}%) were not identified.'
                )
        else:
            print('WARNING: no phases were identified')
            continue

        # Show backup predictions
        if config.show_indiv:
            print(f"XRD predicted phases: {results['XRD']['phases'][idx]}")
            print(f"XRD confidence: {results['XRD']['confs'][idx]}")
            if config.inc_pdf:
                print(f"PDF predicted phases: {results['PDF']['phases'][idx]}")
                print(f"PDF confidence: {results['PDF']['confs'][idx]}")

        # Plot the results
        phasenames = [f'{phase}.cif' for phase in phase_set]
        visualizer.plot_results(
            spectrum=spectrum,
            reference_cifs=reference_cifs,
            phasenames=phasenames,
            heights=heights,
            final_spectrum=final_spectrum,
            min_angle=config.min_angle,
            max_angle=config.max_angle,
            wavelength=config.wavelength,
            show_reduced=config.show_reduced,
            inc_pdf=config.inc_pdf,
            raw=config.raw,
        )

    end = time.time()
    print(f'Total time: {round(end - start, 1)} sec')

    # Convert results to a JSON serializable format
    serializable_results = convert_to_serializable(results)

    # Return the results dictionary
    return serializable_results
