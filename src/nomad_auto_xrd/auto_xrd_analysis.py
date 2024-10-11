import json
import os
import time

import numpy as np  # Import numpy for array handling
from autoXRD import spectrum_analysis, visualizer


def convert_to_serializable(obj):
    """Convert non-serializable objects like numpy arrays to serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def run_analysis(  # noqa: PLR0913
    references_folder='References',
    spectra_folder='Spectra',
    xrd_model_path='Models/XRD_Model.h5',
    pdf_model_path='Models/PDF_Model.h5',
    results_file='results.json',
    max_phases=3,
    cutoff_intensity=1,
    min_conf=40,
    wavelength='CuKa',
    unknown_threshold=25.0,
    show_reduced=False,
    inc_pdf=False,
    parallel=False,
    raw=True,
    show_indiv=False,
    min_angle=25.00,
    max_angle=80.00,
):
    start = time.time()

    # Check for spectra
    if not os.path.exists(spectra_folder) or len(os.listdir(spectra_folder)) == 0:
        print(f'Please provide at least one pattern in the {spectra_folder} directory.')
        return

    results = {'XRD': {}, 'PDF': {}}

    # XRD/PDF ensemble requires all predictions
    if inc_pdf:
        final_conf = min_conf
        min_conf = 10.0

    # Ensure temp directory exists
    if not os.path.exists('temp'):
        os.mkdir('temp')

    # Get predictions from XRD analysis
    (
        results['XRD']['filenames'],
        results['XRD']['phases'],
        results['XRD']['confs'],
        results['XRD']['backup_phases'],
        results['XRD']['scale_factors'],
        results['XRD']['reduced_spectra'],
    ) = spectrum_analysis.main(
        spectra_folder,
        references_folder,
        max_phases,
        cutoff_intensity,
        min_conf,
        wavelength,
        min_angle,
        max_angle,
        parallel,
        xrd_model_path,
        is_pdf=False,
    )

    if inc_pdf:
        # Get predictions from PDF analysis
        (
            results['PDF']['filenames'],
            results['PDF']['phases'],
            results['PDF']['confs'],
            results['PDF']['backup_phases'],
            results['PDF']['scale_factors'],
            results['PDF']['reduced_spectra'],
        ) = spectrum_analysis.main(
            spectra_folder,
            references_folder,
            max_phases,
            cutoff_intensity,
            min_conf,
            wavelength,
            min_angle,
            max_angle,
            parallel,
            pdf_model_path,
            is_pdf=True,
        )

        # Merge results
        results['Merged'] = spectrum_analysis.merge_results(
            results, final_conf, max_phases
        )
    else:
        results['Merged'] = results['XRD']

    # Process results
    for idx, (
        spectrum_fname,
        phase_set,
        confidence,
        backup_set,
        heights,
        final_spectrum,
    ) in enumerate(
        zip(
            results['Merged']['filenames'],
            results['Merged']['phases'],
            results['Merged']['confs'],
            results['Merged']['backup_phases'],
            results['Merged']['scale_factors'],
            results['Merged']['reduced_spectra'],
        )
    ):
        # Display phase ID info
        print(f'Filename: {spectrum_fname}')
        print(f'Predicted phases: {phase_set}')
        print(f'Confidence: {confidence}')

        # Check for unknown peaks
        if len(phase_set) > 0 and 'None' not in phase_set:
            remaining_I = max(final_spectrum)
            if remaining_I > unknown_threshold:
                print(
                    f'WARNING: some peaks (I ~ {int(remaining_I)}%) were not identified.'  # noqa: E501
                )
        else:
            print('WARNING: no phases were identified')
            continue

        # Show backup predictions
        if show_indiv:
            print(f"XRD predicted phases: {results['XRD']['phases'][idx]}")
            print(f"XRD confidence: {results['XRD']['confs'][idx]}")
            if inc_pdf:
                print(f"PDF predicted phases: {results['PDF']['phases'][idx]}")
                print(f"PDF confidence: {results['PDF']['confs'][idx]}")

        # Plot the results
        phasenames = [f'{phase}.cif' for phase in phase_set]
        visualizer.main(
            spectra_folder,
            spectrum_fname,
            phasenames,
            heights,
            final_spectrum,
            min_angle,
            max_angle,
            wavelength,
            save=False,
            show_reduced=show_reduced,
            inc_pdf=inc_pdf,
            plot_both=False,
            raw=raw,
        )

    end = time.time()
    print(f'Total time: {round(end - start, 1)} sec')

    # Convert results to a JSON serializable format
    serializable_results = convert_to_serializable(results)

    # Save the results dictionary as a JSON file
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f'Results saved to {results_file}')
