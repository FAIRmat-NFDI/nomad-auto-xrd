import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from autoXRD import spectrum_analysis, visualizer

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger


def convert_to_serializable(obj):
    """Convert non-serializable objects like numpy arrays to serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def extract_min_max_angles(patterns):
    """Extract the minimum and maximum two-theta angles from the given patterns."""
    all_angles = []
    for pattern in patterns:
        all_angles.extend(pattern['two_theta_angles'])
    return min(all_angles), max(all_angles)


def extract_min_max_angles_from_folder(folder_path):
    """Extract the minimum and maximum two-theta angles from spectra files in the given folder."""  # noqa: E501
    all_angles = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.xy'):
            with open(os.path.join(folder_path, filename)) as f:
                for line in f:
                    try:
                        angle = float(line.split()[0])
                        all_angles.append(angle)
                    except ValueError:
                        continue  # Skip lines that do not contain valid data
    return min(all_angles), max(all_angles)


@dataclass
class AnalysisSettings:
    structure_references_directory: Optional[str] = (
        None  # Defaults to the directory containing the CIF files from the archive if not provided  # noqa: E501
    )
    patterns_folder_directory: str = 'temp_patterns'
    xrd_model: Optional[str] = (
        None  # Defaults to the XRD model file from the archive if not provided
    )
    pdf_model: Optional[str] = None
    max_phases: int = 5
    cutoff_intensity: float = 0.05
    min_confidence: float = 10.0
    wavelength: str = 'CuKa'
    unknown_threshold: float = 0.2
    show_reduced: bool = False
    include_pdf: bool = False
    parallel: bool = False
    raw: bool = False
    show_individual: bool = False
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None


@dataclass
class AnalysisResult:
    filenames: list
    phases: list
    confidences: list
    backup_phases: list
    scale_factors: list
    reduced_spectra: list

    def to_dict(self):
        return {
            'filenames': self.filenames,
            'phases': self.phases,
            'confidences': self.confidences,
            'backup_phases': self.backup_phases,
            'scale_factors': self.scale_factors,
            'reduced_spectra': self.reduced_spectra,
        }


def analyze_pattern(  # noqa: PLR0912, PLR0915
    settings: AnalysisSettings, archive: 'EntryArchive', logger: Optional['BoundLogger']
) -> None:
    references_folder = (
        os.path.dirname(archive['data']['cif_files'][0])
        if settings.structure_references_directory is None
        else settings.structure_references_directory
    )
    spectra_folder = settings.patterns_folder_directory

    start = time.time()

    # Check for spectra
    if not os.path.exists(spectra_folder) or len(os.listdir(spectra_folder)) == 0:
        if logger:
            logger.error(
                f'Please provide at least one pattern in the {spectra_folder} directory.'  # noqa: E501
            )
        else:
            print(
                f'ERROR: Please provide at least one pattern in the {spectra_folder} directory.'  # noqa: E501
            )
        return

    results = {'XRD': None, 'PDF': None, 'Merged': None}

    # XRD/PDF ensemble requires all predictions
    final_conf = settings.min_confidence
    if settings.include_pdf:
        settings.min_confidence = 10.0

    model_path = (
        settings.xrd_model if settings.xrd_model else archive['data']['xrd_model_file']
    )

    # Get predictions from XRD analysis
    results['XRD'] = AnalysisResult(
        *spectrum_analysis.main(
            spectra_folder,
            references_folder,
            settings.max_phases,
            settings.cutoff_intensity,
            settings.min_confidence,
            settings.wavelength,
            settings.min_angle,
            settings.max_angle,
            settings.parallel,
            model_path,
            is_pdf=False,
        )
    )

    if settings.include_pdf and settings.pdf_model:
        # Get predictions from PDF analysis
        model_path = settings.pdf_model
        results['PDF'] = AnalysisResult(
            *spectrum_analysis.main(
                spectra_folder,
                references_folder,
                settings.max_phases,
                settings.cutoff_intensity,
                settings.min_confidence,
                settings.wavelength,
                settings.min_angle,
                settings.max_angle,
                settings.parallel,
                model_path,
                is_pdf=True,
            )
        )

        # Merge results
        results['Merged'] = AnalysisResult(
            *spectrum_analysis.merge_results(
                {'XRD': results['XRD'].to_dict(), 'PDF': results['PDF'].to_dict()},
                final_conf,
                settings.max_phases,
            )
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
            results['Merged'].filenames,
            results['Merged'].phases,
            results['Merged'].confidences,
            results['Merged'].backup_phases,
            results['Merged'].scale_factors,
            results['Merged'].reduced_spectra,
        )
    ):
        # Display phase ID info
        if logger:
            logger.info(f'Filename: {spectrum_fname}')
            logger.info(f'Predicted phases: {phase_set}')
            logger.info(f'Confidence: {confidence}')
        else:
            print(f'Filename: {spectrum_fname}')
            print(f'Predicted phases: {phase_set}')
            print(f'Confidence: {confidence}')

        # Check for unknown peaks
        if phase_set and 'None' not in phase_set:
            remaining_I = max(final_spectrum)
            if remaining_I > settings.unknown_threshold:
                if logger:
                    logger.warning(
                        f'some peaks (I ~ {int(remaining_I)}%) were not identified.'
                    )
                else:
                    print(
                        f'WARNING: some peaks (I ~ {int(remaining_I)}%) were not identified.'  # noqa: E501
                    )
        else:
            if logger:
                logger.warning('No phases were identified')
            else:
                print('WARNING: No phases were identified')
            continue

        # Show backup predictions
        if settings.show_individual:
            if logger:
                logger.info(f'XRD predicted phases: {results["XRD"].phases[idx]}')
                logger.info(f'XRD confidence: {results["XRD"].confidences[idx]}')
                if settings.include_pdf:
                    logger.info(f'PDF predicted phases: {results["PDF"].phases[idx]}')
                    logger.info(f'PDF confidence: {results["PDF"].confidences[idx]}')
            else:
                print(f'XRD predicted phases: {results["XRD"].phases[idx]}')
                print(f'XRD confidence: {results["XRD"].confidences[idx]}')
                if settings.include_pdf:
                    print(f'PDF predicted phases: {results["PDF"].phases[idx]}')
                    print(f'PDF confidence: {results["PDF"].confidences[idx]}')

        # Plot the results
        phasenames = [f'{phase}.cif' for phase in phase_set]
        visualizer.main(
            spectra_folder,
            spectrum_fname,
            phasenames,
            heights,
            final_spectrum,
            settings.min_angle,
            settings.max_angle,
            settings.wavelength,
            save=False,
            show_reduced=settings.show_reduced,
            inc_pdf=settings.include_pdf,
            plot_both=False,
            raw=settings.raw,
        )

    end = time.time()
    if logger:
        logger.info(f'Total time: {round(end - start, 1)} sec')
    else:
        print(f'Total time: {round(end - start, 1)} sec')

    # Convert results to a JSON serializable format
    serializable_results = convert_to_serializable(results['Merged'].to_dict())

    # Save the results dictionary as a JSON file
    results_file = 'results.json'
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    if logger:
        logger.info(f'Results saved to {results_file}')
    else:
        print(f'Results saved to {results_file}')


def run_analysis_with_patterns_archive(
    settings: AnalysisSettings,
    model_archive_path: str,
    patterns_archive_path: str,
    results_file: str,
) -> None:
    # Load the NOMAD archive data for the model metadata
    with open(model_archive_path) as file:
        archive_data = json.load(file)

    # Load the NOMAD archive data for the patterns
    with open(patterns_archive_path) as file:
        patterns_data = json.load(file)

    # Extract patterns data from the patterns archive
    patterns = patterns_data['archive']['results']['properties']['structural'][
        'diffraction_pattern'
    ]

    # If the provided spectra folder already exists and has files, ask whether to overwrite or use existing files  # noqa: E501
    if (
        os.path.exists(settings.patterns_folder_directory)
        and len(os.listdir(settings.patterns_folder_directory)) > 0
    ):
        user_choice = (
            input(
                f"The folder '{settings.patterns_folder_directory}' already contains spectra files. Do you want to (o)verwrite or (u)se existing files? (o/u): "  # noqa: E501
            )
            .strip()
            .lower()
        )
        if user_choice == 'o':
            # Overwrite: Clear the folder and write new spectra files
            for filename in os.listdir(settings.patterns_folder_directory):
                if filename.endswith('.xy'):
                    os.remove(
                        os.path.join(settings.patterns_folder_directory, filename)
                    )
            settings.min_angle, settings.max_angle = extract_min_max_angles(patterns)
            for i, pattern in enumerate(patterns):
                with open(
                    f'{settings.patterns_folder_directory}/spectrum_{i}.xy', 'w'
                ) as f:
                    two_theta_angles = pattern['two_theta_angles']
                    intensities = pattern['intensity']
                    for angle, intensity in zip(two_theta_angles, intensities):
                        f.write(f'{angle} {intensity}\n')
        elif user_choice == 'u':
            # Use existing files: Recalculate min and max angles from existing spectra
            settings.min_angle, settings.max_angle = extract_min_max_angles_from_folder(
                settings.patterns_folder_directory
            )
        else:
            print(
                "Invalid choice. Please enter 'o' to overwrite or 'u' to use existing files."  # noqa: E501
            )
            return
    else:
        # Otherwise, create spectra files from the archive and calculate min and max angles  # noqa: E501
        settings.min_angle, settings.max_angle = extract_min_max_angles(patterns)
        os.makedirs(settings.patterns_folder_directory, exist_ok=True)
        for i, pattern in enumerate(patterns):
            with open(
                f'{settings.patterns_folder_directory}/spectrum_{i}.xy', 'w'
            ) as f:
                two_theta_angles = pattern['two_theta_angles']
                intensities = pattern['intensity']
                for angle, intensity in zip(two_theta_angles, intensities):
                    f.write(f'{angle} {intensity}\n')

    # Extract settings from the archive data if needed
    archive = archive_data  # Placeholder for the actual archive loading process

    # Run analysis
    analyze_pattern(
        settings, archive, logger=None
    )  # Replace 'logger=None' with an actual logger instance if available


def run_analysis_existing_spectra(
    settings: AnalysisSettings, model_archive_path: str, results_file: str
) -> None:
    # Load the NOMAD archive data for the model metadata
    with open(model_archive_path) as file:
        archive_data = json.load(file)

    # Extract settings from the archive data if needed
    archive = archive_data  # Placeholder for the actual archive loading process

    # Run analysis with existing spectra files in `settings.patterns_folder_directory`
    settings.min_angle, settings.max_angle = extract_min_max_angles_from_folder(
        settings.patterns_folder_directory
    )

    # Run analysis
    analyze_pattern(
        settings, archive, logger=None
    )  # Replace 'logger=None' with an actual logger instance if available


# Example usage
if __name__ == '__main__':
    settings = AnalysisSettings(
        structure_references_directory='References',
        patterns_folder_directory='temp_patterns',
        xrd_model='path/to/xrd_model.h5',
        max_phases=5,
        min_confidence=30,
    )

    # Choose either to run with existing spectra files or to create from a patterns archive  # noqa: E501
    run_type = input(
        'Run analysis with (1) existing spectra files or (2) create from patterns archive? (1/2): '  # noqa: E501
    ).strip()

    if run_type == '1':
        run_analysis_existing_spectra(
            settings, 'model_metadata.archive.json', 'custom_results.json'
        )
    elif run_type == '2':
        run_analysis_with_patterns_archive(
            settings,
            'model_metadata.archive.json',
            'patterns_archive.json',
            'custom_results.json',
        )
    else:
        print("Invalid choice. Please select either '1' or '2'.")
