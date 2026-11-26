"""Generate actual combined seed counts, using native white pixels for fixed seeds.

Primary counts are copied unchanged. Adjacent white pixels are separate seeds;
the >26 category belongs only to plotting and is never written as a count here.
Importing this module does not regenerate or overwrite metadata.
"""
import json
from pathlib import Path

from cldm.seed_counts import count_seed_pixels

from cldm.config import (
    EXP_FOLDER_TEST_FIXED,
    SEED_FOLDER_TEST_SEEDINFO,
    SEED_FOLDER_TEST_SEEDINFO_COMBINED,
    SIM_FOLDER_TEST_FIXED_INPUT,
)

def main():
    with open(SEED_FOLDER_TEST_SEEDINFO) as file:
        combined_counts = json.load(file)

    fixed_stems = sorted(Path(filename).stem for filename in Path(EXP_FOLDER_TEST_FIXED).iterdir()
                          if filename.suffix.lower() == '.tif')

    for stem in fixed_stems:
        seed_number = stem.split('_')[1]  # "Fixed_10_1" -> "10"
        seed_path = Path(SIM_FOLDER_TEST_FIXED_INPUT) / f'Input_Fixed_{seed_number}.png'
        seed_count = count_seed_pixels(seed_path)
        combined_counts[stem + '.png'] = int(seed_count)

    with open(SEED_FOLDER_TEST_SEEDINFO_COMBINED, 'w') as file:
        json.dump(combined_counts, file, indent=2)

    print(f'{len(combined_counts)} total seed-count entries written to {SEED_FOLDER_TEST_SEEDINFO_COMBINED}')
    print(f'({len(fixed_stems)} Fixed_N_M stems added)')


if __name__ == "__main__":
    main()
