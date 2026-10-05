import librosa
import numpy as np
import soundfile as sf

from pathlib import Path


# =========================================================
# 1. CONFIGURATION
# =========================================================

SAMPLE_RATE = 16000

FULL_DURATION = 10
FULL_SAMPLES = SAMPLE_RATE * FULL_DURATION

CROP_DURATION = 2
CROP_SAMPLES = SAMPLE_RATE * CROP_DURATION

# Candidate crop starts every 1 second
CROP_HOP_SECONDS = 1
CROP_HOP_SAMPLES = SAMPLE_RATE * CROP_HOP_SECONDS

TOP_K = 3

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "balanced_data"
DEST_DIR = BASE_DIR / "topk_data"

CLASS_NAMES = [
    "jackhammer",
    "siren",
    "chainsaw"
]

SPLITS = [
    "train",
    "validation",
    "test"
]


# =========================================================
# 2. LOAD AUDIO
# =========================================================

def load_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Pad if shorter than 10 seconds
    if len(audio) < FULL_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                FULL_SAMPLES - len(audio)
            )
        )

    # Truncate if longer than 10 seconds
    else:

        audio = audio[:FULL_SAMPLES]

    return audio


# =========================================================
# 3. FIND TOP-K CROPS
# =========================================================

def get_top_k_crops(audio):

    candidates = []

    # Possible windows:
    #
    # 0-2 sec
    # 1-3 sec
    # 2-4 sec
    # ...
    # 8-10 sec

    for start in range(
        0,
        FULL_SAMPLES - CROP_SAMPLES + 1,
        CROP_HOP_SAMPLES
    ):

        end = start + CROP_SAMPLES

        crop = audio[start:end]

        # RMS energy
        rms = np.sqrt(
            np.mean(crop ** 2) + 1e-8
        )

        candidates.append(
            (
                rms,
                start,
                crop
            )
        )

    # Highest energy first
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = []

    for energy, start, crop in candidates:

        # Avoid nearly identical selected crops
        far_enough = True

        for selected_start, _, _ in selected:

            if abs(
                start - selected_start
            ) < CROP_HOP_SAMPLES:

                far_enough = False
                break

        if far_enough:

            selected.append(
                (
                    start,
                    energy,
                    crop
                )
            )

        if len(selected) == TOP_K:
            break

    return selected


# =========================================================
# 4. PROCESS ONE RECORDING
# =========================================================

def process_file(
    file_path,
    split,
    class_name
):

    audio = load_audio(
        file_path
    )

    crops = get_top_k_crops(
        audio
    )

    # One folder per original recording
    output_folder = (
        DEST_DIR
        / split
        / class_name
        / file_path.stem
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    for rank, (
        start,
        energy,
        crop
    ) in enumerate(
        crops,
        start=1
    ):

        start_seconds = (
            start / SAMPLE_RATE
        )

        output_path = (
            output_folder
            / f"crop_{rank}.wav"
        )

        sf.write(
            output_path,
            crop,
            SAMPLE_RATE
        )

        print(
            f"  crop {rank}: "
            f"start={start_seconds:.1f}s, "
            f"RMS={energy:.5f}"
        )


# =========================================================
# 5. PROCESS ENTIRE DATASET
# =========================================================

def main():

    # Prevent accidental mixing with old results
    if DEST_DIR.exists():

        raise FileExistsError(
            "topk_data already exists. "
            "Rename or delete it before running again."
        )

    total_files = 0

    for split in SPLITS:

        for class_name in CLASS_NAMES:

            folder = (
                SOURCE_DIR
                / split
                / class_name
            )

            files = sorted(
                folder.glob("*.wav")
            )

            print(
                f"\n{split}/{class_name}: "
                f"{len(files)} recordings"
            )

            for index, file_path in enumerate(
                files,
                start=1
            ):

                print(
                    f"\nProcessing "
                    f"{index}/{len(files)}: "
                    f"{file_path.name}"
                )

                process_file(
                    file_path,
                    split,
                    class_name
                )

                total_files += 1

    print(
        f"\nFinished processing "
        f"{total_files} recordings."
    )

    print(
        f"Top-k crops saved to:\n"
        f"{DEST_DIR}"
    )


# =========================================================
# 6. RUN
# =========================================================

if __name__ == "__main__":
    main()