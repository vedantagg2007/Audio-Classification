import pandas as pd
import shutil #to copy the wav files
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "annotations.csv")

source = BASE_DIR / "audio"
destination = BASE_DIR / "filtered_audio"

#the column names in annotations.csv that indicate if something is present
classes = {
    "jackhammer": "2-2_jackhammer_presence",
    "siren": "5-3_siren_presence",
    "chainsaw": "4-1_chainsaw_presence"
}

# This finds all the WAV files
audio_files = {
    p.name: p for p in source.rglob("*.wav")
}

counts = Counter()
missing = 0
#This will keep track of how many recordings are mentioned in annotations that I haven't downloaded yet


# This groups all of the annotations by the recording recording
groups = df.groupby(["split", "audio_filename"])

for (split, filename), group in groups:

    if filename not in audio_files:
        missing += 1
        continue

    if split == "validate":
        split = "validation"

    # Check each target sound
    for name, column in classes.items():

        # Keep if at least one annotation is positive
        if (group[column] == 1).any():

            folder = destination / split / name
            folder.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                audio_files[filename],
                folder / filename
            )

            counts[(split, name)] += 1

# Print results
for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()}")

    for name in classes:
        print(f"{name}: {counts[(split, name)]}")

print(f"\nMissing recordings: {missing}")
print("Filtering complete!")
