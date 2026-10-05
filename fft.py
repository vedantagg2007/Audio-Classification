import librosa
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

#also this uses the epoch with the lowest validation loss

SAMPLE_RATE = 16000 #16khz

CROP_DURATION = 2 #cropped samples after top k, and then averaged
NUM_SAMPLES = SAMPLE_RATE * CROP_DURATION

TOP_K = 3 #amount of samples per recording

FRAME_SIZE = 1024 #how many raw audio samples go into each STFT chunk (1024/16000 = 0.064 seconds per chunk)
HOP_LENGTH = 512 #hop length is 32ms to create overlap so that each chunk doesn't miss boundary events
NUM_FREQUENCIES = 256 #16000/1024 gives 15.625 Hz. Using 256 instead of 1024 frequecies focuses on lower frequency content

BATCH_SIZE = 8 #trains on 8 recordings before updating weights
EPOCHS = 20 #1200/8 is about 150 batches per epoch
LEARNING_RATE = 0.001 #how much the weights change after each batch

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "topk_data"

MODEL_PATH = BASE_DIR / "best_models/best_stft_topk_model.pth"

CLASS_NAMES = [
    "jackhammer",
    "siren",
    "chainsaw"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)

def load_crop(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = torch.tensor(
        audio,
        dtype=torch.float32
    )

    # Guarantee exactly 2 seconds
    if len(audio) < NUM_SAMPLES:

        audio = nn.functional.pad(
            audio,
            (
                0,
                NUM_SAMPLES - len(audio)
            )
        )

    else:

        audio = audio[:NUM_SAMPLES]

    return audio


def extract_stft(audio):

    # Divide 2-second waveform into overlapping frames
    frames = audio.unfold(
        dimension=0,
        size=FRAME_SIZE,
        step=HOP_LENGTH
    )

    # Hann window
    window = torch.hann_window(
        FRAME_SIZE
    )

    frames = frames * window

    # FFT of each short frame
    fft = torch.fft.rfft(
        frames,
        dim=-1
    )

    # Magnitude
    magnitude = torch.abs(
        fft
    )

    # Keep first 256 frequency bins
    magnitude = magnitude[
        :,
        :NUM_FREQUENCIES
    ]

    # Log scaling
    features = torch.log1p(
        magnitude
    )

    # Normalize each crop
    features = (
        features
        - features.mean()
    ) / (
        features.std()
        + 1e-6
    )

    # Before transpose:
    # (time_frames, 256)
    #
    # After transpose:
    # (256, time_frames)
    #
    # Final:
    # (1, 256, time_frames)

    return features.T.unsqueeze(0)




class STFTDataset(Dataset):

    def __init__(self, split):

        self.recording_folders = []
        self.labels = []

        print(
            f"\nLoading {split} recordings..."
        )

        for label, class_name in enumerate(
            CLASS_NAMES
        ):

            class_folder = (
                DATA_DIR
                / split
                / class_name
            )

            recording_folders = sorted(
                [
                    folder
                    for folder in class_folder.iterdir()
                    if folder.is_dir()
                ]
            )

            print(
                f"{class_name}: "
                f"{len(recording_folders)} recordings"
            )

            for folder in recording_folders:

                self.recording_folders.append(
                    folder
                )

                self.labels.append(
                    label
                )

    def __len__(self):

        return len(
            self.recording_folders
        )

    def __getitem__(self, index):

        folder = (
            self.recording_folders[index]
        )

        crop_features = []

        # Load:
        #
        # crop_1.wav
        # crop_2.wav
        # crop_3.wav

        for crop_number in range(
            1,
            TOP_K + 1
        ):

            crop_path = (
                folder
                / f"crop_{crop_number}.wav"
            )

            if not crop_path.exists():

                raise FileNotFoundError(
                    f"Missing crop: {crop_path}"
                )

            audio = load_crop(
                crop_path
            )

            features = extract_stft(
                audio
            )

            crop_features.append(
                features
            )

        # Each crop:
        #
        # (1, 256, 61)
        #
        # Three crops:
        #
        # (3, 1, 256, 61)

        crop_features = torch.stack(
            crop_features
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return (
            crop_features,
            label
        )



class STFTCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(
                (4, 4)
            )
        )

        self.classifier = nn.Linear(
            64 * 4 * 4,
            len(CLASS_NAMES)
        )

    def forward(self, x):

        # Input:
        #
        # (batch, crops, channel, frequency, time)
        #
        # Example:
        #
        # (8, 3, 1, 256, 61)

        batch_size = x.shape[0]
        num_crops = x.shape[1]

        # Merge batch and crop dimensions
        #
        # (8, 3, 1, 256, 61)
        #
        # becomes
        #
        # (24, 1, 256, 61)

        x = x.view(
            batch_size * num_crops,
            x.shape[2],
            x.shape[3],
            x.shape[4]
        )

        # Run every crop through same CNN
        x = self.features(
            x
        )

        x = x.flatten(
            start_dim=1
        )

        logits = self.classifier(
            x
        )

        # Current:
        #
        # (24, 3)
        #
        # Restore crop grouping:
        #
        # (8, 3, 3)

        logits = logits.view(
            batch_size,
            num_crops,
            len(CLASS_NAMES)
        )

        # Average the three crop predictions
        #
        # (8, 3, 3)
        # ->
        # (8, 3)

        logits = logits.mean(
            dim=1
        )

        return logits



def evaluate_model(
    model,
    loader
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                device
            )

            outputs = model(
                features
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    actual = np.array(
        all_labels
    )

    predicted = np.array(
        all_predictions
    )

    accuracy = np.mean(
        actual == predicted
    )

    return (
        accuracy,
        actual,
        predicted
    )


if __name__ == "__main__":

    print(
        "Using device:",
        device
    )

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = STFTDataset(
        "train"
    )

    val_dataset = STFTDataset(
        "validation"
    )

    test_dataset = STFTDataset(
        "test"
    )

    if (
        len(train_dataset) == 0
        or len(val_dataset) == 0
        or len(test_dataset) == 0
    ):

        raise RuntimeError(
            "One or more dataset splits are empty."
        )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        "\nTraining recordings:",
        len(train_dataset)
    )

    print(
        "Validation recordings:",
        len(val_dataset)
    )

    print(
        "Testing recordings:",
        len(test_dataset)
    )

    # -----------------------------------------------------
    # Verify input shape
    # -----------------------------------------------------

    features, labels = next(
        iter(train_loader)
    )

    print(
        "\nBatch feature shape:",
        features.shape
    )

    print(
        "Batch label shape:",
        labels.shape
    )

    # Should be approximately:
    #
    # torch.Size([8, 3, 1, 256, 61])


    model = STFTCNN().to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float(
        "inf"
    )


    print(
        "\nStarting STFT top-k training..."
    )

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0

        for features, labels in train_loader:

            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
                * len(labels)
            )

        train_loss = (
            total_train_loss
            / len(train_dataset)
        )



        model.eval()

        total_val_loss = 0
        val_correct = 0

        with torch.no_grad():

            for features, labels in val_loader:

                features = features.to(
                    device
                )

                labels = labels.to(
                    device
                )

                outputs = model(
                    features
                )

                loss = criterion(
                    outputs,
                    labels
                )

                total_val_loss += (
                    loss.item()
                    * len(labels)
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    predictions
                    == labels
                ).sum().item()

        val_loss = (
            total_val_loss
            / len(val_dataset)
        )

        val_accuracy = (
            val_correct
            / len(val_dataset)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2%}"
        )

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )


    print(
        "\nTraining complete"
    )

    print(
        "Best model saved:",
        MODEL_PATH
    )



    model.load_state_dict(
        torch.load(
            MODEL_PATH, #replace model path with best_model.py or whatever
            map_location=device,
            weights_only=True
        )
    )

    model.eval()



    accuracy, actual, predicted = (
        evaluate_model(
            model,
            test_loader
        )
    )

    print(
        f"\nTest Accuracy: "
        f"{accuracy:.2%}"
    )




    num_classes = len(
        CLASS_NAMES
    )

    cm = np.zeros(
        (
            num_classes,
            num_classes
        ),
        dtype=int
    )

    for actual_label, predicted_label in zip(
        actual,
        predicted
    ):

        cm[
            actual_label,
            predicted_label
        ] += 1

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "\nClass order:",
        CLASS_NAMES
    )


    print(
        "\nClassification Report:"
    )

    print(
        f"{'Class':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
        f"{'Support':>10}"
    )

    f1_scores = []

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        tp = cm[i, i]

        fp = (
            cm[:, i].sum()
            - tp
        )

        fn = (
            cm[i, :].sum()
            - tp
        )

        support = (
            cm[i, :].sum()
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if precision + recall > 0
            else 0
        )

        f1_scores.append(
            f1
        )

        print(
            f"{class_name:<15}"
            f"{precision:>12.2f}"
            f"{recall:>12.2f}"
            f"{f1:>12.2f}"
            f"{support:>10}"
        )


    macro_f1 = np.mean(
        f1_scores
    )

    print(
        f"\nMacro F1-score: "
        f"{macro_f1:.4f}"
    )
