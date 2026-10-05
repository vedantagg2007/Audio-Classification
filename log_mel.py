import librosa
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 16000

CROP_DURATION = 2
CROP_SAMPLES = SAMPLE_RATE * CROP_DURATION

TOP_K = 3
#top k is 3
FRAME_SIZE = 1024
HOP_LENGTH = 512
NUM_MELS = 128

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "topk_data"

MODEL_PATH = (
    BASE_DIR / "best_models/best_log_mel_topk_model.pth"
)

CLASS_NAMES = [
    "jackhammer",
    "siren",
    "chainsaw"
]

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)


def hz_to_mel(hz):

    return (
        2595
        * torch.log10(
            1 + hz / 700
        )
    )


def mel_to_hz(mel):

    return (
        700
        * (
            10 ** (mel / 2595)
            - 1
        )
    )


def create_mel_filter_bank():

    num_fft_bins = (
        FRAME_SIZE // 2 + 1
    )

    fft_frequencies = torch.linspace(
        0,
        SAMPLE_RATE / 2,
        num_fft_bins
    )

    min_mel = hz_to_mel(
        torch.tensor(0.0)
    )

    max_mel = hz_to_mel(
        torch.tensor(
            SAMPLE_RATE / 2.0
        )
    )

    mel_points = torch.linspace(
        min_mel,
        max_mel,
        NUM_MELS + 2
    )

    hz_points = mel_to_hz(
        mel_points
    )

    filters = []

    for i in range(NUM_MELS):

        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        rising = (
            (fft_frequencies - left)
            / (center - left)
        )

        falling = (
            (right - fft_frequencies)
            / (right - center)
        )

        filter_values = torch.clamp(
            torch.minimum(
                rising,
                falling
            ),
            min=0
        )

        filters.append(
            filter_values
        )

    return torch.stack(
        filters
    )


MEL_FILTER_BANK = (
    create_mel_filter_bank()
)


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

    # Each saved crop should already be 2 seconds,
    # but this guarantees consistent size.

    if len(audio) < CROP_SAMPLES:

        audio = nn.functional.pad(
            audio,
            (
                0,
                CROP_SAMPLES - len(audio)
            )
        )

    else:

        audio = audio[:CROP_SAMPLES]

    return audio


def extract_log_mel(audio):

    window = torch.hann_window(
        FRAME_SIZE
    )

    # STFT
    stft = torch.stft(
        audio,
        n_fft=FRAME_SIZE,
        hop_length=HOP_LENGTH,
        win_length=FRAME_SIZE,
        window=window,
        center=False,
        return_complex=True
    )

    # Power spectrogram
    power = (
        torch.abs(stft) ** 2
    )

    # Apply mel filter bank
    mel = torch.matmul(
        MEL_FILTER_BANK,
        power
    )

    # Log scaling
    features = torch.log1p(
        mel
    )

    # Normalize each crop
    features = (
        features
        - features.mean()
    ) / (
        features.std()
        + 1e-6
    )

    # Shape:
    # (1, 128, time_frames)

    return features.unsqueeze(0)



class TopKLogMelDataset(Dataset):

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

            # Each subfolder corresponds to
            # one original 10-second recording.
            recording_folders = sorted(
                [
                    folder
                    for folder
                    in class_folder.iterdir()
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
                    f"Missing crop: "
                    f"{crop_path}"
                )

            audio = load_crop(
                crop_path
            )

            features = extract_log_mel(
                audio
            )

            crop_features.append(
                features
            )

        # Shape:
        #
        # (3, 1, 128, time_frames)

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




class LogMelTopKCNN(nn.Module):

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
            3
        )

    def forward(self, x):

        # Input shape:
        #
        # batch,
        # crops,
        # channel,
        # mel,
        # time

        batch_size = x.shape[0]
        num_crops = x.shape[1]

        # Merge batch and crop dimensions
        #
        # Example:
        #
        # (8, 3, 1, 128, 61)
        #
        # becomes
        #
        # (24, 1, 128, 61)

        x = x.view(
            batch_size * num_crops,
            x.shape[2],
            x.shape[3],
            x.shape[4]
        )

        # Run every crop through
        # the same CNN
        x = self.features(
            x
        )

        x = x.flatten(
            start_dim=1
        )

        logits = self.classifier(
            x
        )

        # Restore crop grouping
        #
        # (24, 3)
        #
        # becomes
        #
        # (8, 3, 3)

        logits = logits.view(
            batch_size,
            num_crops,
            len(CLASS_NAMES)
        )

        # Average the predictions
        # from the three crops
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



    train_dataset = TopKLogMelDataset(
        "train"
    )

    val_dataset = TopKLogMelDataset(
        "validation"
    )

    test_dataset = TopKLogMelDataset(
        "test"
    )

    if (
        len(train_dataset) == 0
        or len(val_dataset) == 0
        or len(test_dataset) == 0
    ):

        raise RuntimeError(
            "One or more dataset splits "
            "are empty."
        )


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



    model = LogMelTopKCNN().to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float(
        "inf"
    )



    print(
        "\nStarting top-k log-mel training..."
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
            f"Val Accuracy: "
            f"{val_accuracy:.2%}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

    print(
        "\nTraining complete!"
    )

    print(
        "Best model saved:",
        MODEL_PATH
    )



    model.load_state_dict(
        torch.load(
            MODEL_PATH,
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
