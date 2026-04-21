"""
src/data/dataset.py

PyTorch Dataset for Anti-UAV v4 Track 3.
Loads frames from video files on the fly and pairs them
with their corresponding YOLO-format labels.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np


class VideoDataset(Dataset):
    """
    Iterates over all (frame, label) pairs across all sequences
    in the processed Anti-UAV v4 dataset directory.

    Each item returned is:
        image  : torch.Tensor of shape (C, H, W), float32, range [0, 1]
        labels : torch.Tensor of shape (N, 5)  — N boxes, each (class, cx, cy, w, h)
                 Empty tensor of shape (0, 5) if no UAVs in frame
        meta   : dict with sequence name and frame_id for debugging
    """

    def __init__(self, processed_dir: str | Path, videos_dir: str | Path, transform=None):
        self.processed_dir = Path(processed_dir)
        self.videos_dir    = Path(videos_dir)
        self.transform     = transform

        self.samples = []  # list of (video_path, label_path, frame_id, seq_name)

        self._build_index()

    def _build_index(self):
        """
        Walks processed_dir for each sequence folder, then for each
        label .txt file finds the matching video and records a sample tuple.
        Sequences are sorted for reproducibility.
        """
        seq_dirs = sorted(self.processed_dir.iterdir())

        for seq_dir in seq_dirs:
            if not seq_dir.is_dir():
                continue

            seq_name  = seq_dir.name  # e.g. "MultiUAV-002"
            video_path = self.videos_dir / f"{seq_name}.mp4"

            if not video_path.exists():
                print(f"  [WARN] No video found for {seq_name}, skipping.")
                continue

            # Each .txt file corresponds to one frame
            label_files = sorted(seq_dir.glob("*.txt"))

            for label_path in label_files:
                # Frame id is encoded in the filename e.g. 000001.txt -> 1
                frame_id = int(label_path.stem)
                self.samples.append((video_path, label_path, frame_id, seq_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label_path, frame_id, seq_name = self.samples[idx]

        # --- Load frame from video ---
        cap = cv2.VideoCapture(str(video_path))
        # Seek directly to the target frame (0-indexed)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")

        # Convert BGR (OpenCV default) to RGB, then to float32 in [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0

        # --- Load YOLO labels ---
        label_text = label_path.read_text().strip()

        if label_text:
            rows = []
            for line in label_text.splitlines():
                parts = line.split()
                rows.append([float(x) for x in parts])
            labels = torch.tensor(rows, dtype=torch.float32)  # (N, 5)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)  # empty frame

        # --- Apply augmentations if provided ---
        if self.transform is not None:
            frame = self.transform(image=frame)["image"]

        # Convert HWC numpy array to CHW torch tensor
        image = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W)

        meta = {"seq_name": seq_name, "frame_id": frame_id}

        return image, labels, meta
