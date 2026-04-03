"""
scripts/install_pytorch.py

Run this once after cloning the repo on any new machine.
Detects available hardware and installs the correct PyTorch build.

Usage:
    python scripts/install_pytorch.py
"""

import subprocess
import sys


def get_cuda_version():
    """Returns CUDA major.minor as a string, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "release" in line:
                # e.g. "Cuda compilation tools, release 12.1, V12.1.105"
                version = line.split("release")[-1].strip().split(",")[0].strip()
                return version
    except FileNotFoundError:
        pass

    # nvcc not found — try nvidia-smi as fallback
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "CUDA Version" in line:
                version = line.split("CUDA Version:")[-1].strip().split()[0]
                return version
    except FileNotFoundError:
        pass

    return None


def select_torch_index(cuda_version):
    """Maps CUDA version string to the correct PyTorch wheel index URL."""
    if cuda_version is None:
        return "cpu", "https://download.pytorch.org/whl/cpu"

    major, minor = int(cuda_version.split(".")[0]), int(cuda_version.split(".")[1])

    if major == 12 and minor >= 1:
        return "cu121", "https://download.pytorch.org/whl/cu121"
    elif major == 11 and minor >= 8:
        return "cu118", "https://download.pytorch.org/whl/cu118"
    else:
        print(f"  CUDA {cuda_version} detected but no matching wheel found.")
        print("  Falling back to CPU build.")
        return "cpu", "https://download.pytorch.org/whl/cpu"


def install_torch(index_url):
    packages = [
        "torch",
        "torchvision",
        "torchaudio",
    ]
    cmd = [
        sys.executable, "-m", "pip", "install",
        *packages,
        "--index-url", index_url,
        "--upgrade"
    ]
    subprocess.run(cmd, check=True)


def verify():
    """Quick sanity check after install."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print('PyTorch:', torch.__version__); "
         "print('CUDA available:', torch.cuda.is_available())"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Verification failed:")
        print(result.stderr)


def main():
    print("=" * 50)
    print("UAV Tracker — PyTorch Installer")
    print("=" * 50)

    cuda_version = get_cuda_version()

    if cuda_version:
        print(f"  CUDA detected: {cuda_version}")
    else:
        print("  No CUDA detected — installing CPU build")

    build, index_url = select_torch_index(cuda_version)
    print(f"  Selected build : {build}")
    print(f"  Index URL      : {index_url}")
    print()

    install_torch(index_url)

    print()
    print("Verifying installation...")
    verify()
    print("Done.")


if __name__ == "__main__":
    main()
