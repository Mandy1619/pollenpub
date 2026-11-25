# 6. Reproduce it yourself

[← Evaluation](05-evaluation.md) · [Index](README.md) · [Next: Code guide →](07-code-guide.md)

This page takes you from a fresh clone to the results in [chapter 5](05-evaluation.md). No GPU is needed.

## Requirements

| | Minimum | Tested with |
|---|---|---|
| OS | Linux (macOS likely works; Windows untested) | Ubuntu, kernel 7.0 |
| Python | 3.10+ | 3.12 and 3.14 |
| PyTorch | 2.2+ | 2.15 (CPU build) |
| RAM | 8 GB | 12 GB |
| Disk | about 2.5 GB (venv 1.5 GB, weights 0.4 GB, test data 0.6 GB) | |
| Time | about 45 minutes in total on a 4-core CPU | |

## Step 1: Get the code

```bash
git clone https://github.com/Mandy1619/pollenpub.git
cd pollenpub
```

## Step 2: Create the Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r code/requirements.txt
```

The `--index-url` line installs the CPU-only build of PyTorch (about 200 MB instead of several GB of CUDA libraries). If you have an NVIDIA GPU, leave out `--index-url ...`; all scripts use the GPU automatically when it is available.

Check it:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, 'CUDA' if torch.cuda.is_available() else 'CPU')"
```

## Step 3: Download weights and test data

All following commands run from the `code/` folder:

```bash
cd code
../.venv/bin/python utils/download.py -f weights.zip                         # 390 MB → ../weights/
../.venv/bin/python utils/download.py -f data.zip --only pollen_20190523     # 626 MB → ../data/pollen_20190523/
```

The downloader fetches from Zenodo with HTTP range requests. It resumes if the connection drops, and with `--only` it fetches just the part of the 1.6 GB archive you need. To get everything (both days, needed for training), run `../.venv/bin/python utils/download.py` without arguments.

Check it: `ls ../data/pollen_20190523/layers | wc -l` should print `3769`.

## Step 4: Evaluate on the test set

```bash
../.venv/bin/python test.py \
    --weights_path ../weights/pollen_20190526.pth \
    --model_def config/yolov3-pollen.cfg \
    --data_config config/test_20190523.data \
    --batch_size 8
```

This takes about 30 minutes on a 4-core CPU, or a minute or two on a GPU. The end of the output should match:

```
Number of targets removed tensor(322)
True positives 2194.0
Average Precisions:
+ Class '0' (pollen) - AP: 0.8991997839678553
+ Class '1' (maybe) - AP: 0.0
mAP: 0.44959989198392764
F1 [0.87462627 0.        ]
```

Small differences in the last digits are possible on other hardware or library versions.

**Tip: try a small subset first.** To check that everything works in about 2 minutes, evaluate only the first 48 layers:

```bash
head -48 ../data/pollen_20190523/test.txt > /tmp/tiny.txt
sed "s#^valid=.*#valid=/tmp/tiny.txt#" config/test_20190523.data > /tmp/tiny.data
../.venv/bin/python test.py --data_config /tmp/tiny.data --batch_size 8
```

(The defaults of `--weights_path` and `--model_def` already point to the published model.) On this subset the pollen AP is about 0.73; such a small sample says little about the real score.

**Long runs:** to keep the evaluation running after you close the terminal, start it with `nohup ... > ../output/test.log 2>&1 &` and follow progress with `tail -f ../output/test.log`.

## Step 5: Look at detections

```bash
../.venv/bin/python detect.py --image_folder ../data/pollen_20190523/layers/ --output_folder ../output/detections/
```

This writes one PNG with boxes per input image. For a quick look, copy a few images into a separate folder and point `--image_folder` at it.

## Step 6 (optional, GPU): Train

```bash
../.venv/bin/python utils/download.py -f data.zip --only pollen_20190526
../.venv/bin/python train.py --name fold0 --epochs 60 \
    --model_def config/yolov3-pollen.cfg \
    --data_config config/train_20190526fold0.data \
    --pretrained_weights ../weights/darknet53.conv.74
```

For a quick check that training runs at all (also on CPU): add `--epochs 1 --max_batches 5`. See [chapter 4](04-training.md) for details.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| Evaluation becomes extremely slow after the first batch; the computer freezes | Out of memory, the system is swapping. Use `--batch_size 8` or lower (the default of 16 needs more RAM than 12 GB machines have to spare). |
| `No space left on device` during download | The old download script stored the whole zip and then extracted it (needs 2×). Use the current `utils/download.py` with `--only`. |
| `429 Too Many Requests` from Zenodo | Rate limiting. Wait a minute and re-run the same command; already extracted files are skipped. |
| `FileNotFoundError: ../data/...` | Commands must run from inside `code/`; all paths in the configs are relative to it. |
| `conda env create -f environment.yml` fails | That file pins Python 3.7 / CUDA 10.1 from 2019 and no longer resolves. Use the pip setup above. |
| `ModuleNotFoundError: tensorboard` | Optional. Training logs are always written to `logs/<name>/metrics.csv`; install `tensorboard` if you want the TensorBoard view. |
