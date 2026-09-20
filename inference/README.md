# ADCS Inference

Standalone document classification pipeline on [podder-task-foundation](https://github.com/podder-ai/podder-task-foundation).


## Layout

```text
.
|-- run.py
|-- requirements.txt
|-- config/
|   |-- pipeline.yaml
|   |-- long_document.yaml
|   |-- output.yaml
|   |-- accept_image/config.yaml
|   |-- orient/config.yaml
|   |-- ocr/config.yaml
|   `-- layoutlm/config.yaml
|-- weights/layoutlm/best_stage2.pt
|-- samples/example.png
`-- processes/
    |-- accept_image/process.py
    |-- orient/{process,orientation}.py
    |-- ocr/{process,ocr}.py
    `-- layoutlm/{process,infer,model,...}.py
```

## Setup

```bash
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/podder-ai/podder-task-foundation.git"
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd
```

Checkpoint: `weights/layoutlm/best_stage2.pt`

## Run

```bash
python run.py --image samples/resume_180.jpg
python run.py --image samples/resume_180.jpg -o samples/prediction_resume_180.json -v
```

# Inference pipeline

Four stages run in order (`config/pipeline.yaml`):

**accept_image => orient => ocr => layoutlm**

Everything hangs off a Podder payload: dictionaries. Each pipeline process has its own config, weights and pipline directory, so it easy to extend and maintain.


## 1. accept_image

Takes whatever path you passed (`--image`)

Output:  
```json
{
  "image_path": "samples/memo.jpg",
  "width": 804,
  "height": 1000
}
```


## 2. orient

Apply rotation in case the docs might be not correctly rotated (I have tried to fix the whole dataset but it takes a lot of time so this case I decided apply rotation in case low confidance):

Output:
```json
{
  "rotate_degrees": 0,
  "detected_orientation": 0,
  "confidence": 1.22,
  "backend": "osd_skipped_low_conf",
  "script": "Latin",
  "raw": { "...": "whatever OSD returned" },
  "min_osd_confidence": 1.5,
  "applied_rotation": 0
}
```

## 3. ocr

Tesseract provide output of words and boxes:

Output:
```json
{
  "n_words": 130,
  "width": 804,
  "height": 1000,
  "box_space": "pixel",
  "engine": "pytesseract",
  "words": ["John", "Doe"],
  "boxes": [[12.0, 40.0, 80.0, 62.0], [90.0, 40.0, 140.0, 62.0]]
}
```


## 4. layoutlm

Stage-2 LayoutLMv3 multi-label head. Weights: `weights/layoutlm/best_stage2.pt`.

Output:
```json
{
  "labels": ["resume"],
  "scores": {
    "letter": 0.0589,
    "form": 0.0838,
    "email": 0.0297,
    "resume": 0.5596
  },
  "is_none": false,
  "checkpoint": "weights/layoutlm/best_stage2.pt",
  "stage": 2,
  "device": "cuda",
  "n_ocr_words": 130
}
```

Anything with score ≥ `label_threshold` (0.5) goes into `labels`. Empty list ⇒ `is_none: true` (none of letter/form/email/resume).
