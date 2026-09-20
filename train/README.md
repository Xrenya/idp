# ADCS training

Training **Stage 1**/**Stage 2** LayoutLMv3

## Layout

```
config.yaml
requirements.txt
data/  can be downloaded from hugging face: Xrenya/rvl_ocr
outputs/
src/
  train.py
  model.py
  prepared_dataset.py
  losses.py
  metrics.py
  transforms_albu.py
  ocr.py
  quantize.py
scripts/
  quantize_and_benchmark.py
```

## Setup

Python >=3.10 & CUDA

```bash
python -m pip install -r requirements.txt
```

Put parquet shards under `data/` (can be downloaded from HuggingFace: `Xrenya/rvl_ocr`).
I have made some preprocessing for dataset but only training so I have splitted it later to 80/20 of `train_parquet` for training and validation, I have made preprocessing but I could not handle all rotation since it would take for 20-30 hours. I have rotated image with threshold but there are still left not rotated so I trained on them anyway. The pipeline I have fix that part in order to select the most suitable side. I have tried another models besides tesseract but it would take for me to process it about 60 hours, so I did what I could. 


## Stage 1 - finetuning encoder

Single GPU:

```bash
python -m src.train --config config.yaml --stage 1 --output-dir outputs/layoutlmv3
```

Multiple GPUs (I had some issue with my 2 GPUs so not sure whether it would work 100% because I switch into single GPU training):

```bash
python -m torch.distributed.run --nproc_per_node=2 --module src.train --config config.yaml --stage 1 --output-dir outputs/layoutlmv3
```

Writes `outputs/layoutlmv3/best.pt` (and `best_stage1.pt`). This checkpoint would be used to the second stage since the model was trained on random splits in case the number of tokens exceeding 512 tokens. So I freezed the model for the second stage since it should be handle it properly and train only the features aggregator.

## Stage 2 - freeze encoder, train attention aggregator

Single GPU:

```bash
python -m src.train --config config.yaml --stage 2 --encoder-checkpoint outputs/layoutlmv3/best.pt --output-dir outputs/layoutlmv3_stage2
```

Multiple GPUs (I did not test it at all due to issue above so trained on sigle GPU):

```bash
python -m torch.distributed.run --nproc_per_node=2 --module src.train --config config.yaml --stage 2 --encoder-checkpoint outputs/layoutlmv3/best.pt --output-dir outputs/layoutlmv3_stage2
```

So, as I mentioned above the model generate for overall features which are forwraded into aggregator. But, during the training the encoder is freezed and the aggregator was trained.
Writes `outputs/layoutlmv3_stage2/best_stage2.pt`.

## Quantize + benchmark (CUDA / bitsandbytes)

Borrowed some code to write it since I do not usually quantize models.

```bash
python scripts/quantize_and_benchmark.py --config config.yaml --checkpoint Xrenya/layoutlmv3_stage2 --output-dir outputs/quantization
```

For a local checkpoint:

```bash
python scripts/quantize_and_benchmark.py --config config.yaml --checkpoint outputs/layoutlmv3_stage2/best_stage2.pt --output-dir outputs/quantization --batch-size 16 --max-eval-samples 256
```

Writes the INT8 checkpoint and `quantization_report.json` under `outputs/quantization/`.

Overall model is not that fast (only 5%), the model should be larger to get speed up from quantization, while the checkpoint reduced in half which quite good. There is not significant changes in metrics so it should be tested on the larger subset.

```mardown
Model: INT8 LayoutLMv3 encoder (bitsandbytes) + FP32 aggregator/classifier
size_reduction_pct: 50.52098084292356
speedup_x: 1.0469634888131667
macro_f1_drop: -0.004447115384615508
macro_f1_drop_pct: -0.48964190172245226
```
