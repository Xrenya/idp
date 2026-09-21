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

Writes `outputs/layoutlmv3/best.pt` (and `best_stage1.pt`). This checkpoint would be used to the second stage since the model was trained on random splits in case the number of tokens exceeding 512 tokens. So I freezed the model for the second stage since it should be handle it properly and train only the features aggregator, but in this case it generates features 400 tokens with overal 100 and each chunk gives cls token feature which goes into the aggregator to make the final prediction.

## Stage 2 - freeze encoder, train attention aggregator

Single GPU:

```bash
python -m src.train --config config.yaml --stage 2 --encoder-checkpoint outputs/layoutlmv3/best.pt --output-dir outputs/layoutlmv3_stage2
```

Multiple GPUs (I did not test it at all due to issue above so I trained on sigle GPU):

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

Overall model is not that fast (only 6%), the model should be larger to get speed up from quantization (I did not convert the aggregation layer into int8 since it is too small to give any noticeable boots to the model, but probably in production better convert into tensorrt rather then keeping the pure torch), while the checkpoint was reduced in half which is quite good. Metrics are slightly worse so it should be tested on the larger subset.

```markdown
Model: INT8 LayoutLMv3 encoder (bitsandbytes) + FP32 aggregator/classifier
size_reduction_pct: 50.52098084292356
speedup_x: 1.0623752956427268
macro_f1_drop: 0.004627766599597627 (macro_f1_drop = FP32 macro-F1 − INT8 macro-F1)
macro_f1_drop_pct: 0.8391004245049617 (macro_f1_drop_pct = (macro_f1_drop_pct / FP32 macro-F1) * 100)
```


## Evaluate model

Evaluate the first stage model
```bash
python -m src.evaluate --config config.yaml --stage 1 --checkpoint outputs/layoutlmv3/best.pt
```

Evaluate the second stage model
```bash
python -m src.evaluate --config config.yaml --stage 2 --checkpoint outputs/layoutlmv3_stage2/best_stage2.pt
```

## Models

The aggregation attention looks like make sense for long document predictions, as I explained above, we process the with trained frozen backbone and the aggregate features from each overlap window to make final prediction based on the sum of the weighted feature output which is used to make the final prediction using the classification layer. Probabaly, any seq2seq model would work fine RNN, LSTM, Bidirectional models including transformer, since the current models are mostly attention based then the attention model should work here as well. So, it would increase features toward most relevant features while reducing affect of less relevant features, also it would be possible to debug and check which particular chuck of data makes the strongest contribution towards the final output (like it is done in SHAP).    
Truncation is not good since the most relevant information might be located outside the context window, the generated features for long context would beneficial since it would be possible to get the most relevant information from any position within the text. Pooling would allow to get the most relevant based on the features values, same goes to averaging - adding a lot of relevant features would get even more feature value. The long context allows to get the capture the most relevant features from it (e.g. the current best examples are RAGs when you encode whole documents into chunks of embeddings and then it is quite easy to retrieve it again).

## Loss
I have used the focal loss (BCE) in order to tackle class disbalance, since the original dataset contains 15 classes and the current one has only 4 targets, I have sampled around approximately the same number of from the rest of data, so we have a lot target and non-target samples, which are a lot less than target. 

## Data
The data target class can viewed on Hugging Face since it was saved as well. The augmentations were done with AlbumentationsX which is working with bounding boxes and image quite well.

## Architecture
The inference pipeline for document (image) processing I have include into separate repo (inferece), which show how the production pipeline look like with image reading, image rotation, OCR and finally prediction. See details in **inference** folder.