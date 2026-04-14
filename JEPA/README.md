# JEPA

JEPA-based code repair pipeline.

## Structure

- `src/jepa/`: core source code
- `src/jepa/tasks/encoder/`: encoder training and embedding inference
- `src/jepa/tasks/decoder/`: decoder downstream training and decoding
- `scripts/`: runnable entrypoints
- `configs/`: experiment configs


## Pipeline

1. Train the JEPA encoder and predictor on buggy/fixed code pairs.
2. Run embedding inference on buggy code to get predicted fixed embeddings.
3. Map predicted embeddings to soft prompts for the decoder.
4. Train the decoder with either:
   - projector only
   - projector + LoRA
5. Decode repaired code and evaluate it with downstream tests.

## Main Entry Points

Encoder:
- `python3 scripts/encoder/train.py`
- `python3 scripts/encoder/train_Sigreg.py`
- `python3 scripts/encoder/embed.py`

Decoder:
- `python3 scripts/decoder/train_projector.py`
- `python3 scripts/decoder/train_lora.py`
- `python3 scripts/decoder/test_projector.py`
- `python3 scripts/decoder/test_lora.py`

