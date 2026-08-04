This directory bundles a third-party audio event classification model.

- **Model**: YAMNet (521 AudioSet classes)
- **Original weights**: Google, released under the Apache License 2.0 as part of
  https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
  (`https://storage.googleapis.com/audioset/yamnet.h5`).
- **PyTorch conversion**: `torch_audioset` by w-hc (MIT License),
  https://github.com/w-hc/torch_audioset
- **ONNX export**: Qualcomm AI Hub Models (BSD-3-Clause),
  https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/yamnet
  Downloaded from the v0.59.0 release asset:
  https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/yamnet/releases/v0.59.0/yamnet-onnx-float.zip

`yamnet.onnx` + `yamnet.data` are the unmodified ONNX graph/weights from that release.
`labels.txt`-derived class names are vendored into `crates/game-events/assets/yamnet_class_map.csv`.

Model input: `"audio"`, float32 tensor of shape `(1, 1, 96, 64)` — a single
96-frame x 64-mel-bin log-mel spectrogram patch (25ms window / 10ms hop / 16kHz).
Model output: `"class_scores"`, float32 tensor of shape `(1, 521)`.
