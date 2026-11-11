# Public measurements used for the processing-time model

`nvidia_bert_tf_t4_fp16.csv` is transcribed from the "Inference performance:
NVIDIA Tesla T4 (1x T4 16GB)" table of the NVIDIA DeepLearningExamples
TensorFlow BERT README (retrieved 2026-11-11):

https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/README.md

Conditions stated by the source: SQuAD v1.1 fine-tuning inference,
`scripts/finetune_inference_benchmark.sh`, TensorFlow 20.06-py3 NGC container,
one T4 16GB GPU, numbers averaged over 1024 iterations.

Only the FP16 rows are transcribed. Columns kept: throughput (sentences per
second) and the 50th-percentile latency in milliseconds. The source's average
latency column is not used: it is inflated by a multi-second maximum (the
100% column), which looks like a first-iteration warm-up.

The source measures batch sizes 1, 2, 4 and 8 and sequence lengths 128 and
384 only. Anything outside that range is extrapolation.
