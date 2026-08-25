# Module 1 Report

## Baseline Model Metrics

- Validation RMSE: 7.966 minutes
- Validation MAE: 4.395 minutes
- Validation R² = 0.417

## Serialization Comparison

The trained model was serialized using both Pickle and ONNX.

Prediction parity was evaluated using 500 validation samples.

### Prediction Parity

```text
Maximum prediction difference: 0.00000237
Tolerance: 0.0001
Result: Passed
```

The ONNX predictions closely match the original scikit-learn model predictions.

### Inference Latency

The benchmark measures model inference after feature transformation.

| Format | Mean Latency | P95 Latency |
| --- | ---: | ---: |
| Pickle / scikit-learn | 0.2649 ms | 0.3630 ms |
| ONNX Runtime | 0.0351 ms | 0.0626 ms |

In this benchmark, ONNX Runtime provided lower inference latency than the Pickle/scikit-learn model.

### Model Size

| Format | File Size |
| --- | ---: |
| Pickle | 141 KB |
| ONNX | 23 KB |

The ONNX model artifact is considerably smaller than the Pickle artifact.

### Serialization Format Trade-offs

| Format | Human Readable | Cross-language | Schema Enforcement | Safe for Untrusted Files |
| --- | --- | --- | --- | --- |
| JSON | Yes | Yes | No, unless validated separately | Generally safer as data |
| Protobuf | No | Yes | Yes | Safer than executable serialization |
| Pickle | No | Python-focused | No | No |
| ONNX | No | Yes | Yes, model graph format | More suitable for portable model inference |

Pickle is convenient for Python and preserves the scikit-learn objects directly, including the fitted `DictVectorizer`. However, Pickle files should only be loaded from trusted sources because unpickling malicious files can execute arbitrary code.

ONNX provides a portable model representation, smaller model size, and faster inference in the benchmark performed for this project.

For the current service, ONNX is the preferred format for model inference, while the fitted `DictVectorizer` is still loaded from the trusted Pickle artifact for feature transformation.