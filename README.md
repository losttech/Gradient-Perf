### TensorFlow .NET

```
tensorflow/core/platform/cpu_feature_guard.cc:142 Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Training epoch: 1
iter 000: Loss=2.3022, Training Accuracy=14.00% 221ms
iter 100: Loss=0.5189, Training Accuracy=88.00% 3041ms
iter 200: Loss=0.1782, Training Accuracy=95.00% 3021ms
iter 300: Loss=0.1938, Training Accuracy=91.00% 3016ms
iter 400: Loss=0.0924, Training Accuracy=96.00% 3037ms
iter 500: Loss=0.1010, Training Accuracy=98.00% 3017ms
---------------------------------------------------------
Epoch: 1, validation loss: 0.1122, validation accuracy: 96.74%
---------------------------------------------------------
Training epoch: 2
iter 000: Loss=0.1888, Training Accuracy=95.00% 1777ms
iter 100: Loss=0.0528, Training Accuracy=99.00% 3010ms
iter 200: Loss=0.0763, Training Accuracy=98.00% 3045ms
iter 300: Loss=0.0360, Training Accuracy=99.00% 3040ms
```

### Gradient

```
WARNING:tensorflow:From C:\Users\lost\.conda\envs\tf-1.x-gpu\lib\site-packages\tensorflow_core\python\util\deprecation.py:503: calling argmax (from tensorflow.python.ops.math_ops) with dimension is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
2020-03-19 18:32:46.419558: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-03-19 18:32:46.444918: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2020-03-19 18:32:46.448919: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: lost-pc
2020-03-19 18:32:46.452414: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: lost-pc
2020-03-19 18:32:46.455298: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Training epoch: 1
iter 000: Loss=2.3023, Training Accuracy=14.00% 271ms
iter 100: Loss=0.4832, Training Accuracy=89.00% 2479ms
iter 200: Loss=0.3046, Training Accuracy=95.00% 2482ms
iter 300: Loss=0.1041, Training Accuracy=95.00% 2464ms
iter 400: Loss=0.0930, Training Accuracy=96.00% 2471ms
iter 500: Loss=0.1119, Training Accuracy=96.00% 2483ms
---------------------------------------------------------
Epoch: 1, validation loss: 0.1042, validation accuracy: 96.76%
---------------------------------------------------------
Training epoch: 2
iter 000: Loss=0.1296, Training Accuracy=96.00% 1659ms
iter 100: Loss=0.1223, Training Accuracy=98.00% 2546ms
iter 200: Loss=0.0770, Training Accuracy=96.00% 2469ms
iter 300: Loss=0.0419, Training Accuracy=100.00% 2517ms
```

### Instructions to run (tested on Windows only)

- clone --recursive
- create a Conda environment `tf-1.x-cpu` with Python 3.7
- in the Conda environment install `tensorflow==1.15.0`
- Launch the solution, switch config to Release
- In `Debug` section of `Bench.Gradient` project properties add `GRADIENT_PYTHON_ENVIRONMENT` = `conda:tf-1.x-cpu` environment variable
- launch `Bench.Gradient` or `Bench.TF.NET` without debugging (e.g. Shift-F5)

### Remarks

This uses SciSharp's own sample code, a version, that works with their latest NuGet packages.
Gradient adaptation is in `src\Bench.Gradient\DigitRecognitionCNN.cs`. It is basically
a copy-paste with a few shims and Gradient-specific edits (see file history).
