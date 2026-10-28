# Decisions

*[Versión en español](DECISIONS.es.md)*

Here are the choices made along the project and a small reason for each.

| Date | Decision | Why |
|---|---|---|
| 2026-10-23 | Study DeiT-Small, with DeiT-Tiny as the smaller uncompressed alternative. | Understanding the model to be compressed and have something to compare it against. |
| 2026-10-23 | Every point is scored on ImageNet-1k val, ImageNet-V2 and ImageNet-R, with paired-bootstrap intervals on the same images. | One test set can hide how compression behaves under distribution shift, and small differences need a paired test. |
| 2026-10-23 | Only methods that need no training and have public code are replicated. | We can run them under one protocol; methods that retrain are listed but left out because we lack the compute or specialized kernel. |
| 2026-10-23 | Token-reduction methods are left off the chart. | They do not change the file size. |
| 2026-10-23 | The grid of pruning and bit-width steps for our own method is not fixed yet. | It will be set after a pilot that measures each block's sensitivity. |
| 2026-10-26 | Only weights are compressed; activations stay in full precision. | Activations do not change the file size. |
| 2026-10-26 | Each published method runs with its authors' code and settings: the layers it quantizes, its grid shape and its number of calibration images. | The point on the chart should be as close to the authors' method as possible for accurate comparison. |
| 2026-10-26 | Only three things change for every method: the calibration images, the test sets and the activations. | We want to test the models under the same conditions. |
| 2026-10-26 | The calibration images are one training image per class, shuffled with seed 0 (`splits/calib_seed0.txt`), and a method that uses N images takes the first N. | Every method uses the same image set shuffled in the same way to represent as many classes as possible. |
| 2026-10-26 | A method's output is stored as a real packed file, keeping its grid values at the precision the method uses, and the weights read back must equal the method's own weights bit for bit. | The published code keeps FP32 weights and rounds them during the forward pass, so its saved checkpoint does not show the real size. |
| 2026-10-26 | Round-to-nearest stays on the chart as the reference. | It shows how much each method's search gains at each size. |
| 2026-10-26 | Latency is left out for now. | The packed files are expanded to floats to run, so the timing would not reflect the compressed formats and we don't have the kernel needed for inference at these sizes. |
