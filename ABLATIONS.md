# Ablation predictions

These are our predictions, written before each test ran and kept as they were written. The results are in notebooks 05 and 06.

## Our allocation

Each ablation changes one choice in our per-block allocation (`scripts/allocate/allocate.py`) or in a pruned model. The allocation itself keeps using KL from the sensitivity test, and an ablation that scores better on the test sets is reported as what it is, without replacing the method.

| Ablation | What changes | What we expect, and why |
|---|---|---|
| Block 0's attention left unpruned | CORP joint at 30 %, with block 0's attention kept whole, alone and with FIMA-Q at 3 bits | A higher joint curve, since at 50 % this unit costs far more than every other block's attention together. Its file grows, so some of the gain goes back at the same size. |
| Block 0 forced down | Every layer of block 0 fixed at 2 bits | A clear loss, since block 0 changes the output most under all four quantizers. |
| `proj` forced down | Every `proj` fixed at 2 bits | A moderate loss. `proj` the proyection of qkv, O layers are not as fragile as the attention ones. |
| Block 11's MLP forced down | Both of its layers fixed at 2 bits | Little or no top-1 loss, even though KL ranks it highest among the MLPs, since it never cost more than a fraction of a point alone. |
| Top-1 as the signal | The allocation reads the top-1 change on calibration images instead of KL | Almost always worse or the same there is no way top1 accuracy goes up while cutting the model. |
| No allocation | The uniform points of the same size, against the allocated ones | The allocation wins by the amount the sensitivity test's spread suggests, more at low bits. |
| Stitching against searching | The same bits per layer, stitched from the uniform files and searched by FIMA-Q with those bits | The search does better, because block reconstruction lets each layer's rounding adapt to its neighbours at their own widths. |

## Which part of FIMA-Q is fragile

The base is FIMA-Q's W3A3 checkpoint on DeiT-S, the published quantizer that does best at low bits and gives no activation special treatment. It has ten activation points, the image at the patch embedding, the inputs to qkv, proj, fc1 and fc2, Q, K, V, the attention scores after the softmax, and the input to the head. Points are switched off without searching again, so each test measures the checkpoint with parts removed, not what a new search would reach.

| Test | What changes | What we expect, and why |
|---|---|---|
| Weights against activations | Only the weights or only the activations stay quantized, at W3A3, W4A4, W6A6 and W8A8 | Activations are a lot more fragile than weights as seen when recreating the literature weights can be quantized up to W3 with not much accuracy loss but as soon as activations are introduced due to the sparsity on their data values accuracy drops really fast unless the method adapts correctly |
| Remove one | One point goes back to full precision in every block, the rest stay at W3A3 | Getting post gelu layers and qkv to full precision restores a lot more accuracy. |
| Keep one | Only that point stays quantized, in every block, with weights at full precision | V costs more than q or K because the values impact directly on logit output compared to q or k that cause compunding error, accuracy fluctuations are way more noticeable |
| Depth | The same two tests for one block at a time: blocks 0, 6 and 11 | Block 0 matters most and block 11 least, as for weights. Single-block effects will often be small, near the noise of a few tenths of a point. |
| How low it can go | One point at A3, A4, A6 and A8, with its scale searched again by mean squared error on the 128 calibration images and everything else at W3A3 | A8 is pretty much the same as full precision with current methods the most damaging activation quantization is A3, A4 is close and A6 is a usable middle ground |

A point counts as fragile only if its paired difference from the base is at least 0.5 points on val, with an interval that excludes zero, and goes the same way on V2 and R. This rule was set before any result was read.

Two things changed on 2026-11-04, before any fragility score was read. The depth test was dropped, since MixFrag and our own weight sensitivity already rank the layers, and the width test covers only the three points the first tests rank highest. The switch tests use the checkpoint from after calibration, because block reconstruction writes the rounded values into the weights and a finished checkpoint cannot return to FP32 (34.58 on val with every quantizer off, against 79.86). From that checkpoint the W3A3 base is 23.48, which reconstruction later lifts to 70.05, so these tests show what breaks before reconstruction, and full FIMA-Q searches with one point left at full precision connect them to the finished method. Tests 2 and 3 also run at W4A4, whose base is 68.00, where we expect the same points to lead with clearer differences.
