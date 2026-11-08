# Ablations, with what we expect before running them

*Written on 2026-11-04, before any of these ran.* Each one changes a single choice in our per-block allocation (`scripts/allocate/allocate.py`) or in a pruned model, because the replications and the sensitivity test point at that part of the model. The allocation looks only at calibration images, and the primary allocation stays the one that uses KL from the sensitivity test whatever these show. An ablation that scores better on the test sets is reported as what it is, and it does not replace the method after the fact.

Most of them are first looks, built by taking each layer from FIMA-Q's uniform files, so they are compared with the same construction at the same target size, 7.4 MB unless it says otherwise.

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

*Written on 2026-11-04, before any of these ran. These expectations were drafted with AI assistance and are for the team to review and own.* The base is FIMA-Q's W3A3 checkpoint on DeiT-S, chosen because it is the published quantizer that does best at low bits and gives no activation special treatment, so it does not favour any answer in advance. It has ten activation points: the image at the patch embedding, the input to qkv, Q, K, the attention scores after the softmax, V, the input to proj, the input to fc1, the input to fc2 after the GELU, and the input to the head. The published configs set the image to 8 bits, which FIMA-Q's code treats as no quantization at all, so that point can only be tested by lowering it. Points are switched off without searching again, so their scales stay those found for all parts together, and the numbers measure a checkpoint with parts removed, not what a new search would reach.

| Test | What changes | What we expect, and why |
|---|---|---|
| Weights against activations | Only the weights or only the activations stay quantized, at W3A3, W4A4, W6A6 and W8A8 | Activations are a lot more fragile than weights as seen when recreating the literature weights can be quantized up to W3 with not much accuracy loss but as soon as activations are introduced due to the sparsity on their data values accuracy drops really fast unless the method adapts correctly |
| Remove one | One point goes back to full precision in every block, the rest stay at W3A3 | Getting post gelu layers and qkv to full precision restores a lot more accuracy. |
| Keep one | Only that point stays quantized, in every block, with weights at full precision | V costs more than q or K because the values impact directly on logit output compared to q or k that cause compunding error, accuracy fluctuations are way more noticeable |
| Depth | The same two tests for one block at a time: blocks 0, 6 and 11 | Block 0 matters most and block 11 least, as for weights. Single-block effects will often be small, near the noise of a few tenths of a point. |
| How low it can go | One point at A3, A4, A6 and A8, with its scale searched again by mean squared error on the 128 calibration images and everything else at W3A3 | A8 is pretty much the same as full precision with current methods the most damaging activation quantization is A3, A4 is close and A6 is a usable middle ground |

The primary result is which points return the most accuracy on val, confirmed on V2 and R with paired comparisons on the same images.

A point or block counts as fragile only if its paired difference from the W3A3 base is at least 0.5 points on val, with an interval that excludes zero, and goes the same way on V2 and R. This threshold was set on 2026-11-04 before any test result had been read, to keep 117 comparisons from turning noise into findings.

Changed on 2026-11-04, before any test result had been read. The depth test is dropped, because MixFrag already ranks layers and finds the early blocks most fragile, and our own weight sensitivity agrees. The width test covers only the three points that the first tests rank as most fragile. Compression order is no longer an experiment: we follow the Progressive Intensity Hypothesis (Kim et al., ICLR 2026), which applies the step that costs less accuracy on its own first. One short check at CORP joint 20 and 30 % with W3 tests whether our pipeline's order costs anything there.

Changed on 2026-11-04, after the switches were tested and before any fragility score was read. The switch tests use FIMA-Q's checkpoint from after calibration, not the finished one. Block reconstruction ends by writing the rounded values into the weights (`block_recon.py`, line 383), so a finished checkpoint cannot return to FP32: with every quantizer off it reaches 34.58 on val, against 79.86. The calibration checkpoint returns to exactly 79.86 with every quantizer off, so its switches are valid. Its W3A3 base is 23.48, the value the authors' own log prints, and reconstruction later lifts it to 70.05. These tests therefore show what breaks before reconstruction, and full searches with one point left at full precision are what connect them to the finished method.

Added on 2026-11-04, before any fragility score was read: tests 2 and 3 also run on the W4A4 calibration checkpoint, whose base is 68.00. At W3A3 the base is 23.48, and restoring one point may show little while so much else is broken. We expect the same points to lead at both widths, with clearer differences at W4A4.
