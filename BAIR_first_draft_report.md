# Action-conditioned video prediction on a BAIR subset

## Abstract

We compared persistence, a single-step convolutional model (Model A), and the same architecture trained on five-step self-fed rollouts (Model B). Using 900 training clips, 100 validation clips, and 256 held-out test clips, Model B reduced horizon-5 MSE by 48.65% relative to persistence and horizon-10 MSE by 45.46% relative to Model A. Both exceeded the proposal's 10% improvement targets in this experiment. Persistence retained higher full-frame SSIM and lower one-step MSE, so the learned model did not dominate every metric. Results apply to one seed and a community-reencoded subset of BAIR.

## Research question

Does training an action-conditioned model on its own multi-step predictions reduce error accumulation compared with training the same architecture only on real one-step inputs? We hypothesized that Model B would improve longer-horizon pixel accuracy. The predefined goals were at least 10% lower MSE than persistence at horizon 5 and at least 10% lower MSE than Model A at horizon 10.

## Data and preprocessing

We used auxiliary-camera videos and paired robot commands from `guyuchao/BAIR`, revision `776ad3b`. The converter streamed 1,000 training clips and 256 original test clips from shards 0 and 4 into Parquet. The training subset was selected in source order, then split into 900 training and 100 validation clips using seed 4620. The community index named `bair_val.json` refers to original test-directory clips; these were preserved as our test set.

Each clip contains 30 RGB frames at 64×64 resolution and four action dimensions. Images were scaled to [0,1], and action standardization used training clips only. Clips were split before creating windows. MP4 decoding may differ from the original raw pixels, so these results should not be presented as a directly comparable full-release BAIR benchmark. The implementation assumes action[t] corresponds to the transition from frame[t] to frame[t+1]; that physical timing convention has not been independently verified against the original collection logs.

## Models and training

The encoder receives three consecutive RGB frames as nine channels and compresses them to an 8×8 feature map. A learned action embedding is broadcast spatially and fused with visual features. A convolutional decoder predicts the next RGB image. Recursive predictions replace the oldest frame in the three-frame context.

Persistence repeats the last observed frame at every horizon. Model A minimizes one-step pixel MSE. Model B minimizes mean pixel MSE over five self-fed steps with gradients through the entire rollout. Both share the same architecture, initial parameters, eligible window population, and seeded window order. Future commands are provided during rollouts; the system predicts video conditional on actions rather than choosing actions.

Both models trained for 10 epochs with Adam, learning rate 0.001, batch size 32, width 32, and gradient clipping at 1.0. Training used PyTorch 2.9.1 on the Mac's MPS device. Model A took 22.6 minutes and Model B 35.2 minutes. Checkpoints were selected by validation endpoint MSE at horizon 10: epoch 2 for Model A and epoch 8 for Model B. Model B's greater computational cost should be considered when interpreting its advantage.

## Evaluation

Each held-out clip contributes one fixed window starting at frame 0, with three observed context frames and ten future targets. Learned models receive no future observed images. We average endpoint MSE, per-clip PSNR, and RGB SSIM across clips at each horizon. SSIM uses a 7×7 uniform window and data range 1. Motion MSE includes pixels whose mean absolute RGB change between consecutive true frames exceeds 10/255. Empty motion masks are excluded: horizon 5 has 254 clips with motion, while horizons 1 and 10 have 256.

The paired bootstrap resamples whole clips 2,000 times. Its intervals describe uncertainty across these test clips conditional on this training run; they do not measure variation across seeds.

## Test results

| Model | Horizon | MSE ↓ | PSNR (dB) ↑ | SSIM ↑ | Motion MSE ↓ |
|---|---:|---:|---:|---:|---:|
| Persistence | 1 | 0.003894 | 27.64 | 0.946 | 0.042706 |
| Model A: single-step | 1 | 0.007785 | 21.40 | 0.799 | 0.032892 |
| Model B: multi-step | 1 | 0.007661 | 21.48 | 0.805 | 0.033956 |
| Persistence | 5 | 0.020114 | 18.41 | 0.822 | 0.104706 |
| Model A: single-step | 5 | 0.016419 | 18.10 | 0.648 | 0.055389 |
| Model B: multi-step | 5 | 0.010330 | 20.15 | 0.756 | 0.044339 |
| Persistence | 10 | 0.025239 | 17.31 | 0.782 | 0.090266 |
| Model A: single-step | 10 | 0.025351 | 16.18 | 0.538 | 0.058243 |
| Model B: multi-step | 10 | 0.013825 | 18.85 | 0.692 | 0.040636 |

The proposal's comparisons were:

* Model A: single-step versus Persistence at horizon 5: **18.37% lower MSE**, paired bootstrap 95% interval [11.97%, 24.28%].
* Model B: multi-step versus Persistence at horizon 5: **48.65% lower MSE**, paired bootstrap 95% interval [44.39%, 52.67%].
* Model B: multi-step versus Model A: single-step at horizon 10: **45.46% lower MSE**, paired bootstrap 95% interval [43.78%, 47.08%].

Both predefined Model B targets were met, with their reported intervals above the 10% threshold. Model B's horizon-10 MSE was approximately 45.2% lower than persistence as well. The two learned models had similar one-step MSE, but Model A's error grew substantially more by horizon 10. This supports the multi-step training hypothesis for the tested subset and run.

Persistence was best on one-step full-frame MSE and had higher full-frame SSIM at all three reported horizons. Model A was slightly worse than persistence on horizon-10 MSE. Thus, single-step training alone did not reliably improve long-horizon accuracy. Model B had lower motion-region MSE than persistence at all reported horizons; this is useful evidence beyond a full-frame metric dominated by static background. It still does not establish physically accurate contacts or realistic object dynamics.

![Test metrics by prediction horizon](artifacts/bair_hf_aux1_v1/run_20261117T163624_197610Z/evaluation_test/metrics_by_horizon.png)

![Predictions for the fixed first test clip](artifacts/bair_hf_aux1_v1/run_20261117T163624_197610Z/evaluation_test/prediction_grid.png)

The prediction grid uses the first test clip by the fixed manifest order. It is an illustration rather than evidence that every clip behaves similarly. Full per-clip metrics and paired comparisons are saved in the run's `evaluation_test` directory.

## Limitations and ethical implications

This is a small source-order training subset, one seed, one camera, one collection setting, and ten epochs per model. It does not establish full-dataset performance, statistical reproducibility across retraining, or generalization to another robot. MP4 compression affects comparisons with raw-pixel benchmarks. The assumed action timing remains a data-verification limitation. A deterministic MSE objective can blur uncertain outcomes. Changing-pixel masks are not object segmentation, and improved image metrics do not establish safe physical planning.

The experiments use recorded videos and do not control a robot. Future use in planning would require separate checks for unfamiliar scenes, unreliable predictions, and collision risks. More convincing follow-up experiments would repeat training across seeds, test action-shuffling on validation data, and compare methods under matched compute budgets. These are follow-ups, not prerequisites for reporting this initial result.

## Conclusion

For this BAIR subset and training configuration, five-step self-fed training improved long-horizon pixel prediction and met both proposed MSE targets on the held-out test set. Persistence remained competitive on structural similarity and superior at the shortest horizon, limiting any claim of overall visual superiority.

## Sources and reproducibility

* Supplied project proposal: research question, three-system comparison, and 10% targets.
* [Community BAIR source](https://huggingface.co/datasets/guyuchao/BAIR/tree/776ad3b).
* [Official TFDS BAIR schema](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small).
* Run: `run_20261117T163624_197610Z`; seed: 4620. Exact configuration, manifest fingerprint, training histories, checkpoints, and evaluation tables are preserved under `artifacts/bair_hf_aux1_v1/`.
