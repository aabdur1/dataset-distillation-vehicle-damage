# Dataset Distillation for Vehicle Damage Classification

Capstone project for the MS in Management Information Systems at the University of Illinois Chicago (IDS 507, Fall 2026), sponsored by CCC Intelligent Solutions. Team of three.

This repository is a write-up only. The code is in the team's private repository, and no dataset images or distilled images are included here.

## The question

Dataset distillation learns a small set of synthetic training images that can stand in for a full dataset. If a few hundred images can train a model nearly as well as a few thousand, training gets cheaper and there is less data to store and move. Most published results come from benchmarks like CIFAR-10. We tested whether it holds up on photos of vehicle damage.

## Data

- **CarDD**: car damage photos, cropped to the damaged area and resized to 64 by 64, in six classes (dent, scratch, crack, glass shatter, lamp broken, tire flat). The test split has 785 crops.
- **VehiDE**: vehicle damage photos in seven classes, 24,575 training crops. The test split has 4,940 crops.

## Method

**Distillation.** We used distribution matching (Zhao and Bilen, 2023). It updates the synthetic images so that their average features, taken through randomly initialized networks, match the real images' features class by class. Each synthetic set starts from real crops picked by herding, and a small pixel learning rate keeps the images close to those starting crops.

**Budgets.** 10, 50, and 100 images per class.

**Comparison sets.** The herding picks the distilled set started from, randomly picked real crops, and the full training set.

**Evaluation.** Each set trains a small ConvNet and a ResNet-18 from scratch with hard labels. Every comparison uses the same five training seeds for both sets, so each seed gives a paired difference. The headline score is macro-F1, also reported as the share of full-data macro-F1 that a set recovers. Accuracy is recorded alongside it.

**Test protocol.** We wrote down the claims and the decision rules, froze the code at a commit, and then scored every configuration once on the untouched test splits. A gain counts as a win when at least four of the five seeds favor the distilled set and the mean gain is at least twice its standard error.

## Results on the held-out test split

Macro-F1 of the distilled set minus the herding picks, in points, over five paired seeds. Validation figures are in brackets.

| Dataset | Images per class | ConvNet | Seeds ahead | ResNet-18 | Seeds ahead |
|---|---|---|---|---|---|
| CarDD | 10 | +7.3 [+7.1] | 5 of 5 | +1.7 [+1.3] | 4 of 5 |
| CarDD | 50 | +3.0 [+4.5] | 5 of 5 | +1.7 [+2.2] | 4 of 5 |
| CarDD | 100 | +4.8 [+3.9] | 5 of 5 | +2.2 [+2.2] | 5 of 5 |
| VehiDE | 50 | +5.8 [+5.9] | 5 of 5 | +4.0 [+3.7] | 5 of 5 |

- On the ConvNet, the distilled set beats the herding picks at every budget on both datasets.
- On ResNet-18, it is at or above the picks on CarDD at every budget. At 50 per class the mean gain misses our twice-the-standard-error rule by 0.04 of a point. On VehiDE, it beats the picks on all five seeds.
- At 100 images per class on CarDD, which is 9.7% of the training crops, the distilled set recovers 84% of full-data macro-F1 on the ConvNet and 83% on ResNet-18. The herding picks recover 78% and 80%.
- In accuracy the distilled set is ahead of the picks on every seed, on both networks and both datasets, at every budget.
- Test and validation agree within about a point on every row except the ConvNet at 50 per class, which dropped 1.5 points on test.

## What did not work

- **Starting from noise.** Sets distilled from random noise instead of real crops landed far below the herding picks on both networks (8.4 and 18.9 points below at 50 per class on CarDD).
- **A large pixel step.** A set distilled with our first learning rate, about 12 times the published step, moved its images much further from the real crops. On validation it kept a ConvNet gain, but on test it was 0.8 points below the picks on the ConvNet and 8.5 below on ResNet-18.
- **A pixel-change penalty.** We tried penalizing the distillation for moving each pixel away from its starting crop, which we did not find in the literature. Holding the large step to the same pixel distance as the small step did not produce an equally good set. We report it as a negative result. (validation split)
- **IDM-style matching** (a queue of partly trained matching networks plus a class-aware term, from Zhao et al., 2023) gave the same outcomes as plain distribution matching and showed no gain at any budget we tried. (validation split)

## Limitations

- Two damage datasets at 64x64 resolution, hard labels only.
- On test, each configuration is one distilled set trained under five seeds. Variation between separate distillation runs was not measured on test.
- The herding picks themselves scored higher on test than on validation at 50 per class, so part of each difference at that budget moves with the comparison set.
- Work on a larger benchmark (Tiny ImageNet) is in progress and has no test protocol yet, so it is not reported here.

## My role

I designed the experiments, ran the reported PyTorch runs, did the analysis, wrote the weekly updates to the sponsor, and gave most of the sponsor presentations. A teammate reproduced the first result rows on his own machine. Much of the code was drafted with Claude Code; I reviewed, ran, and checked it.

## Tools

Python, PyTorch, torchvision, Google Colab (A100), Git.

## Reproducing

The code stays in the team's private repository, and both datasets come with their own terms of use, so there is nothing runnable here. I would be happy to walk through the code and the results on a call.

## References

- Zhao, B., and Bilen, H. (2023). Dataset Condensation with Distribution Matching. WACV.
- Zhao, G., Li, G., Qin, Y., and Yu, Y. (2023). Improved Distribution Matching for Dataset Condensation. CVPR.
- Welling, M. (2009). Herding Dynamical Weights to Learn. ICML.
- Wang, X., Li, W., and Wu, Z. (2023). CarDD: A New Dataset for Vision-Based Car Damage Detection. IEEE Transactions on Intelligent Transportation Systems.
