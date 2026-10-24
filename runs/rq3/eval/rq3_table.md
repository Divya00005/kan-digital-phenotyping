# RQ3: conformal prediction sets (nested student-level CV, 170 train + validation students)

Test set not used. Coverage = share of forecasts whose true score lies in the set, averaged over students (each student counts once; this is what the guarantee is about). 95% CIs: 1,000 student-level bootstrap resamples, sets held fixed. 'values' = mean number of the 7 possible scores (0-6) in the set. Over-coverage is expected with integer scores (no randomised tie-breaking).

Students per group: all 170, F 117, M 51, gender_both 1, gender_unknown 1, asian 42, black 8, hispanic_other 6, multiracial_native 9, race_unknown 1, white 104

## Method B (main), pooled calibration, target 90%

| Model | Task | 1 wk: coverage [CI] · values | 2 wk: coverage [CI] · values | 4 wk: coverage [CI] · values | 8 wk: coverage [CI] · values |
|---|---|---|---|---|---|
| Persistence | depression | 0.968 [0.959, 0.975] · 3.77 | 0.957 [0.946, 0.966] · 3.77 | 0.945 [0.934, 0.956] · 3.77 | 0.931 [0.917, 0.945] · 3.77 |
| Persistence | anxiety | 0.968 [0.960, 0.974] · 3.98 | 0.954 [0.945, 0.962] · 3.97 | 0.945 [0.932, 0.956] · 3.97 | 0.931 [0.916, 0.944] · 3.97 |
| Linear (history) | depression | 0.914 [0.900, 0.927] · 2.68 | 0.913 [0.898, 0.926] · 2.89 | 0.917 [0.900, 0.932] · 3.18 | 0.918 [0.901, 0.934] · 3.37 |
| Linear (history) | anxiety | 0.910 [0.897, 0.923] · 2.93 | 0.913 [0.897, 0.926] · 3.17 | 0.914 [0.896, 0.930] · 3.30 | 0.915 [0.897, 0.931] · 3.50 |
| KAN no-ODE H | depression | 0.921 [0.907, 0.935] · 2.90 | 0.922 [0.908, 0.936] · 3.16 | 0.923 [0.907, 0.939] · 3.33 | 0.922 [0.905, 0.938] · 3.46 |
| KAN no-ODE H | anxiety | 0.915 [0.897, 0.930] · 3.09 | 0.914 [0.897, 0.930] · 3.28 | 0.913 [0.893, 0.929] · 3.35 | 0.915 [0.896, 0.934] · 3.58 |
| KAN no-ODE B | depression | 0.919 [0.902, 0.933] · 3.05 | 0.916 [0.900, 0.932] · 3.21 | 0.919 [0.902, 0.936] · 3.34 | 0.916 [0.899, 0.932] · 3.48 |
| KAN no-ODE B | anxiety | 0.908 [0.889, 0.926] · 3.21 | 0.911 [0.892, 0.927] · 3.33 | 0.911 [0.893, 0.927] · 3.35 | 0.912 [0.892, 0.929] · 3.58 |
| KAN no-ODE Hn (no gender/race) | depression | 0.916 [0.901, 0.930] · 2.85 | 0.917 [0.901, 0.931] · 3.12 | 0.917 [0.899, 0.932] · 3.25 | 0.920 [0.902, 0.936] · 3.42 |
| KAN no-ODE Hn (no gender/race) | anxiety | 0.913 [0.897, 0.928] · 3.07 | 0.914 [0.898, 0.929] · 3.23 | 0.914 [0.894, 0.931] · 3.34 | 0.917 [0.897, 0.933] · 3.58 |
| KAN-ODE H | depression | 0.918 [0.902, 0.933] · 2.88 | 0.920 [0.904, 0.935] · 3.16 | 0.919 [0.901, 0.936] · 3.33 | 0.919 [0.901, 0.936] · 3.50 |
| KAN-ODE H | anxiety | 0.918 [0.903, 0.931] · 3.08 | 0.915 [0.900, 0.929] · 3.26 | 0.914 [0.895, 0.930] · 3.35 | 0.919 [0.900, 0.936] · 3.64 |
| Linear quantile (CQR) | depression | 0.939 [0.929, 0.949] · 3.08 | 0.940 [0.929, 0.949] · 3.24 | 0.940 [0.930, 0.949] · 3.37 | 0.936 [0.923, 0.948] · 3.56 |
| Linear quantile (CQR) | anxiety | 0.945 [0.937, 0.952] · 3.42 | 0.945 [0.937, 0.953] · 3.58 | 0.942 [0.930, 0.952] · 3.66 | 0.941 [0.929, 0.952] · 3.83 |
| Quantile KAN H (CQR) | depression | 0.919 [0.906, 0.930] · 3.02 | 0.920 [0.908, 0.932] · 3.14 | 0.916 [0.902, 0.929] · 3.25 | 0.920 [0.905, 0.934] · 3.39 |
| Quantile KAN H (CQR) | anxiety | 0.914 [0.898, 0.927] · 3.23 | 0.911 [0.895, 0.924] · 3.33 | 0.903 [0.886, 0.919] · 3.40 | 0.913 [0.896, 0.927] · 3.62 |

## Method B (main), pooled calibration, target 80%

| Model | Task | 1 wk: coverage [CI] · values | 2 wk: coverage [CI] · values | 4 wk: coverage [CI] · values | 8 wk: coverage [CI] · values |
|---|---|---|---|---|---|
| Persistence | depression | 0.859 [0.841, 0.875] · 2.47 | 0.835 [0.816, 0.855] · 2.47 | 0.889 [0.867, 0.909] · 3.25 | 0.931 [0.917, 0.945] · 3.77 |
| Persistence | anxiety | 0.833 [0.813, 0.851] · 2.59 | 0.954 [0.945, 0.962] · 3.97 | 0.945 [0.932, 0.956] · 3.97 | 0.931 [0.916, 0.944] · 3.97 |
| Linear (history) | depression | 0.813 [0.790, 0.834] · 2.04 | 0.815 [0.793, 0.837] · 2.19 | 0.815 [0.791, 0.839] · 2.33 | 0.822 [0.797, 0.845] · 2.50 |
| Linear (history) | anxiety | 0.808 [0.787, 0.828] · 2.20 | 0.808 [0.784, 0.829] · 2.36 | 0.813 [0.787, 0.836] · 2.52 | 0.811 [0.784, 0.835] · 2.60 |
| KAN no-ODE H | depression | 0.820 [0.796, 0.842] · 2.15 | 0.816 [0.791, 0.839] · 2.27 | 0.819 [0.795, 0.842] · 2.39 | 0.819 [0.794, 0.843] · 2.51 |
| KAN no-ODE H | anxiety | 0.809 [0.787, 0.831] · 2.27 | 0.814 [0.791, 0.836] · 2.43 | 0.813 [0.788, 0.836] · 2.56 | 0.812 [0.785, 0.837] · 2.68 |
| KAN no-ODE B | depression | 0.809 [0.786, 0.831] · 2.21 | 0.810 [0.787, 0.833] · 2.32 | 0.812 [0.788, 0.835] · 2.42 | 0.816 [0.790, 0.839] · 2.54 |
| KAN no-ODE B | anxiety | 0.804 [0.779, 0.828] · 2.35 | 0.807 [0.781, 0.831] · 2.47 | 0.813 [0.789, 0.836] · 2.57 | 0.814 [0.789, 0.838] · 2.73 |
| KAN no-ODE Hn (no gender/race) | depression | 0.812 [0.790, 0.834] · 2.12 | 0.814 [0.791, 0.837] · 2.24 | 0.816 [0.790, 0.840] · 2.37 | 0.820 [0.796, 0.844] · 2.50 |
| KAN no-ODE Hn (no gender/race) | anxiety | 0.807 [0.783, 0.828] · 2.25 | 0.812 [0.788, 0.833] · 2.39 | 0.815 [0.790, 0.838] · 2.54 | 0.813 [0.787, 0.837] · 2.64 |
| KAN-ODE H | depression | 0.816 [0.792, 0.838] · 2.13 | 0.816 [0.793, 0.839] · 2.25 | 0.817 [0.793, 0.842] · 2.39 | 0.823 [0.798, 0.848] · 2.55 |
| KAN-ODE H | anxiety | 0.812 [0.789, 0.832] · 2.25 | 0.814 [0.791, 0.836] · 2.41 | 0.817 [0.792, 0.839] · 2.56 | 0.815 [0.790, 0.840] · 2.67 |
| Linear quantile (CQR) | depression | 0.870 [0.855, 0.885] · 2.50 | 0.887 [0.874, 0.900] · 2.71 | 0.886 [0.871, 0.901] · 2.83 | 0.881 [0.862, 0.899] · 2.96 |
| Linear quantile (CQR) | anxiety | 0.854 [0.837, 0.870] · 2.70 | 0.882 [0.868, 0.895] · 2.96 | 0.876 [0.859, 0.892] · 3.04 | 0.888 [0.872, 0.904] · 3.21 |
| Quantile KAN H (CQR) | depression | 0.826 [0.806, 0.845] · 2.28 | 0.826 [0.806, 0.844] · 2.36 | 0.820 [0.800, 0.840] · 2.47 | 0.820 [0.796, 0.841] · 2.59 |
| Quantile KAN H (CQR) | anxiety | 0.808 [0.790, 0.826] · 2.49 | 0.807 [0.787, 0.826] · 2.57 | 0.811 [0.791, 0.830] · 2.71 | 0.804 [0.779, 0.827] · 2.83 |

## Methods compared, target 90%, pooled calibration (coverage [CI] · values)

| Model | Method | Task | 1 wk | 2 wk | 4 wk | 8 wk |
|---|---|---|---|---|---|---|
| Persistence | B: student-level repeated subsampling (main) | depression | 0.968 [0.959, 0.975] · 3.77 | 0.957 [0.946, 0.966] · 3.77 | 0.945 [0.934, 0.956] · 3.77 | 0.931 [0.917, 0.945] · 3.77 |
| Persistence | B: student-level repeated subsampling (main) | anxiety | 0.968 [0.960, 0.974] · 3.98 | 0.954 [0.945, 0.962] · 3.97 | 0.945 [0.932, 0.956] · 3.97 | 0.931 [0.916, 0.944] · 3.97 |
| Persistence | B1: single subsample | depression | 0.968 [0.959, 0.975] · 3.77 | 0.933 [0.918, 0.947] · 3.50 | 0.945 [0.934, 0.956] · 3.77 | 0.940 [0.926, 0.952] · 3.98 |
| Persistence | B1: single subsample | anxiety | 0.968 [0.960, 0.974] · 3.98 | 0.954 [0.945, 0.962] · 3.97 | 0.945 [0.932, 0.956] · 3.97 | 0.945 [0.932, 0.956] · 4.38 |
| Persistence | A: survey-level split conformal | depression | 0.968 [0.959, 0.975] · 3.77 | 0.957 [0.946, 0.966] · 3.77 | 0.945 [0.934, 0.956] · 3.77 | 0.931 [0.917, 0.945] · 3.77 |
| Persistence | A: survey-level split conformal | anxiety | 0.968 [0.960, 0.974] · 3.98 | 0.954 [0.945, 0.962] · 3.97 | 0.945 [0.932, 0.956] · 3.97 | 0.931 [0.916, 0.944] · 3.97 |
| Persistence | pooled CDF | depression | 0.968 [0.959, 0.975] · 3.77 | 0.957 [0.946, 0.966] · 3.77 | 0.945 [0.934, 0.956] · 3.77 | 0.931 [0.917, 0.945] · 3.77 |
| Persistence | pooled CDF | anxiety | 0.968 [0.960, 0.974] · 3.98 | 0.954 [0.945, 0.962] · 3.97 | 0.945 [0.932, 0.956] · 3.97 | 0.931 [0.916, 0.944] · 3.97 |
| Persistence | B + CV+ (empirical only) | depression | 0.968 [0.959, 0.975] · 3.77 | 0.957 [0.946, 0.966] · 3.77 | 0.945 [0.934, 0.956] · 3.77 | 0.931 [0.917, 0.945] · 3.77 |
| Persistence | B + CV+ (empirical only) | anxiety | 0.968 [0.960, 0.974] · 3.98 | 0.954 [0.945, 0.962] · 3.97 | 0.945 [0.932, 0.956] · 3.97 | 0.931 [0.916, 0.944] · 3.97 |
| Linear (history) | B: student-level repeated subsampling (main) | depression | 0.914 [0.900, 0.927] · 2.68 | 0.913 [0.898, 0.926] · 2.89 | 0.917 [0.900, 0.932] · 3.18 | 0.918 [0.901, 0.934] · 3.37 |
| Linear (history) | B: student-level repeated subsampling (main) | anxiety | 0.910 [0.897, 0.923] · 2.93 | 0.913 [0.897, 0.926] · 3.17 | 0.914 [0.896, 0.930] · 3.30 | 0.915 [0.897, 0.931] · 3.50 |
| Linear (history) | B1: single subsample | depression | 0.930 [0.917, 0.941] · 2.93 | 0.909 [0.893, 0.922] · 2.91 | 0.886 [0.869, 0.903] · 2.86 | 0.917 [0.900, 0.932] · 3.35 |
| Linear (history) | B1: single subsample | anxiety | 0.911 [0.897, 0.924] · 2.94 | 0.892 [0.874, 0.908] · 3.06 | 0.894 [0.875, 0.911] · 3.10 | 0.931 [0.915, 0.946] · 3.81 |
| Linear (history) | A: survey-level split conformal | depression | 0.888 [0.872, 0.902] · 2.47 | 0.892 [0.876, 0.907] · 2.67 | 0.893 [0.875, 0.910] · 2.90 | 0.898 [0.879, 0.915] · 3.15 |
| Linear (history) | A: survey-level split conformal | anxiety | 0.878 [0.860, 0.893] · 2.65 | 0.880 [0.861, 0.898] · 2.83 | 0.879 [0.857, 0.897] · 2.96 | 0.881 [0.860, 0.901] · 3.10 |
| Linear (history) | pooled CDF | depression | 0.901 [0.887, 0.915] · 2.56 | 0.901 [0.885, 0.915] · 2.76 | 0.902 [0.885, 0.919] · 3.00 | 0.908 [0.890, 0.924] · 3.25 |
| Linear (history) | pooled CDF | anxiety | 0.898 [0.883, 0.912] · 2.82 | 0.899 [0.883, 0.913] · 3.03 | 0.899 [0.878, 0.917] · 3.15 | 0.903 [0.883, 0.920] · 3.35 |
| Linear (history) | B + CV+ (empirical only) | depression | 0.907 [0.893, 0.921] · 2.61 | 0.907 [0.892, 0.921] · 2.82 | 0.909 [0.892, 0.923] · 3.04 | 0.906 [0.888, 0.923] · 3.26 |
| Linear (history) | B + CV+ (empirical only) | anxiety | 0.908 [0.894, 0.920] · 2.89 | 0.907 [0.891, 0.921] · 3.09 | 0.909 [0.890, 0.925] · 3.25 | 0.906 [0.887, 0.923] · 3.38 |
| KAN no-ODE H | B: student-level repeated subsampling (main) | depression | 0.921 [0.907, 0.935] · 2.90 | 0.922 [0.908, 0.936] · 3.16 | 0.923 [0.907, 0.939] · 3.33 | 0.922 [0.905, 0.938] · 3.46 |
| KAN no-ODE H | B: student-level repeated subsampling (main) | anxiety | 0.915 [0.897, 0.930] · 3.09 | 0.914 [0.897, 0.930] · 3.28 | 0.913 [0.893, 0.929] · 3.35 | 0.915 [0.896, 0.934] · 3.58 |
| KAN no-ODE H | B1: single subsample | depression | 0.922 [0.907, 0.936] · 2.96 | 0.936 [0.922, 0.949] · 3.38 | 0.893 [0.872, 0.911] · 3.04 | 0.908 [0.889, 0.925] · 3.28 |
| KAN no-ODE H | B1: single subsample | anxiety | 0.910 [0.892, 0.927] · 3.11 | 0.914 [0.894, 0.931] · 3.43 | 0.896 [0.876, 0.914] · 3.16 | 0.923 [0.905, 0.941] · 3.71 |
| KAN no-ODE H | A: survey-level split conformal | depression | 0.904 [0.888, 0.920] · 2.73 | 0.901 [0.885, 0.917] · 2.90 | 0.902 [0.884, 0.919] · 3.05 | 0.902 [0.882, 0.919] · 3.23 |
| KAN no-ODE H | A: survey-level split conformal | anxiety | 0.881 [0.861, 0.899] · 2.81 | 0.883 [0.862, 0.901] · 2.93 | 0.882 [0.860, 0.901] · 3.02 | 0.875 [0.854, 0.896] · 3.17 |
| KAN no-ODE H | pooled CDF | depression | 0.909 [0.894, 0.924] · 2.74 | 0.907 [0.892, 0.922] · 2.95 | 0.910 [0.893, 0.926] · 3.13 | 0.910 [0.891, 0.928] · 3.33 |
| KAN no-ODE H | pooled CDF | anxiety | 0.901 [0.883, 0.918] · 2.98 | 0.901 [0.883, 0.918] · 3.12 | 0.898 [0.878, 0.916] · 3.19 | 0.902 [0.882, 0.920] · 3.41 |
| KAN no-ODE H | B + CV+ (empirical only) | depression | 0.914 [0.898, 0.927] · 2.77 | 0.913 [0.897, 0.927] · 3.00 | 0.914 [0.897, 0.930] · 3.21 | 0.910 [0.892, 0.928] · 3.34 |
| KAN no-ODE H | B + CV+ (empirical only) | anxiety | 0.915 [0.900, 0.929] · 3.09 | 0.918 [0.904, 0.931] · 3.30 | 0.916 [0.897, 0.932] · 3.37 | 0.910 [0.891, 0.927] · 3.50 |
| KAN no-ODE B | B: student-level repeated subsampling (main) | depression | 0.919 [0.902, 0.933] · 3.05 | 0.916 [0.900, 0.932] · 3.21 | 0.919 [0.902, 0.936] · 3.34 | 0.916 [0.899, 0.932] · 3.48 |
| KAN no-ODE B | B: student-level repeated subsampling (main) | anxiety | 0.908 [0.889, 0.926] · 3.21 | 0.911 [0.892, 0.927] · 3.33 | 0.911 [0.893, 0.927] · 3.35 | 0.912 [0.892, 0.929] · 3.58 |
| KAN no-ODE B | B1: single subsample | depression | 0.929 [0.914, 0.943] · 3.24 | 0.933 [0.918, 0.946] · 3.42 | 0.875 [0.855, 0.894] · 2.86 | 0.923 [0.906, 0.938] · 3.58 |
| KAN no-ODE B | B1: single subsample | anxiety | 0.910 [0.889, 0.928] · 3.22 | 0.900 [0.877, 0.920] · 3.41 | 0.894 [0.873, 0.911] · 3.16 | 0.926 [0.907, 0.941] · 3.81 |
| KAN no-ODE B | A: survey-level split conformal | depression | 0.899 [0.881, 0.914] · 2.83 | 0.899 [0.881, 0.915] · 2.97 | 0.900 [0.882, 0.918] · 3.10 | 0.900 [0.881, 0.918] · 3.26 |
| KAN no-ODE B | A: survey-level split conformal | anxiety | 0.878 [0.857, 0.896] · 2.88 | 0.880 [0.858, 0.899] · 3.00 | 0.880 [0.859, 0.899] · 3.06 | 0.882 [0.860, 0.902] · 3.22 |
| KAN no-ODE B | pooled CDF | depression | 0.903 [0.885, 0.918] · 2.86 | 0.903 [0.885, 0.919] · 3.02 | 0.908 [0.890, 0.925] · 3.17 | 0.906 [0.887, 0.923] · 3.34 |
| KAN no-ODE B | pooled CDF | anxiety | 0.896 [0.875, 0.913] · 3.08 | 0.895 [0.874, 0.913] · 3.18 | 0.897 [0.877, 0.914] · 3.22 | 0.901 [0.879, 0.919] · 3.44 |
| KAN no-ODE B | B + CV+ (empirical only) | depression | 0.919 [0.904, 0.933] · 2.94 | 0.920 [0.905, 0.935] · 3.14 | 0.921 [0.905, 0.936] · 3.29 | 0.916 [0.898, 0.933] · 3.39 |
| KAN no-ODE B | B + CV+ (empirical only) | anxiety | 0.920 [0.902, 0.936] · 3.24 | 0.920 [0.903, 0.936] · 3.35 | 0.919 [0.900, 0.936] · 3.40 | 0.914 [0.894, 0.932] · 3.55 |
| KAN no-ODE Hn (no gender/race) | B: student-level repeated subsampling (main) | depression | 0.916 [0.901, 0.930] · 2.85 | 0.917 [0.901, 0.931] · 3.12 | 0.917 [0.899, 0.932] · 3.25 | 0.920 [0.902, 0.936] · 3.42 |
| KAN no-ODE Hn (no gender/race) | B: student-level repeated subsampling (main) | anxiety | 0.913 [0.897, 0.928] · 3.07 | 0.914 [0.898, 0.929] · 3.23 | 0.914 [0.894, 0.931] · 3.34 | 0.917 [0.897, 0.933] · 3.58 |
| KAN no-ODE Hn (no gender/race) | B1: single subsample | depression | 0.917 [0.900, 0.932] · 2.97 | 0.913 [0.898, 0.927] · 3.08 | 0.873 [0.853, 0.892] · 2.80 | 0.911 [0.892, 0.928] · 3.32 |
| KAN no-ODE Hn (no gender/race) | B1: single subsample | anxiety | 0.899 [0.880, 0.915] · 2.90 | 0.897 [0.877, 0.916] · 3.11 | 0.891 [0.870, 0.909] · 3.07 | 0.919 [0.900, 0.936] · 3.65 |
| KAN no-ODE Hn (no gender/race) | A: survey-level split conformal | depression | 0.895 [0.877, 0.911] · 2.63 | 0.890 [0.872, 0.907] · 2.80 | 0.897 [0.878, 0.914] · 3.00 | 0.896 [0.876, 0.914] · 3.19 |
| KAN no-ODE Hn (no gender/race) | A: survey-level split conformal | anxiety | 0.878 [0.858, 0.896] · 2.74 | 0.885 [0.867, 0.902] · 2.89 | 0.878 [0.856, 0.898] · 2.99 | 0.881 [0.859, 0.902] · 3.13 |
| KAN no-ODE Hn (no gender/race) | pooled CDF | depression | 0.904 [0.888, 0.918] · 2.70 | 0.904 [0.887, 0.919] · 2.92 | 0.905 [0.888, 0.922] · 3.10 | 0.906 [0.887, 0.924] · 3.30 |
| KAN no-ODE Hn (no gender/race) | pooled CDF | anxiety | 0.904 [0.886, 0.919] · 2.95 | 0.903 [0.886, 0.920] · 3.09 | 0.901 [0.880, 0.920] · 3.18 | 0.901 [0.879, 0.920] · 3.39 |
| KAN-ODE H | B: student-level repeated subsampling (main) | depression | 0.918 [0.902, 0.933] · 2.88 | 0.920 [0.904, 0.935] · 3.16 | 0.919 [0.901, 0.936] · 3.33 | 0.919 [0.901, 0.936] · 3.50 |
| KAN-ODE H | B: student-level repeated subsampling (main) | anxiety | 0.918 [0.903, 0.931] · 3.08 | 0.915 [0.900, 0.929] · 3.26 | 0.914 [0.895, 0.930] · 3.35 | 0.919 [0.900, 0.936] · 3.64 |
| KAN-ODE H | B1: single subsample | depression | 0.924 [0.909, 0.938] · 3.05 | 0.931 [0.915, 0.944] · 3.33 | 0.885 [0.865, 0.905] · 2.95 | 0.910 [0.891, 0.929] · 3.38 |
| KAN-ODE H | B1: single subsample | anxiety | 0.910 [0.892, 0.924] · 3.03 | 0.904 [0.886, 0.921] · 3.21 | 0.889 [0.869, 0.908] · 3.08 | 0.924 [0.906, 0.941] · 3.82 |
| KAN-ODE H | A: survey-level split conformal | depression | 0.901 [0.884, 0.916] · 2.67 | 0.899 [0.881, 0.916] · 2.88 | 0.898 [0.879, 0.917] · 3.04 | 0.901 [0.881, 0.920] · 3.29 |
| KAN-ODE H | A: survey-level split conformal | anxiety | 0.887 [0.869, 0.902] · 2.75 | 0.890 [0.871, 0.906] · 2.93 | 0.881 [0.861, 0.900] · 3.03 | 0.883 [0.862, 0.903] · 3.21 |
| KAN-ODE H | pooled CDF | depression | 0.905 [0.889, 0.920] · 2.69 | 0.905 [0.888, 0.922] · 2.93 | 0.904 [0.885, 0.923] · 3.11 | 0.909 [0.889, 0.927] · 3.37 |
| KAN-ODE H | pooled CDF | anxiety | 0.905 [0.888, 0.919] · 2.92 | 0.905 [0.888, 0.920] · 3.13 | 0.897 [0.878, 0.914] · 3.18 | 0.904 [0.885, 0.923] · 3.44 |

B + CV+ is an empirical comparison only: the combination of student subsampling and CV+ is not covered by either paper's theorem. CV+'s own bound for one subsample with n = 136 students and K = 4 is 1 - 2 alpha - 0.043 (0.757 at 90%, 0.557 at 80%).

## Coverage by gender, method B, target 90% (pooled vs per-group calibration)

| Model | Calibration | Task | 1 wk: F / M · gap [CI] | 2 wk: F / M · gap [CI] | 4 wk: F / M · gap [CI] | 8 wk: F / M · gap [CI] |
|---|---|---|---|---|---|---|
| Persistence | pooled | depression | 0.965 / 0.974 · -0.009 [-0.025, +0.006] | 0.954 / 0.963 · -0.009 [-0.029, +0.011] | 0.944 / 0.950 · -0.006 [-0.030, +0.017] | 0.928 / 0.940 · -0.012 [-0.041, +0.018] |
| Persistence | pooled | anxiety | 0.965 / 0.974 · -0.009 [-0.022, +0.005] | 0.949 / 0.967 · -0.018 [-0.035, -0.002] ⚑ | 0.936 / 0.964 · -0.028 [-0.048, -0.010] ⚑ | 0.923 / 0.948 · -0.025 [-0.052, +0.000] |
| Persistence | per-group | depression | 0.965 / 0.974 · -0.009 [-0.025, +0.006] | 0.954 / 0.963 · -0.009 [-0.029, +0.011] | 0.944 / 0.964 · -0.020 [-0.041, +0.002] | 0.928 / 0.968 · -0.040 [-0.065, -0.014] ⚑ |
| Persistence | per-group | anxiety | 0.965 / 0.974 · -0.009 [-0.022, +0.005] | 0.949 / 0.967 · -0.018 [-0.035, -0.002] ⚑ | 0.936 / 0.964 · -0.028 [-0.048, -0.010] ⚑ | 0.923 / 0.964 · -0.041 [-0.064, -0.018] ⚑ |
| Linear (history) | pooled | depression | 0.907 / 0.930 · -0.023 [-0.050, +0.003] | 0.909 / 0.924 · -0.016 [-0.044, +0.011] | 0.916 / 0.918 · -0.002 [-0.035, +0.030] | 0.917 / 0.922 · -0.005 [-0.039, +0.031] |
| Linear (history) | pooled | anxiety | 0.903 / 0.927 · -0.024 [-0.049, +0.001] | 0.900 / 0.939 · -0.039 [-0.065, -0.014] ⚑ | 0.901 / 0.941 · -0.040 [-0.070, -0.010] ⚑ | 0.905 / 0.937 · -0.032 [-0.064, -0.004] ⚑ |
| Linear (history) | per-group | depression | 0.913 / 0.951 · -0.039 [-0.064, -0.017] ⚑ | 0.911 / 0.956 · -0.046 [-0.071, -0.022] ⚑ | 0.907 / 0.952 · -0.045 [-0.073, -0.016] ⚑ | 0.914 / 0.955 · -0.040 [-0.070, -0.009] ⚑ |
| Linear (history) | per-group | anxiety | 0.909 / 0.942 · -0.033 [-0.057, -0.009] ⚑ | 0.913 / 0.946 · -0.033 [-0.057, -0.010] ⚑ | 0.911 / 0.949 · -0.038 [-0.068, -0.010] ⚑ | 0.919 / 0.949 · -0.030 [-0.059, -0.004] ⚑ |
| KAN no-ODE H | pooled | depression | 0.917 / 0.931 · -0.014 [-0.044, +0.018] | 0.919 / 0.931 · -0.011 [-0.041, +0.022] | 0.922 / 0.926 · -0.004 [-0.038, +0.031] | 0.920 / 0.926 · -0.006 [-0.041, +0.031] |
| KAN no-ODE H | pooled | anxiety | 0.909 / 0.927 · -0.018 [-0.047, +0.016] | 0.903 / 0.940 · -0.037 [-0.066, -0.010] ⚑ | 0.899 / 0.942 · -0.044 [-0.075, -0.014] ⚑ | 0.904 / 0.940 · -0.036 [-0.067, -0.005] ⚑ |
| KAN no-ODE H | per-group | depression | 0.913 / 0.964 · -0.050 [-0.073, -0.025] ⚑ | 0.912 / 0.963 · -0.052 [-0.077, -0.026] ⚑ | 0.913 / 0.966 · -0.054 [-0.079, -0.029] ⚑ | 0.911 / 0.961 · -0.050 [-0.079, -0.021] ⚑ |
| KAN no-ODE H | per-group | anxiety | 0.917 / 0.936 · -0.019 [-0.048, +0.015] | 0.915 / 0.947 · -0.032 [-0.059, -0.004] ⚑ | 0.911 / 0.945 · -0.034 [-0.065, -0.005] ⚑ | 0.915 / 0.949 · -0.033 [-0.064, -0.006] ⚑ |
| KAN no-ODE B | pooled | depression | 0.921 / 0.915 · +0.006 [-0.031, +0.052] | 0.920 / 0.910 · +0.010 [-0.028, +0.056] | 0.923 / 0.911 · +0.013 [-0.025, +0.058] | 0.920 / 0.907 · +0.012 [-0.029, +0.059] |
| KAN no-ODE B | pooled | anxiety | 0.897 / 0.932 · -0.034 [-0.068, -0.002] ⚑ | 0.895 / 0.945 · -0.050 [-0.079, -0.023] ⚑ | 0.899 / 0.941 · -0.042 [-0.072, -0.014] ⚑ | 0.902 / 0.935 · -0.033 [-0.064, -0.002] ⚑ |
| KAN no-ODE B | per-group | depression | 0.908 / 0.947 · -0.039 [-0.073, +0.001] | 0.909 / 0.947 · -0.038 [-0.070, +0.000] | 0.914 / 0.947 · -0.033 [-0.066, +0.007] | 0.912 / 0.950 · -0.038 [-0.072, +0.003] |
| KAN no-ODE B | per-group | anxiety | 0.904 / 0.946 · -0.041 [-0.072, -0.009] ⚑ | 0.907 / 0.948 · -0.041 [-0.069, -0.012] ⚑ | 0.910 / 0.950 · -0.040 [-0.068, -0.013] ⚑ | 0.915 / 0.945 · -0.030 [-0.059, +0.001] |
| KAN no-ODE Hn (no gender/race) | pooled | depression | 0.914 / 0.922 · -0.008 [-0.039, +0.026] | 0.915 / 0.923 · -0.009 [-0.039, +0.023] | 0.917 / 0.918 · -0.001 [-0.034, +0.035] | 0.917 / 0.925 · -0.008 [-0.043, +0.029] |
| KAN no-ODE Hn (no gender/race) | pooled | anxiety | 0.905 / 0.932 · -0.027 [-0.054, +0.002] | 0.900 / 0.946 · -0.046 [-0.071, -0.020] ⚑ | 0.900 / 0.946 · -0.046 [-0.076, -0.016] ⚑ | 0.905 / 0.943 · -0.038 [-0.070, -0.009] ⚑ |
| KAN no-ODE Hn (no gender/race) | per-group | depression | 0.915 / 0.951 · -0.036 [-0.064, -0.008] ⚑ | 0.910 / 0.955 · -0.045 [-0.071, -0.017] ⚑ | 0.908 / 0.955 · -0.046 [-0.075, -0.018] ⚑ | 0.911 / 0.956 · -0.045 [-0.075, -0.013] ⚑ |
| KAN no-ODE Hn (no gender/race) | per-group | anxiety | 0.912 / 0.946 · -0.034 [-0.059, -0.006] ⚑ | 0.913 / 0.950 · -0.036 [-0.062, -0.011] ⚑ | 0.911 / 0.948 · -0.037 [-0.068, -0.008] ⚑ | 0.915 / 0.949 · -0.034 [-0.064, -0.007] ⚑ |
| KAN-ODE H | pooled | depression | 0.919 / 0.914 · +0.005 [-0.032, +0.047] | 0.924 / 0.913 · +0.011 [-0.027, +0.054] | 0.923 / 0.910 · +0.013 [-0.026, +0.057] | 0.920 / 0.917 · +0.002 [-0.037, +0.046] |
| KAN-ODE H | pooled | anxiety | 0.912 / 0.930 · -0.018 [-0.045, +0.010] | 0.904 / 0.940 · -0.035 [-0.062, -0.009] ⚑ | 0.901 / 0.941 · -0.040 [-0.072, -0.010] ⚑ | 0.910 / 0.938 · -0.027 [-0.060, +0.005] |
| KAN-ODE H | per-group | depression | 0.914 / 0.946 · -0.032 [-0.064, +0.003] | 0.916 / 0.950 · -0.034 [-0.065, +0.001] | 0.911 / 0.954 · -0.043 [-0.076, -0.005] ⚑ | 0.912 / 0.950 · -0.039 [-0.072, -0.001] ⚑ |
| KAN-ODE H | per-group | anxiety | 0.918 / 0.942 · -0.024 [-0.050, +0.006] | 0.916 / 0.948 · -0.033 [-0.058, -0.006] ⚑ | 0.912 / 0.945 · -0.033 [-0.064, -0.003] ⚑ | 0.919 / 0.951 · -0.032 [-0.061, -0.003] ⚑ |
| Linear quantile (CQR) | pooled | depression | 0.945 / 0.933 · +0.012 [-0.004, +0.029] | 0.945 / 0.937 · +0.008 [-0.009, +0.027] | 0.944 / 0.930 · +0.014 [-0.006, +0.034] | 0.940 / 0.933 · +0.007 [-0.017, +0.031] |
| Linear quantile (CQR) | pooled | anxiety | 0.945 / 0.944 · +0.001 [-0.014, +0.017] | 0.944 / 0.948 · -0.004 [-0.019, +0.012] | 0.945 / 0.944 · +0.001 [-0.017, +0.018] | 0.943 / 0.934 · +0.010 [-0.014, +0.033] |
| Linear quantile (CQR) | per-group | depression | 0.945 / 0.941 · +0.004 [-0.012, +0.021] | 0.945 / 0.949 · -0.004 [-0.019, +0.014] | 0.944 / 0.943 · +0.001 [-0.018, +0.021] | 0.940 / 0.956 · -0.015 [-0.039, +0.008] |
| Linear quantile (CQR) | per-group | anxiety | 0.945 / 0.949 · -0.003 [-0.019, +0.012] | 0.944 / 0.949 · -0.005 [-0.020, +0.011] | 0.945 / 0.948 · -0.002 [-0.020, +0.015] | 0.943 / 0.953 · -0.009 [-0.030, +0.011] |
| Quantile KAN H (CQR) | pooled | depression | 0.918 / 0.920 · -0.002 [-0.029, +0.028] | 0.921 / 0.919 · +0.002 [-0.024, +0.031] | 0.918 / 0.912 · +0.007 [-0.023, +0.037] | 0.921 / 0.917 · +0.004 [-0.025, +0.038] |
| Quantile KAN H (CQR) | pooled | anxiety | 0.914 / 0.915 · -0.001 [-0.027, +0.027] | 0.908 / 0.917 · -0.009 [-0.037, +0.020] | 0.902 / 0.904 · -0.002 [-0.038, +0.035] | 0.910 / 0.920 · -0.010 [-0.039, +0.020] |
| Quantile KAN H (CQR) | per-group | depression | 0.912 / 0.952 · -0.041 [-0.062, -0.016] ⚑ | 0.909 / 0.956 · -0.047 [-0.069, -0.024] ⚑ | 0.906 / 0.963 · -0.057 [-0.079, -0.033] ⚑ | 0.912 / 0.960 · -0.049 [-0.072, -0.023] ⚑ |
| Quantile KAN H (CQR) | per-group | anxiety | 0.919 / 0.929 · -0.010 [-0.036, +0.018] | 0.916 / 0.933 · -0.017 [-0.043, +0.012] | 0.908 / 0.933 · -0.025 [-0.052, +0.002] | 0.913 / 0.944 · -0.031 [-0.059, -0.003] ⚑ |

## Coverage by race, method B, target 90% (pooled vs per-group calibration)

| Model | Calibration | Task | 1 wk: white / asian · gap [CI] | 2 wk: white / asian · gap [CI] | 4 wk: white / asian · gap [CI] | 8 wk: white / asian · gap [CI] |
|---|---|---|---|---|---|---|
| Persistence | pooled | depression | 0.971 / 0.961 · +0.011 [-0.010, +0.034] | 0.959 / 0.951 · +0.008 [-0.017, +0.035] | 0.948 / 0.939 · +0.009 [-0.019, +0.038] | 0.933 / 0.922 · +0.010 [-0.025, +0.049] |
| Persistence | pooled | anxiety | 0.971 / 0.957 · +0.014 [-0.006, +0.037] | 0.956 / 0.947 · +0.009 [-0.012, +0.032] | 0.948 / 0.935 · +0.013 [-0.014, +0.045] | 0.934 / 0.918 · +0.016 [-0.018, +0.056] |
| Persistence | per-group | depression | 0.971 / 0.961 · +0.011 [-0.010, +0.034] | 0.959 / 0.959 · +0.000 [-0.024, +0.027] | 0.948 / 0.964 · -0.016 [-0.038, +0.007] | 0.948 / 0.967 · -0.019 [-0.042, +0.006] |
| Persistence | per-group | anxiety | 0.971 / 0.957 · +0.014 [-0.006, +0.037] | 0.956 / 0.961 · -0.005 [-0.023, +0.016] | 0.948 / 0.965 · -0.017 [-0.039, +0.008] | 0.934 / 0.956 · -0.022 [-0.050, +0.006] |
| Linear (history) | pooled | depression | 0.918 / 0.910 · +0.008 [-0.026, +0.041] | 0.916 / 0.905 · +0.012 [-0.022, +0.045] | 0.919 / 0.909 · +0.010 [-0.027, +0.053] | 0.920 / 0.912 · +0.008 [-0.028, +0.050] |
| Linear (history) | pooled | anxiety | 0.918 / 0.898 · +0.020 [-0.013, +0.058] | 0.920 / 0.894 · +0.025 [-0.009, +0.063] | 0.919 / 0.900 · +0.019 [-0.021, +0.067] | 0.921 / 0.896 · +0.025 [-0.019, +0.077] |
| Linear (history) | per-group | depression | 0.925 / 0.951 · -0.026 [-0.051, +0.001] | 0.927 / 0.946 · -0.018 [-0.046, +0.015] | 0.932 / 0.949 · -0.017 [-0.044, +0.016] | 0.931 / 0.948 · -0.017 [-0.046, +0.014] |
| Linear (history) | per-group | anxiety | 0.923 / 0.947 · -0.024 [-0.048, +0.001] | 0.926 / 0.947 · -0.021 [-0.047, +0.006] | 0.924 / 0.946 · -0.023 [-0.051, +0.009] | 0.928 / 0.953 · -0.025 [-0.053, +0.008] |
| KAN no-ODE H | pooled | depression | 0.919 / 0.922 · -0.003 [-0.034, +0.030] | 0.923 / 0.919 · +0.005 [-0.029, +0.040] | 0.921 / 0.921 · -0.001 [-0.038, +0.041] | 0.918 / 0.923 · -0.005 [-0.040, +0.036] |
| KAN no-ODE H | pooled | anxiety | 0.912 / 0.920 · -0.008 [-0.041, +0.025] | 0.915 / 0.907 · +0.008 [-0.030, +0.047] | 0.917 / 0.899 · +0.018 [-0.025, +0.066] | 0.918 / 0.902 · +0.016 [-0.030, +0.070] |
| KAN no-ODE H | per-group | depression | 0.937 / 0.954 · -0.017 [-0.041, +0.009] | 0.937 / 0.948 · -0.011 [-0.037, +0.017] | 0.938 / 0.951 · -0.013 [-0.041, +0.017] | 0.936 / 0.948 · -0.012 [-0.042, +0.020] |
| KAN no-ODE H | per-group | anxiety | 0.924 / 0.954 · -0.030 [-0.058, -0.003] ⚑ | 0.928 / 0.956 · -0.028 [-0.058, +0.004] | 0.924 / 0.953 · -0.029 [-0.060, +0.003] | 0.926 / 0.956 · -0.030 [-0.062, +0.010] |
| KAN no-ODE B | pooled | depression | 0.916 / 0.921 · -0.006 [-0.040, +0.031] | 0.914 / 0.920 · -0.007 [-0.043, +0.033] | 0.916 / 0.921 · -0.005 [-0.043, +0.038] | 0.913 / 0.914 · -0.001 [-0.039, +0.041] |
| KAN no-ODE B | pooled | anxiety | 0.905 / 0.909 · -0.004 [-0.042, +0.043] | 0.911 / 0.906 · +0.005 [-0.034, +0.049] | 0.910 / 0.913 · -0.003 [-0.039, +0.035] | 0.915 / 0.899 · +0.016 [-0.027, +0.067] |
| KAN no-ODE B | per-group | depression | 0.933 / 0.954 · -0.020 [-0.049, +0.010] | 0.935 / 0.950 · -0.016 [-0.046, +0.016] | 0.930 / 0.943 · -0.013 [-0.047, +0.023] | 0.927 / 0.947 · -0.020 [-0.055, +0.017] |
| KAN no-ODE B | per-group | anxiety | 0.919 / 0.955 · -0.036 [-0.065, -0.008] ⚑ | 0.925 / 0.955 · -0.030 [-0.059, +0.001] | 0.920 / 0.948 · -0.029 [-0.059, +0.002] | 0.924 / 0.951 · -0.027 [-0.060, +0.007] |
| KAN no-ODE Hn (no gender/race) | pooled | depression | 0.917 / 0.914 · +0.003 [-0.031, +0.037] | 0.919 / 0.911 · +0.009 [-0.027, +0.044] | 0.916 / 0.913 · +0.003 [-0.034, +0.046] | 0.916 / 0.921 · -0.005 [-0.042, +0.038] |
| KAN no-ODE Hn (no gender/race) | pooled | anxiety | 0.914 / 0.916 · -0.002 [-0.033, +0.031] | 0.916 / 0.908 · +0.008 [-0.029, +0.047] | 0.919 / 0.901 · +0.018 [-0.025, +0.069] | 0.917 / 0.912 · +0.006 [-0.037, +0.056] |
| KAN no-ODE Hn (no gender/race) | per-group | depression | 0.934 / 0.958 · -0.024 [-0.049, +0.002] | 0.935 / 0.949 · -0.015 [-0.042, +0.017] | 0.930 / 0.952 · -0.022 [-0.050, +0.008] | 0.931 / 0.948 · -0.017 [-0.047, +0.015] |
| KAN no-ODE Hn (no gender/race) | per-group | anxiety | 0.926 / 0.947 · -0.022 [-0.049, +0.008] | 0.930 / 0.948 · -0.018 [-0.047, +0.014] | 0.927 / 0.945 · -0.018 [-0.049, +0.014] | 0.925 / 0.950 · -0.025 [-0.058, +0.013] |
| KAN-ODE H | pooled | depression | 0.916 / 0.920 · -0.005 [-0.039, +0.030] | 0.918 / 0.922 · -0.004 [-0.040, +0.032] | 0.917 / 0.916 · +0.001 [-0.040, +0.048] | 0.916 / 0.916 · -0.000 [-0.041, +0.045] |
| KAN-ODE H | pooled | anxiety | 0.920 / 0.916 · +0.004 [-0.027, +0.037] | 0.918 / 0.906 · +0.012 [-0.021, +0.050] | 0.917 / 0.906 · +0.011 [-0.029, +0.057] | 0.921 / 0.906 · +0.015 [-0.029, +0.064] |
| KAN-ODE H | per-group | depression | 0.931 / 0.959 · -0.028 [-0.055, +0.000] | 0.931 / 0.953 · -0.022 [-0.051, +0.008] | 0.930 / 0.949 · -0.019 [-0.051, +0.015] | 0.929 / 0.945 · -0.016 [-0.051, +0.021] |
| KAN-ODE H | per-group | anxiety | 0.931 / 0.945 · -0.014 [-0.040, +0.015] | 0.932 / 0.963 · -0.031 [-0.057, -0.002] ⚑ | 0.923 / 0.948 · -0.025 [-0.057, +0.012] | 0.928 / 0.955 · -0.027 [-0.060, +0.015] |
| Linear quantile (CQR) | pooled | depression | 0.937 / 0.931 · +0.006 [-0.017, +0.030] | 0.939 / 0.935 · +0.003 [-0.022, +0.027] | 0.939 / 0.930 · +0.009 [-0.017, +0.035] | 0.937 / 0.923 · +0.014 [-0.017, +0.044] |
| Linear quantile (CQR) | pooled | anxiety | 0.947 / 0.937 · +0.010 [-0.008, +0.029] | 0.942 / 0.944 · -0.002 [-0.020, +0.018] | 0.947 / 0.934 · +0.012 [-0.011, +0.043] | 0.940 / 0.936 · +0.004 [-0.025, +0.041] |
| Linear quantile (CQR) | per-group | depression | 0.937 / 0.947 · -0.009 [-0.030, +0.012] | 0.939 / 0.948 · -0.010 [-0.032, +0.011] | 0.939 / 0.947 · -0.008 [-0.031, +0.017] | 0.941 / 0.948 · -0.007 [-0.032, +0.020] |
| Linear quantile (CQR) | per-group | anxiety | 0.947 / 0.945 · +0.002 [-0.016, +0.019] | 0.942 / 0.956 · -0.014 [-0.031, +0.005] | 0.947 / 0.946 · +0.001 [-0.021, +0.029] | 0.940 / 0.949 · -0.009 [-0.035, +0.024] |
| Quantile KAN H (CQR) | pooled | depression | 0.917 / 0.916 · +0.001 [-0.032, +0.035] | 0.917 / 0.922 · -0.005 [-0.037, +0.024] | 0.910 / 0.920 · -0.009 [-0.044, +0.026] | 0.918 / 0.915 · +0.002 [-0.033, +0.041] |
| Quantile KAN H (CQR) | pooled | anxiety | 0.910 / 0.929 · -0.019 [-0.048, +0.007] | 0.902 / 0.924 · -0.022 [-0.053, +0.008] | 0.895 / 0.914 · -0.019 [-0.055, +0.022] | 0.905 / 0.918 · -0.013 [-0.048, +0.028] |
| Quantile KAN H (CQR) | per-group | depression | 0.932 / 0.950 · -0.018 [-0.042, +0.006] | 0.937 / 0.947 · -0.010 [-0.034, +0.016] | 0.935 / 0.954 · -0.019 [-0.044, +0.007] | 0.939 / 0.946 · -0.007 [-0.035, +0.023] |
| Quantile KAN H (CQR) | per-group | anxiety | 0.920 / 0.954 · -0.033 [-0.059, -0.008] ⚑ | 0.921 / 0.958 · -0.037 [-0.062, -0.011] ⚑ | 0.921 / 0.955 · -0.034 [-0.059, -0.009] ⚑ | 0.925 / 0.950 · -0.025 [-0.056, +0.011] |

⚑ = group-gap flag (paired CI excludes zero). Per-group calibration guarantees coverage within each group; pooled calibration does not.

## Small groups (pooled calibration, method B, target 90%): too small to conclude

| Model | Group | Students | Task | 1 wk | 2 wk | 4 wk | 8 wk |
|---|---|---|---|---|---|---|---|
| Persistence | black | 8 | depression | 0.96 | 0.94 | 0.94 | 0.93 |
| Persistence | black | 8 | anxiety | 0.97 | 0.95 | 0.93 | 0.92 |
| Persistence | hispanic_other | 6 | depression | 0.99 | 0.99 | 0.99 | 0.98 |
| Persistence | hispanic_other | 6 | anxiety | 0.99 | 0.98 | 0.98 | 0.97 |
| Persistence | multiracial_native | 9 | depression | 0.95 | 0.94 | 0.94 | 0.93 |
| Persistence | multiracial_native | 9 | anxiety | 0.96 | 0.95 | 0.94 | 0.94 |
| Persistence | race_unknown | 1 | depression | 1.00 | 0.93 | 0.85 | 0.92 |
| Persistence | race_unknown | 1 | anxiety | 0.98 | 1.00 | 1.00 | 0.92 |
| Persistence | gender_both | 1 | depression | 0.99 | 0.97 | 0.97 | 0.98 |
| Persistence | gender_both | 1 | anxiety | 0.98 | 0.97 | 0.97 | 0.97 |
| Persistence | gender_unknown | 1 | depression | 1.00 | 0.93 | 0.85 | 0.92 |
| Persistence | gender_unknown | 1 | anxiety | 0.98 | 1.00 | 1.00 | 0.92 |
| KAN no-ODE H | black | 8 | depression | 0.91 | 0.89 | 0.92 | 0.93 |
| KAN no-ODE H | black | 8 | anxiety | 0.89 | 0.90 | 0.88 | 0.90 |
| KAN no-ODE H | hispanic_other | 6 | depression | 0.99 | 0.99 | 0.99 | 0.98 |
| KAN no-ODE H | hispanic_other | 6 | anxiety | 0.98 | 0.98 | 0.98 | 0.97 |
| KAN no-ODE H | multiracial_native | 9 | depression | 0.91 | 0.91 | 0.92 | 0.93 |
| KAN no-ODE H | multiracial_native | 9 | anxiety | 0.90 | 0.90 | 0.91 | 0.92 |
| KAN no-ODE H | race_unknown | 1 | depression | 0.90 | 0.84 | 0.85 | 0.90 |
| KAN no-ODE H | race_unknown | 1 | anxiety | 0.94 | 0.95 | 0.96 | 0.88 |
| KAN no-ODE H | gender_both | 1 | depression | 0.96 | 0.95 | 0.94 | 0.97 |
| KAN no-ODE H | gender_both | 1 | anxiety | 0.96 | 0.94 | 0.95 | 0.98 |
| KAN no-ODE H | gender_unknown | 1 | depression | 0.90 | 0.84 | 0.85 | 0.90 |
| KAN no-ODE H | gender_unknown | 1 | anxiety | 0.94 | 0.95 | 0.96 | 0.88 |

## Joint coverage over horizons (method B, pooled)

| Model | Task | Target | Method | Cutoffs | Joint coverage [CI] | Mean values per horizon |
|---|---|---|---|---|---|---|
| Persistence | depression | 90% | Bonferroni | available horizons | 0.985 [0.978, 0.990] | 5.73 |
| Persistence | depression | 90% | Bonferroni | all 4 horizons | 0.980 [0.968, 0.989] | 5.70 |
| Persistence | depression | 90% | max-normalised (extra) | available horizons | 0.920 [0.903, 0.936] | 4.16 |
| Persistence | depression | 90% | max-normalised (extra) | all 4 horizons | 0.906 [0.882, 0.927] | 4.17 |
| Persistence | depression | 80% | Bonferroni | available horizons | 0.943 [0.929, 0.956] | 4.64 |
| Persistence | depression | 80% | Bonferroni | all 4 horizons | 0.932 [0.911, 0.948] | 4.64 |
| Persistence | depression | 80% | max-normalised (extra) | available horizons | 0.824 [0.797, 0.847] | 3.42 |
| Persistence | depression | 80% | max-normalised (extra) | all 4 horizons | 0.786 [0.751, 0.817] | 3.43 |
| Persistence | anxiety | 90% | Bonferroni | available horizons | 0.986 [0.982, 0.990] | 5.92 |
| Persistence | anxiety | 90% | Bonferroni | all 4 horizons | 0.985 [0.978, 0.990] | 5.91 |
| Persistence | anxiety | 90% | max-normalised (extra) | available horizons | 0.917 [0.900, 0.933] | 4.42 |
| Persistence | anxiety | 90% | max-normalised (extra) | all 4 horizons | 0.900 [0.874, 0.922] | 4.43 |
| Persistence | anxiety | 80% | Bonferroni | available horizons | 0.942 [0.929, 0.955] | 4.92 |
| Persistence | anxiety | 80% | Bonferroni | all 4 horizons | 0.929 [0.907, 0.947] | 4.94 |
| Persistence | anxiety | 80% | max-normalised (extra) | available horizons | 0.877 [0.855, 0.897] | 3.90 |
| Persistence | anxiety | 80% | max-normalised (extra) | all 4 horizons | 0.853 [0.824, 0.881] | 3.89 |
| Linear (history) | depression | 90% | Bonferroni | available horizons | 0.970 [0.960, 0.978] | 4.88 |
| Linear (history) | depression | 90% | Bonferroni | all 4 horizons | 0.963 [0.947, 0.976] | 4.87 |
| Linear (history) | depression | 90% | max-normalised (extra) | available horizons | 0.912 [0.894, 0.929] | 3.88 |
| Linear (history) | depression | 90% | max-normalised (extra) | all 4 horizons | 0.893 [0.868, 0.916] | 3.87 |
| Linear (history) | depression | 80% | Bonferroni | available horizons | 0.914 [0.895, 0.931] | 3.87 |
| Linear (history) | depression | 80% | Bonferroni | all 4 horizons | 0.891 [0.864, 0.916] | 3.86 |
| Linear (history) | depression | 80% | max-normalised (extra) | available horizons | 0.817 [0.789, 0.843] | 3.06 |
| Linear (history) | depression | 80% | max-normalised (extra) | all 4 horizons | 0.776 [0.740, 0.810] | 3.06 |
| Linear (history) | anxiety | 90% | Bonferroni | available horizons | 0.970 [0.960, 0.977] | 5.15 |
| Linear (history) | anxiety | 90% | Bonferroni | all 4 horizons | 0.963 [0.950, 0.974] | 5.15 |
| Linear (history) | anxiety | 90% | max-normalised (extra) | available horizons | 0.913 [0.894, 0.929] | 4.09 |
| Linear (history) | anxiety | 90% | max-normalised (extra) | all 4 horizons | 0.885 [0.858, 0.911] | 4.09 |
| Linear (history) | anxiety | 80% | Bonferroni | available horizons | 0.909 [0.890, 0.926] | 4.05 |
| Linear (history) | anxiety | 80% | Bonferroni | all 4 horizons | 0.882 [0.854, 0.907] | 4.05 |
| Linear (history) | anxiety | 80% | max-normalised (extra) | available horizons | 0.811 [0.783, 0.837] | 3.28 |
| Linear (history) | anxiety | 80% | max-normalised (extra) | all 4 horizons | 0.769 [0.732, 0.805] | 3.29 |
| KAN no-ODE H | depression | 90% | Bonferroni | available horizons | 0.969 [0.958, 0.977] | 4.94 |
| KAN no-ODE H | depression | 90% | Bonferroni | all 4 horizons | 0.961 [0.943, 0.974] | 4.91 |
| KAN no-ODE H | depression | 90% | max-normalised (extra) | available horizons | 0.918 [0.899, 0.936] | 4.02 |
| KAN no-ODE H | depression | 90% | max-normalised (extra) | all 4 horizons | 0.899 [0.873, 0.923] | 4.01 |
| KAN no-ODE H | depression | 80% | Bonferroni | available horizons | 0.922 [0.902, 0.939] | 4.06 |
| KAN no-ODE H | depression | 80% | Bonferroni | all 4 horizons | 0.903 [0.877, 0.927] | 4.03 |
| KAN no-ODE H | depression | 80% | max-normalised (extra) | available horizons | 0.826 [0.798, 0.854] | 3.18 |
| KAN no-ODE H | depression | 80% | max-normalised (extra) | all 4 horizons | 0.789 [0.753, 0.824] | 3.18 |
| KAN no-ODE H | anxiety | 90% | Bonferroni | available horizons | 0.969 [0.959, 0.977] | 5.25 |
| KAN no-ODE H | anxiety | 90% | Bonferroni | all 4 horizons | 0.964 [0.951, 0.975] | 5.24 |
| KAN no-ODE H | anxiety | 90% | max-normalised (extra) | available horizons | 0.915 [0.894, 0.932] | 4.22 |
| KAN no-ODE H | anxiety | 90% | max-normalised (extra) | all 4 horizons | 0.894 [0.864, 0.918] | 4.22 |
| KAN no-ODE H | anxiety | 80% | Bonferroni | available horizons | 0.918 [0.897, 0.936] | 4.39 |
| KAN no-ODE H | anxiety | 80% | Bonferroni | all 4 horizons | 0.896 [0.867, 0.921] | 4.39 |
| KAN no-ODE H | anxiety | 80% | max-normalised (extra) | available horizons | 0.811 [0.780, 0.839] | 3.31 |
| KAN no-ODE H | anxiety | 80% | max-normalised (extra) | all 4 horizons | 0.769 [0.730, 0.807] | 3.31 |
| KAN no-ODE B | depression | 90% | Bonferroni | available horizons | 0.972 [0.963, 0.979] | 5.03 |
| KAN no-ODE B | depression | 90% | Bonferroni | all 4 horizons | 0.965 [0.950, 0.976] | 5.00 |
| KAN no-ODE B | depression | 90% | max-normalised (extra) | available horizons | 0.914 [0.893, 0.932] | 4.10 |
| KAN no-ODE B | depression | 90% | max-normalised (extra) | all 4 horizons | 0.898 [0.873, 0.921] | 4.09 |
| KAN no-ODE B | depression | 80% | Bonferroni | available horizons | 0.921 [0.902, 0.939] | 4.23 |
| KAN no-ODE B | depression | 80% | Bonferroni | all 4 horizons | 0.907 [0.884, 0.927] | 4.22 |
| KAN no-ODE B | depression | 80% | max-normalised (extra) | available horizons | 0.817 [0.787, 0.845] | 3.20 |
| KAN no-ODE B | depression | 80% | max-normalised (extra) | all 4 horizons | 0.778 [0.740, 0.813] | 3.20 |
| KAN no-ODE B | anxiety | 90% | Bonferroni | available horizons | 0.967 [0.956, 0.976] | 5.32 |
| KAN no-ODE B | anxiety | 90% | Bonferroni | all 4 horizons | 0.958 [0.940, 0.972] | 5.31 |
| KAN no-ODE B | anxiety | 90% | max-normalised (extra) | available horizons | 0.911 [0.889, 0.930] | 4.30 |
| KAN no-ODE B | anxiety | 90% | max-normalised (extra) | all 4 horizons | 0.893 [0.864, 0.918] | 4.29 |
| KAN no-ODE B | anxiety | 80% | Bonferroni | available horizons | 0.919 [0.898, 0.937] | 4.50 |
| KAN no-ODE B | anxiety | 80% | Bonferroni | all 4 horizons | 0.903 [0.876, 0.926] | 4.49 |
| KAN no-ODE B | anxiety | 80% | max-normalised (extra) | available horizons | 0.811 [0.780, 0.838] | 3.37 |
| KAN no-ODE B | anxiety | 80% | max-normalised (extra) | all 4 horizons | 0.782 [0.746, 0.817] | 3.37 |
| KAN no-ODE Hn (no gender/race) | depression | 90% | Bonferroni | available horizons | 0.970 [0.960, 0.978] | 4.92 |
| KAN no-ODE Hn (no gender/race) | depression | 90% | Bonferroni | all 4 horizons | 0.965 [0.948, 0.977] | 4.90 |
| KAN no-ODE Hn (no gender/race) | depression | 90% | max-normalised (extra) | available horizons | 0.913 [0.893, 0.931] | 3.93 |
| KAN no-ODE Hn (no gender/race) | depression | 90% | max-normalised (extra) | all 4 horizons | 0.889 [0.860, 0.916] | 3.92 |
| KAN no-ODE Hn (no gender/race) | depression | 80% | Bonferroni | available horizons | 0.912 [0.892, 0.930] | 3.91 |
| KAN no-ODE Hn (no gender/race) | depression | 80% | Bonferroni | all 4 horizons | 0.889 [0.861, 0.915] | 3.89 |
| KAN no-ODE Hn (no gender/race) | depression | 80% | max-normalised (extra) | available horizons | 0.819 [0.788, 0.847] | 3.14 |
| KAN no-ODE Hn (no gender/race) | depression | 80% | max-normalised (extra) | all 4 horizons | 0.781 [0.744, 0.816] | 3.14 |
| KAN no-ODE Hn (no gender/race) | anxiety | 90% | Bonferroni | available horizons | 0.970 [0.961, 0.978] | 5.15 |
| KAN no-ODE Hn (no gender/race) | anxiety | 90% | Bonferroni | all 4 horizons | 0.965 [0.952, 0.977] | 5.14 |
| KAN no-ODE Hn (no gender/race) | anxiety | 90% | max-normalised (extra) | available horizons | 0.914 [0.894, 0.932] | 4.17 |
| KAN no-ODE Hn (no gender/race) | anxiety | 90% | max-normalised (extra) | all 4 horizons | 0.898 [0.871, 0.922] | 4.17 |
| KAN no-ODE Hn (no gender/race) | anxiety | 80% | Bonferroni | available horizons | 0.913 [0.893, 0.930] | 4.15 |
| KAN no-ODE Hn (no gender/race) | anxiety | 80% | Bonferroni | all 4 horizons | 0.896 [0.870, 0.920] | 4.14 |
| KAN no-ODE Hn (no gender/race) | anxiety | 80% | max-normalised (extra) | available horizons | 0.816 [0.785, 0.842] | 3.31 |
| KAN no-ODE Hn (no gender/race) | anxiety | 80% | max-normalised (extra) | all 4 horizons | 0.777 [0.739, 0.812] | 3.32 |
| KAN-ODE H | depression | 90% | Bonferroni | available horizons | 0.968 [0.958, 0.976] | 4.94 |
| KAN-ODE H | depression | 90% | Bonferroni | all 4 horizons | 0.957 [0.939, 0.971] | 4.91 |
| KAN-ODE H | depression | 90% | max-normalised (extra) | available horizons | 0.914 [0.894, 0.932] | 4.09 |
| KAN-ODE H | depression | 90% | max-normalised (extra) | all 4 horizons | 0.894 [0.868, 0.919] | 4.07 |
| KAN-ODE H | depression | 80% | Bonferroni | available horizons | 0.916 [0.896, 0.935] | 4.13 |
| KAN-ODE H | depression | 80% | Bonferroni | all 4 horizons | 0.898 [0.872, 0.923] | 4.12 |
| KAN-ODE H | depression | 80% | max-normalised (extra) | available horizons | 0.823 [0.793, 0.852] | 3.18 |
| KAN-ODE H | depression | 80% | max-normalised (extra) | all 4 horizons | 0.787 [0.748, 0.823] | 3.19 |
| KAN-ODE H | anxiety | 90% | Bonferroni | available horizons | 0.971 [0.963, 0.979] | 5.19 |
| KAN-ODE H | anxiety | 90% | Bonferroni | all 4 horizons | 0.965 [0.953, 0.976] | 5.18 |
| KAN-ODE H | anxiety | 90% | max-normalised (extra) | available horizons | 0.913 [0.892, 0.932] | 4.25 |
| KAN-ODE H | anxiety | 90% | max-normalised (extra) | all 4 horizons | 0.889 [0.861, 0.914] | 4.25 |
| KAN-ODE H | anxiety | 80% | Bonferroni | available horizons | 0.917 [0.897, 0.935] | 4.36 |
| KAN-ODE H | anxiety | 80% | Bonferroni | all 4 horizons | 0.897 [0.869, 0.921] | 4.36 |
| KAN-ODE H | anxiety | 80% | max-normalised (extra) | available horizons | 0.815 [0.787, 0.842] | 3.33 |
| KAN-ODE H | anxiety | 80% | max-normalised (extra) | all 4 horizons | 0.776 [0.740, 0.811] | 3.33 |

## Model comparisons, method B, pooled, target 90% (paired; negative values = model 1 narrower)

| Model 1 − model 2 | Task | 1 wk: Δ values [CI] · Δ coverage | 2 wk: Δ values [CI] · Δ coverage | 4 wk: Δ values [CI] · Δ coverage | 8 wk: Δ values [CI] · Δ coverage |
|---|---|---|---|---|---|
| KAN no-ODE H − Persistence | depression | -0.87 [-0.91, -0.83] · -0.047 | -0.61 [-0.66, -0.56] · -0.034 | -0.44 [-0.49, -0.40] · -0.022 | -0.31 [-0.34, -0.27] · -0.009 |
| KAN no-ODE H − Persistence | anxiety | -0.88 [-0.93, -0.83] · -0.053 | -0.69 [-0.74, -0.63] · -0.040 | -0.62 [-0.68, -0.56] · -0.032 | -0.39 [-0.43, -0.34] · -0.016 |
| KAN no-ODE B − KAN no-ODE H | depression | +0.15 [+0.12, +0.18] · -0.002 | +0.06 [+0.03, +0.09] · -0.006 | +0.01 [-0.02, +0.04] · -0.004 | +0.02 [-0.01, +0.04] · -0.006 |
| KAN no-ODE B − KAN no-ODE H | anxiety | +0.12 [+0.08, +0.15] · -0.007 | +0.05 [+0.02, +0.08] · -0.004 | +0.00 [-0.02, +0.03] · -0.001 | +0.00 [-0.03, +0.03] · -0.003 |
| KAN no-ODE H − Linear (history) | depression | +0.23 [+0.20, +0.25] · +0.007 | +0.26 [+0.24, +0.30] · +0.009 | +0.15 [+0.12, +0.17] · +0.006 | +0.09 [+0.07, +0.11] · +0.004 |
| KAN no-ODE H − Linear (history) | anxiety | +0.16 [+0.13, +0.19] · +0.005 | +0.11 [+0.08, +0.13] · +0.002 | +0.05 [+0.03, +0.07] · -0.001 | +0.08 [+0.06, +0.11] · +0.000 |
| KAN no-ODE Hn (no gender/race) − KAN no-ODE H | depression | -0.05 [-0.08, -0.02] · -0.005 | -0.03 [-0.07, -0.00] · -0.005 | -0.07 [-0.09, -0.05] · -0.006 | -0.04 [-0.06, -0.03] · -0.002 |
| KAN no-ODE Hn (no gender/race) − KAN no-ODE H | anxiety | -0.02 [-0.05, +0.01] · -0.001 | -0.05 [-0.08, -0.02] · +0.000 | -0.01 [-0.03, +0.01] · +0.002 | +0.00 [-0.02, +0.02] · +0.002 |
| KAN-ODE H − KAN no-ODE H | depression | -0.02 [-0.05, +0.00] · -0.003 | +0.01 [-0.02, +0.04] · -0.002 | +0.00 [-0.02, +0.03] · -0.004 | +0.04 [+0.02, +0.05] · -0.003 |
| KAN-ODE H − KAN no-ODE H | anxiety | -0.01 [-0.03, +0.01] · +0.003 | -0.02 [-0.04, -0.00] · +0.001 | -0.00 [-0.02, +0.01] · +0.001 | +0.06 [+0.04, +0.08] · +0.004 |
| Quantile KAN H (CQR) − KAN no-ODE H | depression | +0.12 [+0.04, +0.20] · -0.002 | -0.02 [-0.10, +0.07] · -0.002 | -0.08 [-0.17, +0.02] · -0.007 | -0.07 [-0.15, +0.01] · -0.002 |
| Quantile KAN H (CQR) − KAN no-ODE H | anxiety | +0.14 [+0.06, +0.22] · -0.001 | +0.05 [-0.04, +0.13] · -0.004 | +0.05 [-0.04, +0.14] · -0.009 | +0.05 [-0.02, +0.12] · -0.002 |
| Linear quantile (CQR) − Linear (history) | depression | +0.40 [+0.29, +0.52] · +0.025 | +0.35 [+0.24, +0.46] · +0.027 | +0.19 [+0.08, +0.31] · +0.023 | +0.19 [+0.08, +0.29] · +0.018 |
| Linear quantile (CQR) − Linear (history) | anxiety | +0.49 [+0.39, +0.60] · +0.034 | +0.41 [+0.29, +0.51] · +0.033 | +0.36 [+0.25, +0.47] · +0.028 | +0.34 [+0.24, +0.44] · +0.026 |

## Pre-registered verdict (method B)

Reliable = no under-coverage flag (coverage CI entirely below target) overall with pooled calibration, and none within female, male, White, Asian with per-group calibration; both targets, both tasks, all horizons.

| Model | Reliable | Under-coverage overall | Under-coverage within groups | Mean coverage − 90% | Trivial group sets |
|---|---|---|---|---|---|
| Persistence | **YES** | 0 of 16 | 0 of 64 | +0.050 | 0 |
| Linear (history) | **YES** | 0 of 16 | 0 of 64 | +0.014 | 0 |
| KAN no-ODE H | **YES** | 0 of 16 | 0 of 64 | +0.018 | 0 |
| KAN no-ODE B | **YES** | 0 of 16 | 0 of 64 | +0.014 | 0 |
| KAN no-ODE Hn (no gender/race) | **YES** | 0 of 16 | 0 of 64 | +0.016 | 0 |
| KAN-ODE H | **YES** | 0 of 16 | 0 of 64 | +0.018 | 0 |
| Linear quantile (CQR) | **YES** | 0 of 16 | 0 of 64 | +0.041 | 0 |
| Quantile KAN H (CQR) | **YES** | 0 of 16 | 0 of 64 | +0.015 | 0 |
