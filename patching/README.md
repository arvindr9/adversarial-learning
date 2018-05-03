# Patching

Included in this folder are the files for adding adversarial patches.

## File descriptions

### Scripts

* **ad_patch_random.py** Finds the optimal adversarial image after selecting a random n * n patch
* **ad_patch_brute.py** Finds the optimal adversarial image over all n * n patches
* **ad_patch_brute_faster.py** Finds the optimal adversarial image over all n * n patches such that the
top left pixel and top right pixel are both divisible by 3 for each patch
* **ad_patch_est_eps.py** Parallel version of ad_patch_brute

### Data

* **one_iter** Data from ad_patch_brute
* **one_iter_faster** Data from ad_patch_brute_faster
* **est_eps_patch** Data from parallel script (ad_patch_est_eps)