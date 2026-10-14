# VID-DRIFT
Evaluates how video degradation, and restoration of the degraded video, change video-LLM question answering. Each QA item is scored on its clean, degraded and restored video.

## Run order

**1. Build per-benchmark QA lists** → `original_csv/*.csv`
```bash
python generate_baseline_experiment_lists.py --root ./data/ --out ./original_csv
```

**2. Generate degraded videos and the manifest** → `downsamples/` + `DS_csv/downsample.csv`
```bash
python generate_downsamples.py --csv-dir original_csv \
    --out-root /path/to/downsamples --manifest DS_csv/downsample.csv --max-minutes 5
```

**3. Restore the degraded videos** → `restored/{dataset}/{condition}/{RESTORER}/`

Run a restorer from `reconstruction_methods/` over `DS_csv/downsample.csv`:
- `image_upscaling/`: FlashVSR, SwiftVR, DOVE
- `frame_interpolation/`: EDEN, LDMVFI

**4. Evaluate the video-LLMs and compare them** → `VLLM_generation/<OUTDIR>/*_results.csv` + `comparison.txt`
```bash
cd VLLM_generation
OUTDIR=FlashVSR_mvbench MAIN_ENV=vlm MOLMO_ENV=vmolmo OV2_ENV=vlm_tf5 \
    bash run_eval.sh --upscaler FlashVSR
```
To run a single model instead, use `python qwen3vl_generation.py`, `internvl3_generation.py`, `molmo2_generation.py` or `llava_ov2_generation.py`. Set the datasets, conditions and restorer with `--datasets`, `--conditions` and `--upscaler`, or edit the defaults at the top of `vlm_eval_common.py`.

**5. Summarize restoration transitions (RIE/RIC, McNemar)**
```bash
python vlm_eval_common.py FlashVSR_mvbench/*_results.csv
```

**6. Plot** → `figs/`
```bash
cd ..
python plot_mcq_acc.py VLLM_generation/FlashVSR_mvbench VLLM_generation/EDEN_mvbench --out figs/ --fmt pdf
```
