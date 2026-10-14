#!/usr/bin/env python3
"""Molmo2-8B evaluation over the downsample.csv manifest (shared logic: vlm_eval_common.py).

Follows the allenai/Molmo2-8B model card: the processor loads the video file and
samples frames itself ("uniform_last_frame", capped at max_fps=2). Default frame
budget is 128, the setting the VSTAT paper selected for Molmo2-8B.

Usage:
    python molmo2_generation.py --dry-run
    python molmo2_generation.py --limit 20

Requires: transformers==4.57.1 (pinned by the model card), torch, trust_remote_code.
"""
from __future__ import annotations

from vlm_eval_common import move_inputs, run


class Molmo2:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        # explicit bf16 (dtype="auto" can load fp32 and spill onto CPU/disk); see --device-map
        device_map = {"": args.device_map} if args.device_map.startswith("cuda") else args.device_map
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map=device_map).eval()
        placement = getattr(self.model, "hf_device_map", None) or {"": str(self.model.device)}
        offloaded = sorted({str(d) for d in placement.values() if str(d) in ("cpu", "disk", "meta")})
        n_meta = sum(p.is_meta for p in self.model.parameters())
        print(f"[molmo2] device placement: {sorted({str(d) for d in placement.values()})}; "
              f"meta params: {n_meta}", flush=True)
        if offloaded or n_meta:
            raise SystemExit(f"[molmo2] part of the model was offloaded ({offloaded or 'meta'}): not enough "
                             f"free GPU memory. Free up GPU memory (check nvidia-smi) or choose a GPU "
                             f"with ~20 GB free via --device-map cuda:N.")
        self._logged = False

    def answer(self, video_path: str, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": video_path},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, num_frames=self.args.nframes)
        inputs = move_inputs(inputs, self.model.device)
        if not self._logged:  # sanity check that --nframes reaches the processor
            print(f"[molmo2] input tokens for first item: {inputs['input_ids'].shape[1]}", flush=True)
            self._logged = True
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.args.max_new_tokens)
        return self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True).strip()


def add_args(ap):
    ap.add_argument("--device-map", default="auto",
                    help='"auto" (split across visible GPUs) or a single GPU such as "cuda:0"')
    ap.set_defaults(nframes=32)


if __name__ == "__main__":
    run(Molmo2, "allenai/Molmo2-8B", "molmo2_results.csv", add_args)