#!/usr/bin/env python3
"""LLaVA-OneVision-2-8B evaluation over the downsample.csv manifest (shared logic: vlm_eval_common.py).

Follows the lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct model card (frame-sampling
path): the processor loads the video and samples `num_frames` frames, with
video_processor.max_pixels = 200704. Default frame budget is 64, the setting the
VSTAT paper selected for LLaVA-OV-2-8B (frame backend, not the codec backend).

Usage:
    python llava_ov2_generation.py --dry-run
    python llava_ov2_generation.py --limit 20

Requires: transformers>=5.7.0 (separate env from the 4.57 models), torch, trust_remote_code.
"""
from __future__ import annotations

from vlm_eval_common import move_inputs, run


class LlavaOneVision2:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        self.processor.video_processor.max_pixels = args.video_max_pixels
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto").eval()

    def answer(self, video_path: str, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[video_path], return_tensors="pt", padding=True,
                                num_frames=self.args.nframes)
        inputs = move_inputs(inputs, self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.args.max_new_tokens)
        return self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:],
                                               skip_special_tokens=True).strip()


def add_args(ap):
    ap.add_argument("--video-max-pixels", type=int, default=200704,
                    help="per-frame pixel cap set on the video processor (model card value)")
    ap.set_defaults(nframes=32)


if __name__ == "__main__":
    run(LlavaOneVision2, "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct", "llava_ov2_results.csv", add_args)