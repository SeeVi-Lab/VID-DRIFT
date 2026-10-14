#!/usr/bin/env python3
"""InternVL3 evaluation over the downsample.csv manifest (shared logic: vlm_eval_common.py).

Default: frames decoded with the shared sampler (same indices as qwen_vl_utils)
and passed as one video. --native-sampling instead uses the documented HF path
(processor loads the file and samples `--nframes` itself); use it if the
processor rejects pre-decoded frames on your transformers version.

Usage:
    python internvl3_generation.py --limit 20
    python internvl3_generation.py --native-sampling --limit 20

Requires: transformers>=4.52 (native InternVL "-hf" checkpoints), torchcodec, torch.
"""
from __future__ import annotations

from vlm_eval_common import dtype_kwarg, run, sample_frames


class InternVL3:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, device_map="auto", **dtype_kwarg(torch.bfloat16)).eval()

    def answer(self, video_path: str, prompt: str) -> str:
        if self.args.native_sampling:
            messages = [{"role": "user", "content": [{"type": "video", "url": video_path},
                                                     {"type": "text", "text": prompt}]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", num_frames=self.args.nframes)
        else:
            frames = sample_frames(video_path, self.args.nframes, self.args.max_side)
            messages = [{"role": "user", "content": [{"type": "video"},
                                                     {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], videos=[frames], return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.torch.bfloat16)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.args.max_new_tokens)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def add_args(ap):
    ap.add_argument("--native-sampling", action="store_true",
                    help="let the HF processor load and sample the video itself")


if __name__ == "__main__":
    run(InternVL3, "OpenGVLab/InternVL3-8B-hf", "internvl3_results.csv", add_args)