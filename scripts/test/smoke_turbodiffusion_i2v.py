from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.paths import data_dir
from videogen.inference import run_wan22_i2v
from videogen.paths import wan22_i2v_model_paths


def _read_prompt(prompts_path: Path, index: int) -> str:
    lines = [
        line.strip()
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if index < 0 or index >= len(lines):
        raise ValueError(f"prompt_index out of range: {index} (0..{len(lines) - 1})")
    return lines[index]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="TurboDiffusion Wan2.2 I2V smoke test")
    parser.add_argument(
        "--input-index",
        type=int,
        default=0,
        help="Which i2v_input_*.jpg to use (default: 0)",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="Which line in prompts.txt to use (default: 0)",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=Path("assets/turbodiffusion/i2v_inputs"),
        help="Directory containing i2v_input_*.jpg and prompts.txt",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4)
    args = parser.parse_args(argv)

    inputs_dir: Path = args.inputs_dir
    image_path = inputs_dir / f"i2v_input_{args.input_index}.jpg"
    prompts_path = inputs_dir / "prompts.txt"
    prompt = _read_prompt(prompts_path, args.prompt_index)

    model_paths = wan22_i2v_model_paths(quantized=True)
    out = (
        data_dir()
        / "turbodiffusion"
        / "smoke"
        / f"i2v_{args.input_index}_{args.prompt_index}.mp4"
    ).resolve()
    run_wan22_i2v(
        image_path=image_path.resolve(),
        prompt=prompt,
        output_path=out,
        model_paths=model_paths,
        seed=int(args.seed),
        num_steps=int(args.num_steps),
    )

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
