"""Florence-2 object detection on a real image (DaViT + paged BART decoder)."""

import time

import requests
import torch
from PIL import Image

from fastencdec import LLM, SamplingParams

URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
    "transformers/tasks/car.jpg?download=true"
)
TASK = "<OD>"


def main():
    image = Image.open(requests.get(URL, stream=True).raw).convert("RGB")

    llm = LLM(
        "florence-community/Florence-2-base",
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    start = time.perf_counter()
    outputs = llm.generate(
        [{"prompt": TASK, "multi_modal_data": {"image": image}}],
        SamplingParams(max_tokens=1024, num_beams=3),
    )
    text = outputs[0].outputs[0].text
    print(f"{time.perf_counter() - start:.2f}s  {text}")

    parsed = llm.processor.post_process_generation(
        text, task=TASK, image_size=image.size
    )
    print(parsed)


if __name__ == "__main__":
    main()
