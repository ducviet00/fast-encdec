"""Caption a synthetic image with Florence-2 (DaViT vision encoder + BART)."""

import time

from PIL import Image, ImageDraw

from fastencdec import Florence2LLM, SamplingParams


def make_image() -> Image.Image:
    """A simple synthetic scene: a red disc on a blue sky over green grass."""
    image = Image.new("RGB", (512, 384), (90, 150, 230))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 256, 512, 384], fill=(70, 160, 70))
    draw.ellipse([176, 96, 336, 256], fill=(220, 60, 50))
    draw.ellipse([400, 40, 470, 110], fill=(255, 240, 180))
    return image


def main():
    image = make_image()
    llm = Florence2LLM("florence-community/Florence-2-base",
                       num_blocks=1024, block_size=16)

    # One image, several tasks: the image is encoded once per request and the
    # results are decoded greedily.
    tasks = ["<CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>"]
    params = SamplingParams(max_tokens=32, num_beams=1)
    start = time.perf_counter()
    outputs = llm.generate(tasks, image, params)
    elapsed = time.perf_counter() - start
    print(f"({elapsed:.2f}s)")
    for task, output in zip(tasks, outputs):
        print(f"{task:>26}  {output}")


if __name__ == "__main__":
    main()
