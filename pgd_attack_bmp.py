import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "LLaVA_Vanilla"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a targeted PGD attack on a BMP image for LLaVA."
    )
    parser.add_argument("--image", required=True, help="Input BMP image path.")
    parser.add_argument(
        "--output",
        default="pgd_attacked.bmp",
        help="Output BMP path for the adversarial image.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt prefix fed to LLaVA. Example: '<image>\\nUSER: ...\\nASSISTANT: '",
    )
    parser.add_argument(
        "--target-text",
        required=True,
        help="Target text used to define the targeted attack objective.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Local LLaVA model path. Defaults to ./LLaVA_Vanilla",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=8.0 / 255.0,
        help="PGD perturbation budget in image space. Default: 8/255.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0 / 255.0,
        help="PGD step size in image space. Default: 2/255.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of PGD iterations. Default: 10.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device.",
    )
    return parser.parse_args()


def load_model_and_processor(model_path: str, device: str):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model


def build_batch(processor, image, prompt: str, target_text: str, device: str):
    prompt_inputs = processor(images=image, text=prompt, return_tensors="pt")
    full_inputs = processor(
        images=image,
        text=f"{prompt}{target_text}",
        return_tensors="pt",
    )

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    batch = {
        "input_ids": full_inputs["input_ids"].to(device),
        "attention_mask": full_inputs["attention_mask"].to(device),
        "labels": labels.to(device),
    }
    pixel_values = full_inputs["pixel_values"]
    return batch, pixel_values


def to_channel_tensor(values):
    if values is None:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    if isinstance(values, (float, int)):
        values = [float(values)] * 3
    return torch.tensor(list(values), dtype=torch.float32)


def image_bounds(pixel_values: torch.Tensor, processor, value: float):
    image_processor = processor.image_processor
    mean = to_channel_tensor(getattr(image_processor, "image_mean", None)).view(1, -1, 1, 1)
    std = to_channel_tensor(getattr(image_processor, "image_std", None)).view(1, -1, 1, 1)

    value_tensor = torch.full_like(mean, value) / std
    min_tensor = (torch.zeros_like(mean) - mean) / std
    max_tensor = (torch.ones_like(mean) - mean) / std
    return mean, std, value_tensor, min_tensor, max_tensor


def save_adv_image(pixel_values: torch.Tensor, processor, output_path: str):
    mean, std, _, _, _ = image_bounds(pixel_values, processor, value=0.0)
    image = pixel_values.detach().cpu().float() * std + mean
    image = image.clamp(0.0, 1.0)[0].permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(output_path, format="BMP")


def main():
    args = parse_args()
    image_path = Path(args.image)
    if image_path.suffix.lower() != ".bmp":
        raise ValueError("This script is intended for BMP input. Please pass a .bmp file.")

    image = Image.open(image_path).convert("RGB")
    processor, model = load_model_and_processor(args.model_path, args.device)
    batch, base_pixel_values = build_batch(
        processor=processor,
        image=image,
        prompt=args.prompt,
        target_text=args.target_text,
        device=args.device,
    )

    model_dtype = next(model.parameters()).dtype
    base_pixel_values = base_pixel_values.to(args.device, dtype=model_dtype)
    _, _, eps_tensor, min_tensor, max_tensor = image_bounds(base_pixel_values, processor, args.eps)
    _, _, alpha_tensor, _, _ = image_bounds(base_pixel_values, processor, args.alpha)
    eps_tensor = eps_tensor.to(args.device, dtype=model_dtype)
    alpha_tensor = alpha_tensor.to(args.device, dtype=model_dtype)
    min_tensor = min_tensor.to(args.device, dtype=model_dtype)
    max_tensor = max_tensor.to(args.device, dtype=model_dtype)

    adv_pixel_values = base_pixel_values.detach().clone()

    for _ in range(args.steps):
        adv_pixel_values.requires_grad_(True)
        outputs = model(
            pixel_values=adv_pixel_values,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        loss = outputs.loss
        loss.backward()

        with torch.no_grad():
            adv_pixel_values = adv_pixel_values - alpha_tensor * adv_pixel_values.grad.sign()
            adv_pixel_values = torch.max(
                torch.min(adv_pixel_values, base_pixel_values + eps_tensor),
                base_pixel_values - eps_tensor,
            )
            adv_pixel_values = torch.max(torch.min(adv_pixel_values, max_tensor), min_tensor)

    save_adv_image(adv_pixel_values, processor, args.output)
    print(f"Saved PGD adversarial BMP to: {args.output}")
    print(f"Final targeted loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
