# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM3を使い、指定フォルダ内の画像それぞれに対してテキストプロンプトで
指定した領域のマスク画像を生成するスクリプト。

出力マスクは8bitグレースケールPNGで、ポジティブプロンプトに一致する領域を黒(0)、
それ以外を白(255)として保存する。ネガティブプロンプトを指定した場合、
その一致領域はポジティブプロンプトの一致領域から除外される
(= ポジティブ一致 かつ ネガティブ不一致 の領域のみが黒になる)。

使用例:
    conda run -n sam3 python tools/sam3_mask_gen.py \
        --input path/to/images --output path/to/masks \
        --prompt "person" --negative_prompt "mannequin"
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_dir: str) -> list[str]:
    """input_dir直下(サブフォルダは対象外)の画像ファイル名を取得する。"""
    names = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            names.append(name)
    return names


def get_union_mask(state: dict, height: int, width: int) -> np.ndarray:
    """stateに含まれる検出済み全インスタンスの和集合をbool配列(H, W)で返す。"""
    masks = state["masks"]
    if masks is None or len(masks) == 0:
        return np.zeros((height, width), dtype=bool)
    return masks.squeeze(1).any(dim=0).cpu().numpy()


def build_mask_image(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    negative_prompt: str | None = None,
) -> np.ndarray:
    """promptに一致(かつnegative_promptに不一致)する領域を黒(0)、
    それ以外を白(255)にした8bitマスクを返す。"""
    width, height = image.size
    state = processor.set_image(image)

    state = processor.set_text_prompt(prompt=prompt, state=state)
    match_mask = get_union_mask(state, height, width)

    if negative_prompt:
        state = processor.set_text_prompt(prompt=negative_prompt, state=state)
        negative_mask = get_union_mask(state, height, width)
        match_mask = match_mask & ~negative_mask

    mask_out = np.full((height, width), 255, dtype=np.uint8)
    mask_out[match_mask] = 0
    return mask_out


def main():
    parser = argparse.ArgumentParser(
        description="SAM3でプロンプトに一致する領域を黒、それ以外を白にしたマスク画像を生成する"
    )
    parser.add_argument("--input", required=True, help="入力画像フォルダ(直下のファイルのみ処理)")
    parser.add_argument("--output", required=True, help="マスク画像の出力フォルダ")
    parser.add_argument("--prompt", required=True, help="黒(0)にする領域を指定するポジティブなテキストプロンプト")
    parser.add_argument("--negative_prompt", default=None, help="黒(0)領域から除外する領域を指定するネガティブなテキストプロンプト(省略可)")
    parser.add_argument("--checkpoint", default=None, help="SAM3チェックポイントのパス(未指定時はHuggingFaceから自動取得)")
    parser.add_argument("--threshold", type=float, default=0.5, help="検出の確信度しきい値(デフォルト: 0.5)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="推論に使用するデバイス")

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        parser.error(f"入力フォルダが見つかりません: {args.input}")

    image_names = list_images(args.input)
    if not image_names:
        print(f"警告: {args.input} 内に処理対象の画像が見つかりませんでした。")
        return

    os.makedirs(args.output, exist_ok=True)

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    print(f"SAM3モデルを初期化しています (device={args.device}, checkpoint={args.checkpoint})...")
    model = build_sam3_image_model(device=args.device, checkpoint_path=args.checkpoint)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)

    print(f"ポジティブプロンプト: \"{args.prompt}\"")
    if args.negative_prompt:
        print(f"ネガティブプロンプト: \"{args.negative_prompt}\"")
    print(f"{len(image_names)} 件の画像を処理します。")

    for name in image_names:
        image_path = os.path.join(args.input, name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            print(f"  スキップ(画像として読み込めません): {name}")
            continue

        mask = build_mask_image(processor, image, args.prompt, args.negative_prompt)

        out_name = f"{name}.png"
        out_path = os.path.join(args.output, out_name)
        Image.fromarray(mask, mode="L").save(out_path)
        print(f"  保存しました: {out_name}")

    print("完了しました。")


if __name__ == "__main__":
    main()
