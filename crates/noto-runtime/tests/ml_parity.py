"""Port parity of TIPSv2 DPT and SigLIP2 against the Hugging Face models.

`ml_port_dump` (tests/ml_ab.rs) writes, for a few rendered keyframes, the exact
pixels this port gave each model and what the port made of them. This script
runs the Hugging Face implementations (transformers >= 5, CPU PyTorch is
enough) on the same pixels and prints [PARITY] lines:

    python -I crates/noto-runtime/tests/ml_parity.py <MLAB_OUT>/parity <TIPS_DIR> <SIGLIP2_DIR>

TIPS: the port's depth (readout GELU tanh, the checkpoint's `readout_act`, and
the exact form the port used before v1.0) against Hugging Face's
`predicted_depth` resized the way the port resizes (bilinear, half-pixel).

SigLIP2: the pooled image embedding with the antialiased position resize of
the reference and with the plain bilinear one; the text vectors three ways,
because the reference itself is ambiguous: Hugging Face's tokenizer on the
lowercased prompt with and without its attention mask, and the port's own
token ids (punctuation removed by its canonicalisation) without a mask.
"""

import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer
from transformers import Siglip2Model, Tipsv2DptForDepthEstimation


def f32(path):
    return np.fromfile(path, dtype="<f4")


def cos(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def unit(t):
    return F.normalize(t, dim=-1)


def main():
    par, tips_dir, sig_dir = sys.argv[1:4]
    meta = json.load(open(os.path.join(par, "manifest.json"), encoding="utf-8"))
    torch.set_grad_enabled(False)

    # ---------- TIPSv2 DPT ----------
    tips = Tipsv2DptForDepthEstimation.from_pretrained(tips_dir, local_files_only=True).eval()
    for fr in meta["frames"]:
        k = fr["k"]
        img = np.asarray(Image.open(os.path.join(par, f"kf{k}_tips.png")).convert("RGB"), dtype=np.float32) / 255.0
        h, w = img.shape[:2]
        x = torch.from_numpy(img).permute(2, 0, 1)[None].contiguous()
        d = tips(pixel_values=x).predicted_depth
        d = F.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
        for tag in ["tanh", "erf"]:
            r = f32(os.path.join(par, f"kf{k}_tips_{tag}.f32")).reshape(h, w)
            rel = np.abs(r - d) / np.maximum(d, 1e-3)
            print(f"[PARITY] tips kf{k} {tag}: AbsRel(port, HF) {rel.mean():.2e} · max {rel.max():.2e} · median HF {np.median(d):.3f} m port {np.median(r):.3f} m ({fr['scene']})")

    # ---------- SigLIP2 ----------
    sig = Siglip2Model.from_pretrained(sig_dir, local_files_only=True).eval()
    print(f"[PARITY] siglip2 exp(t) HF {float(sig.logit_scale.exp()):.4f} port {meta['logit_scale']:.4f} · b HF {float(sig.logit_bias):.4f} port {meta['logit_bias']:.4f}")
    p = 16
    for fr in meta["frames"]:
        k = fr["k"]
        img = np.asarray(Image.open(os.path.join(par, f"kf{k}_siglip.png")).convert("RGB"), dtype=np.float32)
        gh, gw = img.shape[0] // p, img.shape[1] // p
        chw = (img / 127.5 - 1.0).transpose(2, 0, 1)
        # Hugging Face's convert_image_to_patches: (C, nH, p, nW, p) -> (nH, nW, p, p, C).
        patches = chw.reshape(3, gh, p, gw, p).transpose(1, 3, 2, 4, 0).reshape(gh * gw, -1)
        n = 256
        pix = np.zeros((1, n, patches.shape[1]), dtype=np.float32)
        pix[0, : gh * gw] = patches
        mask = np.zeros((1, n), dtype=np.int64)
        mask[0, : gh * gw] = 1
        out = sig.get_image_features(pixel_values=torch.from_numpy(pix), pixel_attention_mask=torch.from_numpy(mask), spatial_shapes=torch.tensor([[gh, gw]]))
        ref = unit(out.pooler_output)[0].numpy()
        for tag in ["aa", "bilinear"]:
            r = f32(os.path.join(par, f"kf{k}_siglip_{tag}.f32"))
            print(f"[PARITY] siglip2 image kf{k} {tag} ({gh}x{gw}): cos(port, HF) {cos(r, ref):.6f}")

    tk = Tokenizer.from_file(os.path.join(sig_dir, "tokenizer.json"))
    rust = f32(os.path.join(par, "text_rust.f32")).reshape(len(meta["prompts"]), meta["dim"])
    for i, prompt in enumerate(meta["prompts"]):
        enc = tk.encode(prompt.lower())
        ids = torch.tensor([enc.ids])
        am = torch.tensor([enc.attention_mask])
        hf_mask = unit(sig.get_text_features(input_ids=ids, attention_mask=am).pooler_output)[0].numpy()
        hf_nomask = unit(sig.get_text_features(input_ids=ids).pooler_output)[0].numpy()
        port_ids = torch.tensor([meta["ids"][i]])
        hf_portids = unit(sig.get_text_features(input_ids=port_ids).pooler_output)[0].numpy()
        same = enc.ids == meta["ids"][i]
        print(
            f"[PARITY] siglip2 text {prompt!r}: tokens same {same} (HF {sum(enc.attention_mask)} real, port mask {meta['mask_sum'][i]:.0f}) · "
            f"cos(port, HF ids no mask) {cos(rust[i], hf_nomask):.6f} · cos(port, HF ids + mask) {cos(rust[i], hf_mask):.6f} · "
            f"cos(port, port ids on HF) {cos(rust[i], hf_portids):.6f} · cos(HF mask, HF no mask) {cos(hf_mask, hf_nomask):.4f}"
        )


if __name__ == "__main__":
    main()
