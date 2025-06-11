import os, numpy as np, nibabel as nib, torch
import pandas as pd
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer
from models import build_generator
from utils import mae_psnr_ssim

# ────────── USER CONFIG ──────────────────────────────────────────── #
CHECKPOINT = r"C:\Users\MK000025\Desktop\CT_MRI\ADV_attention_10_69%_\generator_epoch_3500.pth"
TEST_DIR   = r"C:\Users\MK000025\Desktop\Baseline\Supervised_3D\baseline3DUNET\210_full_data_organised\Brain_data_150_15_15\test"
OUTPUT_DIR = r"C:\Users\MK000025\Desktop\CT_MRI\ADV_attention_10_69%_"
GENERATOR  = "attention_unet"
PATCH_SIZE = (64, 64, 64)
BASE_CH    = 32
# ─────────────────────────────────────────────────────────────────── #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pad_to_multiple(arr, m):
    d, h, w = arr.shape
    pd, ph, pw = [(m - s % m) % m for s in (d, h, w)]
    return np.pad(arr, ((0, pd), (0, ph), (0, pw))), (d, h, w)

def minmax(x, eps=1e-7):
    return (x - x.min()) / (x.max() - x.min() + eps)

# ---------------- build generator -------------------------- #
g_kw = dict(in_channels=1, out_channels=1)
if GENERATOR.lower() == "unet3d":
    g_kw["base_channels"] = BASE_CH
elif GENERATOR.lower() == "swin_unetr":
    g_kw["img_size"] = PATCH_SIZE

gen = build_generator(GENERATOR, **g_kw).to(DEVICE)
gen.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
gen.eval()

inferer = SlidingWindowInferer(
    roi_size=PATCH_SIZE,
    sw_batch_size=4,
    overlap=0.5,
    mode="gaussian"
)

# ---------------- inference loop -------------------------- #
metrics_per_patient = []
pats = [p for p in sorted(os.listdir(TEST_DIR)) if os.path.isdir(os.path.join(TEST_DIR, p))]

for pid in tqdm(pats, desc="Patients"):
    p_dir = os.path.join(TEST_DIR, pid)
    ct_nii   = nib.load(os.path.join(p_dir, "ct.nii.gz"))
    mri_nii  = nib.load(os.path.join(p_dir, "mr.nii.gz"))
    mask_nii = nib.load(os.path.join(p_dir, "mask.nii.gz"))

    ct   = ct_nii.get_fdata().astype(np.float32)
    ct   = np.clip(ct, -1000, 2000)
    ct   = (ct + 1000) / 3000.0

    mri  = mri_nii.get_fdata().astype(np.float32)
    mask = mask_nii.get_fdata().astype(bool)
    affine = ct_nii.affine

    mri_mean, mri_std = mri.mean(), mri.std() + 1e-8
    mri_z = (mri - mri_mean) / mri_std

    ct_pad, orig_shape = pad_to_multiple(ct, PATCH_SIZE[0])
    with torch.no_grad(), torch.amp.autocast("cuda" if DEVICE.type == "cuda" else "cpu"):
        inp = torch.from_numpy(ct_pad[None, None]).to(DEVICE)
        pred = inferer(inp, gen).cpu().squeeze().numpy()
    D, H, W = orig_shape
    pred = pred[:D, :H, :W]

    # Normalize for visual and metric comparison
    pred_01 = minmax(pred)
    gt_01   = minmax(mri_z)

    # Save visual output (unmasked!)
    nib.save(nib.Nifti1Image(pred_01.astype(np.float32), affine),
             os.path.join(OUTPUT_DIR, f"synth_mri_01_{pid}.nii.gz"))
    nib.save(nib.Nifti1Image(gt_01.astype(np.float32), affine),
             os.path.join(OUTPUT_DIR, f"gt_mri_01_{pid}.nii.gz"))

    # Apply mask only for metrics
    mae, _, ssim = mae_psnr_ssim(pred_01, gt_01, mask)
    pred_orig = pred * mri_std + mri_mean  # denormalize prediction
    _, psnr, _ = mae_psnr_ssim(pred_orig, mri, mask)

    print(f"{pid:>8}  MAE={mae:.3f}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}")
    metrics_per_patient.append({
        "PatientID": pid,
        "MAE": round(mae, 4),
        "PSNR": round(psnr, 2),
        "SSIM": round(ssim, 4)
    })

# ---------------- save metrics summary -------------------------- #
df = pd.DataFrame(metrics_per_patient)
mean_values = df[["MAE", "PSNR", "SSIM"]].mean()
df.loc[len(df.index)] = {
    "PatientID": "MEAN",
    "MAE": round(mean_values["MAE"], 4),
    "PSNR": round(mean_values["PSNR"], 2),
    "SSIM": round(mean_values["SSIM"], 4),
}
csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
df.to_csv(csv_path, index=False)
print(f"\n📄 Metrics saved to: {csv_path}")

# ---------------- final print summary -------------------------- #
print(
    "\nFINAL → MAE %.4f | PSNR %.2f dB | SSIM %.4f" %
    (mean_values["MAE"], mean_values["PSNR"], mean_values["SSIM"])
)
