# import torch

# ckpt_path = "/data1/dohee/model_ckpt/clap_finetuning_best.pt"

# ckpt = torch.load(
#     ckpt_path,
#     map_location="cpu",
#     weights_only=False,   # 핵심
# )

# print("type:", type(ckpt))

# if isinstance(ckpt, dict):
#     print("keys:")
#     for k in ckpt.keys():
#         print(" -", k)

#     for key in ["state_dict", "model_state_dict", "model", "net", "optimizer", "epoch", "best_loss"]:
#         if key in ckpt:
#             print(f"\n[{key}] type:", type(ckpt[key]))

#             if isinstance(ckpt[key], dict):
#                 print(f"[{key}] first 30 keys:")
#                 for i, name in enumerate(ckpt[key].keys()):
#                     print(" ", name)
#                     if i >= 30:
#                         break
# else:
#     print(ckpt)

# import torch

# ckpt_path = "/data1/dohee/model_ckpt/clap_finetuning_best.pt"

# ckpt = torch.load(
#     ckpt_path,
#     map_location="cpu",
#     weights_only=False,
# )

# print("name:", ckpt["name"])
# print("epoch:", ckpt["epoch"])
# print("state_dict keys:", len(ckpt["state_dict"]))

# for i, k in enumerate(ckpt["state_dict"].keys()):
#     print(k, ckpt["state_dict"][k].shape if hasattr(ckpt["state_dict"][k], "shape") else "")
#     if i >= 50:
#         break

import torch

ckpt_path = "/data1/dohee/model_ckpt/clap_finetuning_best.pt"

ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False,
)

sd = ckpt["state_dict"]

print("name:", ckpt["name"])
print("epoch:", ckpt["epoch"])
print("num keys:", len(sd))

print("\n===== text-related keys =====")
cnt = 0
for k in sd.keys():
    if "text" in k.lower() or "token" in k.lower() or "transformer" in k.lower():
        print(k, sd[k].shape if hasattr(sd[k], "shape") else "")
        cnt += 1
        if cnt >= 50:
            break

print("num text-related keys shown:", cnt)