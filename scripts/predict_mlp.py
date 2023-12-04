import torch

from scnn.models import MLPModel

if torch.cuda.is_available():
    device = "cuda"
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
else:
    raise Exception("CUDA not available")

BATCH_SIZE = int(5e2)


# Test dataset

signals = torch.load("../test_signals.pt")
targets = torch.load("../test_targets.pt")

mlp_model = MLPModel(120, 47).to(device)
mlp_model.load_state_dict(torch.load("../mlp_weights.pt"))
mlp_model.eval()

mlp_model_rot = MLPModel(120, 47).to(device)
mlp_model_rot.load_state_dict(torch.load("../mlp_weights_rot.pt"))
mlp_model_rot.eval()

mlp_preds = torch.zeros(targets.size())
mlp_preds_rot = torch.zeros(targets.size())
with torch.no_grad():
    for i in range(0, len(signals), BATCH_SIZE):
        print(f"{int(100 * i / len(signals))}%", end="\r")
        idx = torch.arange(i, i + BATCH_SIZE)
        mlp_preds[idx] = mlp_model(signals[idx].to(device)).cpu()
        mlp_preds_rot[idx] = mlp_model_rot(signals[idx].to(device)).cpu()
print("100%")

torch.save(mlp_preds, "../mlp_preds.pt")
torch.save(mlp_preds_rot, "../mlp_preds_rot.pt")


# Rotation dataset

signals = torch.load("../rotation_signals.pt")

mlp_preds = torch.zeros(signals.size()[0:2] + (47,))
mlp_preds_rot = torch.zeros(signals.size()[0:2] + (47,))
with torch.no_grad():
    for i in range(len(signals)):
        print(f"{int(100 * i / len(signals))}%", end="\r")
        mlp_preds[i] = mlp_model(signals[i].to(device)).cpu()
        mlp_preds_rot[i] = mlp_model_rot(signals[i].to(device)).cpu()
print("100%")

torch.save(mlp_preds, "../rotation_mlp_preds.pt")
torch.save(mlp_preds_rot, "../rotation_mlp_preds_rot.pt")
