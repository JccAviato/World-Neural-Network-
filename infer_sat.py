import argparse, torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

LABELS = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]

def load_model(ckpt):
    blob = torch.load(ckpt, map_location="cpu")
    state = blob["state_dict"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(state)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    model = load_model(args.ckpt)
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        idx = int(logits.argmax(1).numpy()[0])
    print(f"Predicted class: {LABELS[idx]}")

if __name__ == "__main__":
    main()
