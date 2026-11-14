import numpy as np
import torch
import open_clip
from open_clip.tokenizer import SimpleTokenizer
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
bpe = SimpleTokenizer()
print("Appareil :", DEVICE)


def decouper_texte(texte, max_tokens=75, chevauchement=10):
    """Découpe un texte en morceaux de max_tokens tokens CLIP (limite : 77 avec SOT/EOT)."""
    ids = bpe.encode(texte)
    if not ids:
        return []
    pas = max_tokens - chevauchement
    morceaux = []
    for debut in range(0, len(ids), pas):
        morceaux.append(bpe.decode(ids[debut:debut + max_tokens]).strip())
        if debut + max_tokens >= len(ids):
            break
    return morceaux


@torch.no_grad()
def encoder_textes(textes, batch_size=32):
    """Retourne un tableau numpy (n, 512) d'embeddings texte normalisés."""
    vecteurs = []
    for i in range(0, len(textes), batch_size):
        batch = tokenizer(textes[i:i + batch_size]).to(DEVICE)
        emb = model.encode_text(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        vecteurs.append(emb.cpu().numpy())
    return np.concatenate(vecteurs, axis=0)


@torch.no_grad()
def encoder_images(chemins, batch_size=16):
    """Retourne un tableau numpy (n, 512) d'embeddings image normalisés."""
    vecteurs = []
    for i in range(0, len(chemins), batch_size):
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in chemins[i:i + batch_size]]
        batch = torch.stack(imgs).to(DEVICE)
        emb =