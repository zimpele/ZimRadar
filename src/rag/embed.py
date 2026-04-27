from sentence_transformers import SentenceTransformer

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"


class TextEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(MINILM_MODEL)

    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]


class ImageEmbedder:
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel

        self.model = CLIPModel.from_pretrained(CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    def embed(self, image_path: str) -> list[float]:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()
