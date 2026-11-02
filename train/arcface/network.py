from train.models.encoder import EmbeddingEncoder


class ArcFaceEncoder(EmbeddingEncoder):
    """Lightning-independent ArcFace encoder (classification head excluded)."""
