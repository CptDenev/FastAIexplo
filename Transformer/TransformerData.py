from torchvision import datasets, transforms

def load_raw(subset_size):
    train, validation, _ = datasets.load_dataset("roneneldan/TinyStories", )
    pass

def train_tokenizer(texts):
    pass

def encode_split(texts, tokenizer, path):
    pass


def get_batch(split, cfg, device):
    pass

