# src/train.py
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

from src.models import build_efficientnet, build_resnet50, build_densenet121

MODEL_CONFIGS = {
    "efficientnetb3": {
        "target_size": (300, 300),
        "preprocess_fn": effnet_preprocess,
        "builder": build_efficientnet
    },
    "resnet50": {
        "target_size": (224, 224),
        "preprocess_fn": resnet_preprocess,
        "builder": build_resnet50
    },
    "densenet121": {
        "target_size": (224, 224),
        "preprocess_fn": densenet_preprocess,
        "builder": build_densenet121
    }
}

# Inside train():
builder_fn = config["builder"]

model = builder_fn(
    input_image_shape=(*target_size, 3),
    num_metadata_features=num_meta_features,
    num_classes=3,
    learning_rate=learning_rate,
    dropout_rate=dropout_rate
)