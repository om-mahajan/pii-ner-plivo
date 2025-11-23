from transformers import AutoModelForTokenClassification, AutoConfig
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str, dropout: float = None):
    """
    Create a token classification model.
    
    Args:
        model_name: HuggingFace model identifier
        dropout: Optional dropout rate. If provided, overrides default model dropout.
                Lower dropout (e.g., 0.1) can improve recall on underrepresented entities.
                Higher dropout (e.g., 0.3) increases regularization.
    """
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(LABEL2ID)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    
    # Override dropout if specified
    if dropout is not None:
        if hasattr(config, 'hidden_dropout_prob'):
            config.hidden_dropout_prob = dropout
        if hasattr(config, 'attention_probs_dropout_prob'):
            config.attention_probs_dropout_prob = dropout
        if hasattr(config, 'classifier_dropout'):
            config.classifier_dropout = dropout
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )
    return model
