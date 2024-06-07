import torch
import numpy as np
from transformers import ViTImageProcessor , ViTForImageClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

def preprocessing(images):
    inputs = processor(images,
                       return_tensors = 'pt').to(device)
    
    with torch.no_grad():
        output = model(**input,
                       output_hidden_states = True
        ).hidden_states[-1][:, 0, :].detach().cpu().numpy()

    return output
def cosine_similarity(query_vector, src_vectors):
    query_norm = np.linalg.norm(query_vector)
    normalized_query = query_vector / query_norm
    src_norms = np.linalg.norm(src_vectors, axis= 1)
    normalized_src = src_vectors / src_norms[:,  np.newaxis]

    cosine_similarity = np.dot(normalized_src, normalized_query)

    return cosine_similarity

def ranking(preprocessed_query_image, preprocessed_src_images, top_k = 10):
    scores = cosine_similarity(
        preprocessed_query_image,
        preprocessed_src_images
    )
    ranked_list = np.argsort(scores)[::-1][: top_k]
    scores = scores[ranked_list]

    return ranked_list, scores