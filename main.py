import cv2
from Preprocessing import preprocessing, ranking
import matplotlib . pyplot as plt

query_image_paths = []
preprocessed_src_images = preprocessing(src_images)
top_k = 10

for query_image_path in query_image_paths:
    query_image = cv2.imread(query_image_path, 1)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    preprocessed_query_image = preprocessing(query_image).squeeze(0)

    ranked_list , scores = ranking(
        preprocessed_query_image,
        preprocessed_src_image,
        top_k
    )


print('Query Image:')
plt.figure(figsize =(3,3))
plt.imshow(query_image)
plt.axis('off')
plt.show ()
print(f'Top { top_k } results')
for idx in range(len(ranked_list)):
    src_image_idx = ranked_list[idx]
    similarity_score = scores [idx]
    plt.figure( figsize =(3,3))
    plt.imshow(src_images[src_image_idx ])
    plt.title (f'Similarity : { similarity_score }', fontsize =10)
    plt.axis('off')
    plt.show ()
   
