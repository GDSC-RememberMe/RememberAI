from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd

model = SentenceTransformer('jhgan/ko-sroberta-multitask')


def doc_cluster(evt_lst):
    embds = model.encode(evt_lst)
    # print(embds.shape) (701, 768)
    # dbscan = DBSCAN()
    # labels = dbscan.fit_predict(embds)

    kmeans = KMeans(n_clusters=5)
    labels = kmeans.fit_predict(embds)

    return_df = pd.DataFrame({"sentence" : evt_lst,
                                "label" : labels})
    
    return return_df

if __name__ == "__main__":
    with open("./oldsentence.txt", "r", encoding='UTF8') as f:
        lines = f.readlines()
        sentences = [sentence[:-1] for sentence in lines]
    
    df = doc_cluster(evt_lst=sentences)
    print(len(df['label'].unique()))
    for lab in df['label'].unique():
        print(f"Clutser {lab}")
        
        print(df[df['label'] == lab]['sentence'][:10])
        print()
