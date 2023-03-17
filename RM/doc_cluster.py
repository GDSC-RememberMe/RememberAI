from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from keyword_extractor import *

model = SentenceTransformer('jhgan/ko-sroberta-multitask')
cat_lst = ["결혼", "물놀이", "졸업", "명절", "일상", "요리", "운동", "등산", "생일", "식사"]

def tokenizer(sentence):
    mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
    pos_lst = ["NNG", "NNP"]
    filtered_lst = [word for word, pos in mecab.pos(sentence) if pos in pos_lst]

    return filtered_lst

def get_cat_key(sentence, cat_lst=cat_lst):

    sentence_emb = model.encode([sentence])
    cat_emb = model.encode(cat_lst)
    distance = cosine_similarity(sentence_emb, cat_emb)
    category = cat_lst[distance.argsort()[0][-1]]

    kw_model = KeyBERT(model=model)
    
    keywords = kw_model.extract_keywords(sentence, seed_keywords=[category])

    candidates_lst = [word for word, _ in keywords]
    candidates_lst = " ".join(candidates_lst)

    fin_keyword = tokenizer(candidates_lst)[0]

    return category, fin_keyword


if __name__ == "__main__":
    test1 = "추석에 온 가족이 모여 찍은 가족사진. 오랜만에 모여 윳놀이도 하고, 고스톱도 치고 재미있었다."
    test2 = "강릉여행. 온 가족이 함께 강릉으로 여행을 갔다. 다같이 가서 회도 먹고 물놀이도 하고 너무 즐거웠다"
    test3 = "진통이 정말 심했어요. 새벽에 아버지한테 제가 전화해서 너무 아프다고 울었어요. 그래서 그날 가족들 다같이 병원으로 와서 진통하는 걸 봤었지요. 그때 아머지가 계속 손잡고 옆을 지켜주셨어요. 둘째 낳고 아버지가 보양식이며 배냇저고리며 이것저것 엄청 챙겨주셨던 기억이 나요. 낳자마자 아버지가 손수 이름 지어주겠다고 사흘밤낮동안 고민하셨던 것도 눈에 선해요"

    test_lst = [test1, test2, test3]
    for sentence in test_lst:
        print("Sentence")
        print(sentence)
        category, keyword = get_cat_key(sentence=sentence,
                                        cat_lst=cat_lst)
        print("-> Category: ", category)
        print("-> Keyword: ", keyword)
        print()
