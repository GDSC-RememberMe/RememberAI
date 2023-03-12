from keybert import KeyBERT
# from konlpy.tag import Mecab

def tokenizer():
    
    return 

def keyword(sentence):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(sentence)
    return keywords[0][0]

if __name__ == "__main__":
    example1 = "강릉여행. 온 가족이 함께 강릉으로 여행을 갔다. 다같이 가서 회도 먹고 물놀이도 하고 너무 즐거웠다"
    example2 = "추석에 온 가족이 모여 찍은 가족사진. 오랜만에 모여 윳놀이도 하고, 고스톱도 치고 재미있었다."
    result1 = keyword(sentence=example1)
    result2 = keyword(sentence=example2)
    print(result1)
    print(result2)
