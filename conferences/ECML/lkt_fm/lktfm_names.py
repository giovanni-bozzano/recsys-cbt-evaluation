class LKTFMRecommenderPearson:
    RECOMMENDER_NAME = 'LKTFMRecommender_pearson'


class LKTFMRecommenderCosine:
    RECOMMENDER_NAME = 'LKTFMRecommender_cosine'


names = {
    'pearson': LKTFMRecommenderPearson.RECOMMENDER_NAME,
    'cosine': LKTFMRecommenderCosine.RECOMMENDER_NAME
}
