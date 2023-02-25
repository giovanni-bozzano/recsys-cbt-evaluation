class CBTRecommenderPearson:
    RECOMMENDER_NAME = 'CBTRecommender_pearson'


class CBTRecommenderCosine:
    RECOMMENDER_NAME = 'CBTRecommender_cosine'


class CBTBaselineRecommenderPearson:
    RECOMMENDER_NAME = 'CBTBaselineRecommender_pearson'


class CBTBaselineRecommenderCosine:
    RECOMMENDER_NAME = 'CBTBaselineRecommender_cosine'


class CBTSimilarityRecommenderPearson:
    RECOMMENDER_NAME = 'CBTSimilarityRecommender_pearson'


class CBTSimilarityRecommenderCosine:
    RECOMMENDER_NAME = 'CBTSimilarityRecommender_cosine'


names = {
    'standard': {
        'pearson': CBTRecommenderPearson.RECOMMENDER_NAME,
        'cosine': CBTRecommenderCosine.RECOMMENDER_NAME
    },
    'baseline': {
        'pearson': CBTBaselineRecommenderPearson.RECOMMENDER_NAME,
        'cosine': CBTBaselineRecommenderCosine.RECOMMENDER_NAME
    },
    'similarity': {
        'pearson': CBTSimilarityRecommenderPearson.RECOMMENDER_NAME,
        'cosine': CBTSimilarityRecommenderCosine.RECOMMENDER_NAME
    }
}
