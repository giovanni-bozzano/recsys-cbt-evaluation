class EvaluatorLossWrapper(object):

    def evaluateRecommender(self, recommender_object):
        # - loss because it looks for the highest value
        results_dict = {'0': {'loss': - recommender_object._iteration_loss}}
        return results_dict, 'loss: ' + str(recommender_object._iteration_loss)
