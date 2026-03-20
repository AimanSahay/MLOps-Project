class CFRecommender:

    def __init__(self, predictions_df, items_df):
        self.predictions_df = predictions_df
        self.items_df = items_df

    def recommend_items(self, user_id, items_to_ignore=None, topn=5):

        if user_id not in self.predictions_df.columns:
            raise KeyError("User not found")

        if items_to_ignore is None:
            items_to_ignore = []

        # Get predictions
        sorted_preds = self.predictions_df[user_id] \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={user_id: 'score'})

        # Remove seen items
        recommendations = sorted_preds[
            ~sorted_preds['name_encoded'].isin(items_to_ignore)
        ].head(topn)

        # deduplicate items
        items_unique = self.items_df[['name_encoded', 'name']].drop_duplicates()

        recommendations = recommendations.merge(
            items_unique,
            on='name_encoded',
            how='left'
        )

        recommendations = recommendations.drop_duplicates(subset='name')

        return recommendations[['name', 'score']]