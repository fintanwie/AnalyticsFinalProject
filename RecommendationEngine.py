# Load libraries
import os
import numpy as np
import pandas as pd

from surprise import SVD, Reader, KNNBaseline, Dataset

# Global variables
from surprise.model_selection import cross_validate


def clearscreen(msg):
    os.system('clear')
    os.system('CLS')
    print(msg)
    print("Processing...")
    print()


# Load the dataset and return it
def loadData():
    print("Loading data")
    # user_reviews = pd.read_csv('beer_cleaned_2000.csv')
    user_reviews = pd.read_csv('beer_reviews.csv')

    print("Finished Loading data")

    print(user_reviews.info())
    print(user_reviews.head(5))
    return user_reviews


def dropNullValues(user_reviews):
    # Check for null values
    print(user_reviews.isnull().sum())

    # Drop null row values
    user_reviews = user_reviews.dropna()
    print("After dropping null values")
    print(user_reviews.isnull().sum())

    # Result - after null values removed
    user_reviews.info()
    return user_reviews


def dropDuplicateReviews(user_reviews):
    print("Before removing duplicate reviews")
    user_reviews.info()
    # Sort by user overall rating first
    user_reviews = user_reviews.sort_values('review_overall', ascending=False)

    # Keep the highest rating from each user and drop the rest
    user_reviews = user_reviews.drop_duplicates(subset=['review_profilename', 'beer_name'], keep='first')

    # Peep structure
    print("After removing duplicate reviews")
    user_reviews.info()

    # Percent of data that are duplicates
    print("Percent of Duplicate Values:", round((1518478 - 1496263) / 1518478 * 100, 2), "%")
    return user_reviews


def removeLessThanOneRatings(user_reviews):
    # Review scores of >= 1
    user_reviews = user_reviews[(user_reviews['review_overall'] >= 1) | \
                      (user_reviews['review_appearance'] >= 1)]
    # Check it out
    user_reviews.info()
    return user_reviews


def formatBreweryName(user_reviews):
    # Split after / & keep only first string
    user_reviews['brewery_name'] = user_reviews['brewery_name'].str.split(' / ').str[0]
    return user_reviews


# Clean the dataset
def cleanDataset(user_reviews):
    print("Clean DataSet")
    print(user_reviews.info())
    print(user_reviews.head(5))

    user_reviews = dropNullValues(user_reviews)

    user_reviews = dropDuplicateReviews(user_reviews)

    user_reviews = removeLessThanOneRatings(user_reviews)

    user_reviews = formatBreweryName(user_reviews)

    return user_reviews;


# Analyse the user reviews dataset
def analyseUserReviews(user_reviews):
    print(user_reviews.head(5))


def optomizeDatasetForRecommendations(user_reviews):
    print("optomizeDatasetForRecommendations")

    # Sort the rows of dataframe by column in descending order
    user_reviews = user_reviews.sort_values(by='review_overall', ascending=False)
    print(user_reviews[['beer_name', 'review_overall']].head(5))

    print("Displaying Averaging")
    print('AVG1:', user_reviews.groupby('beer_beerid').review_overall.mean())
    print('AVG2:', user_reviews.groupby('beer_beerid').beer_beerid.head(5))

    print("Averaged Rating")
    average_rating = pd.DataFrame(user_reviews.groupby('beer_beerid').beer_beerid, user_reviews.groupby('beer_beerid').review_overall.mean()
                                  , columns=['beer_beerid', 'avg_rating'])
    #average_rating = pd.DataFrame(user_reviews.groupby('beer_beerid').review_overall.mean()
    #                              , columns=['avg_rating'])

    print("final0")
    print(average_rating.info())
    print("final1")
    print(average_rating.avg_rating)
    print("final2")
    print(average_rating.head(5))


#average_rating = average_rating.sort_values(by='beer_beerid', ascending=False)
    #print("Averaged review overall rating")
    #print(average_rating)

    #user_reviews = pd.merge(left=user_reviews, right=average_rating, how='left', left_on='beer_beerid', right_index=True)
    #print("Merged review & ratings")
    #print(user_reviews[['beer_name', 'review_overall', 'beer_avg_rating']].head(5))

    #user_reviews = user_reviews.iloc[:3]


def optomizeDatasetForRecommendationsAlt(user_reviews):
    print("optomizeDatasetForRecommendationsAlt")

    # Create Pandas DF of ratings by user and item
    ratings = user_reviews[['review_profilename', 'beer_name', 'review_overall', 'beer_beerid']]

    # Pivot table of user review counts
    user_pivot = user_reviews[['review_profilename', 'beer_name', 'beer_beerid']] \
        .pivot_table(index="review_profilename", aggfunc=("count")) \
        .reset_index() \
        .rename(columns={'beer_name': 'user_review_count'})

    # Join with ratings
    user_ct = user_pivot.merge(ratings, on='review_profilename', how='inner')

    # Pivot table of beer review counts
    beer_pivot = user_ct[['beer_name', 'review_overall']] \
        .pivot_table(index="beer_name", aggfunc=("count")) \
        .reset_index() \
        .rename(columns={'review_overall': 'beer_review_count'})

    # Join with merged user review counts / ratings
    user_beer_ct = user_ct.merge(beer_pivot, on='beer_name', how='inner')

    # Filter for user_review_count >= 50 & beer_review_count >= 100
    filt_user_beer_ct = user_beer_ct[(user_beer_ct['user_review_count'] >= 50) & \
                                     (user_beer_ct['beer_review_count'] >= 100)]

    # Remove unwanted variables
    ratings = filt_user_beer_ct.drop(['user_review_count', 'beer_review_count', 'beer_beerid_x'], axis=1)

    print("Summary 1")
    ratings.info()
    print(ratings.head(50))

    # Convert Pandas mixed data into strings
    ratings[['review_profilename', 'beer_name']] = ratings[['review_profilename', 'beer_name']].astype(str)

    # Rename columns
    ratings = ratings.rename(columns={'review_profilename': 'user', 'review_overall': 'rating', 'beer_beerid_y': 'beerID'})

    print("Summary 2")
    ratings.info()
    print(ratings.head(10))

    print("Number of unique beers")
    df = ratings.beer_name.nunique()
    print(df)

    return ratings


def reduceDataset(user_reviews):
    print("Creating dataset with selected attributes")
    user_reviews = user_reviews[['review_profilename', 'beer_name', 'review_taste',
                                 'review_palate', 'review_appearance', 'review_aroma', 'review_overall'
                                 ]].copy();
    return user_reviews


def formattingInputNamesToIDs(user_reviews_df2):
    # Create beerID for each beer
    grouped_name = user_reviews_df2.groupby('beer_name')

    temp_df = grouped_name.count()
    temp_df_idx = pd.DataFrame(temp_df.index)

    temp_df_idx['beerID'] = temp_df_idx.index
    dict_df = temp_df_idx[['beerID', 'beer_name']]

    desc_dict = dict_df.set_index('beer_name').to_dict()
    new_dict = desc_dict['beerID']

    user_reviews_df2['beerID'] = user_reviews_df2.beer_name.map(new_dict)
    #dict_df = user_reviews_df2[['beerID', 'beer_name']]

    # Create userID for each user
    group_user = user_reviews_df2.groupby("user")

    temp_df_user = group_user.count()
    temp_df_user_idx = pd.DataFrame(temp_df_user.index)

    temp_df_user_idx['userID'] = temp_df_user_idx.index
    dict_df_user = temp_df_user_idx[['userID', 'user']]

    desc_dict_user = dict_df_user.set_index('user').to_dict()
    new_dict_user = desc_dict_user['userID']

    user_reviews_df2['userID'] = user_reviews_df2.user.map(new_dict_user)

    print("formattingInputNamesToIDs")
    print(user_reviews_df2.info())
    print(user_reviews_df2.head(5))

    print("Number of unique reviewers")
    df = user_reviews_df2.userID.nunique()
    print(df)

    return dict_df, user_reviews_df2


def read_item_names(dict_df, user_reviews_df2):
    """
        return two mappings to convert raw ids into beer names
        and beer names into raw ids
    """
    file_name = dict_df
    rid_to_name = {}
    name_to_rid = {}

    # There are 3959 unique beers after removing the low rating and review count beers
    unique_beers = len(user_reviews_df2.beer_name.unique())

    for i in range(unique_beers):
        line = file_name.iloc[i]
        rid_to_name[line[0]] = line[1]
        name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


def trainEvaluateModel(merged_df2):
    print("Training model")

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(merged_df2[['userID', 'beerID', 'rating']], reader)

    trainset = data.build_full_trainset()

    sim_options = {'name': 'pearson_baseline', 'user_based': False}

    algo = KNNBaseline(sim_options=sim_options)

    algo.fit(trainset)

    print("Finished Fitting Model")

    # print("Run 5-fold cross-validation and printing results")

    # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    return algo


def get_rec(algo, dict_df, user_reviews_df2, beer_name, k_):
    """
        Input beer name and return K recommendations based on item similarity

        Input : String, integer
        Output : String
    """
    output = []

    beer = str(beer_name)

    # Read the mapping.raw id <-> beer name
    rid_to_name, name_to_rid = read_item_names(dict_df, user_reviews_df2)

    # Retrieve inner it of the beer
    beer_input_raw_id = name_to_rid[beer]
    beer_input_inner_id = algo.trainset.to_inner_iid(beer_input_raw_id)

    K = k_

    # Retrieve inner ids of the nearest neighbours of the Beer
    beer_input_neighbours = algo.get_neighbors(beer_input_inner_id, k=K)

    # Convert inner ids of the neighbours into names
    beer_input_neighbours = (algo.trainset.to_raw_iid(inner_id)
                             for inner_id in beer_input_neighbours)
    beer_input_neighbours = (rid_to_name[rid]
                             for rid in beer_input_neighbours)

    for beer_ in beer_input_neighbours:
        output.append(beer_)

    return output


def main():
    print("Running Main")

    user_reviews_raw = loadData()

    user_reviews = cleanDataset(user_reviews_raw)

    analyseUserReviews(user_reviews)

    # Cannot get this to work
    # user_reviews = optomizeDatasetForRecommendations(user_reviews);

    # Alternative Attempt
    user_reviews = optomizeDatasetForRecommendationsAlt(user_reviews);

    #user_reviews_df2 = reduceDataset(user_reviews_df2)

    #dict_df, merged_df2 = formattingInputNamesToIDs(user_reviews)
    dict_df, merged_df2 = formattingInputNamesToIDs(user_reviews)

    algo = trainEvaluateModel(merged_df2)

    # beer_name - from file
    result = get_rec(algo, dict_df, merged_df2, "Raspberry Tart", 10)
    print("Predictions :")
    print(result)
    print("END.")

    # beer_name - from file
    result = get_rec(algo, dict_df, merged_df2, "Founders KBS (Kentucky Breakfast Stout)", 10)
    print("Predictions :")
    print(result)
    print("END.")

    # beer_name - from file
    result = get_rec(algo, dict_df, merged_df2, "Apple Ale", 10)
    print("Predictions :")
    print(result)
    print("END.")


# Start Application
main()
