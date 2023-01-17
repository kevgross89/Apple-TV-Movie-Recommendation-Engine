# Apple TV+ Movie Recommendation Engine

![header](https://github.com/kevgross89/OTT-Capstone-Project/blob/main/Images/1c81beceadedd19f042225269431cd84.png)

## Motivation

Apple TV+ is looking to increase their market share in the streaming industry. One way they think they can attract more subscribers is by increasing the number of movies they offer on their platform. Using plot data from Wikipedia, ratings and vote data from IMDb, and individual user rating data from MovieLens, Apple TV+ would like to explore the use of recommendation engines when selecting movies to include on their platform with the end goal being to attract more subscribers. Using various Python packages, the final machine learning model (`KNNBaseline`) has a RMSE of 0.74, meaning that it is able to predict the rating (out of 5) within 0.74 points of a given movie and user. 

## Navigation and Reproducibility

* [Python Notebook](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Apple%20TV%2B%20Movie%20Recommendation%20Engine.ipynb)
* [PowerPoint Presentation]()
* [Data](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/tree/main/Data)
* [Images](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/tree/main/Images)

This project was created in a Python 3 Jupyter Notebook. To re-create this project, download the three data files from the `Data` links below. Additionally, since this project uses various `Surprise` packages, please make sure that it is installed on your computer (`pip install scikit-surprise`).

## Business Understanding

According to [Business Insider](https://www.businessinsider.com/major-streaming-services-compared-cost-number-of-movies-and-shows-2022-4#prime-video-has-the-most-movies-of-any-service-but-hbo-max-has-the-most-high-quality-movies-2), as of April 11, 2022, Apple TV+ only had 44 total movies available, with only 14 of those movies considered to be "high quality" (rated 7.5+ on IMDb with 300+ votes). Compared to the rest of the streaming landscape, they not only have significantly less movies than its competitors, but also less "high quality" movies.

![Image](https://github.com/kevgross89/OTT-Capstone-Project/blob/main/Images/Streaming%20Service%20Movie%20Share.png)

[KPMG conducted a survey](https://advisory.kpmg.us/articles/2019/consumers-video-streaming.html) to better understand how consumers chose video streaming services and found that:

> "Consumers want access to a broad content mix, including original series, and a **substantial library of movies** and popular TV series"

With these findings, it is reasonable to think that adding additional movies to Apple TV+'s library will attract additional subscribers. 

## Data

This project used 3 datasets, some of which have multiple files within them:

1. [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
  * `wiki_movie_plots_deduped`
2. [IMDB](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset)
  * `ratings_data`
  * `basic_name_data`
  * `title_basics_data`
  * `title_principals_data`
3. [MovieLens 100K](https://grouplens.org/datasets/movielens/latest/)
  * `links`
  * `movies`
  * `ratings`

## Modeling

This project creates multiple personalized recommendation systems:

* **Content-based filtering**, which generates predictions by analyzing item attributes and searching for similarities between them
* **Collaborative filtering**, which generates predictions by analyzing user behavior and matching users with similar tastes
* **Hybrid filtering**, which combines two or more models.

![Rec](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/Recommendation%20Engine%20Creation.png)

### Content-Based Filtering Models

Recall that content-based filtering models generate predictions by analyzing item attributes and searching for similarities between them.

#### Plot Based Model

The first recommendation system is content-based, meaning it generates recommendations of movies based on a movie's written plot description. Using SciKit Learn's `TFIDFVecorizer`, we are able to assign weights to each word of a plot while also considering how often a word appears in our plot description. For example, many movies involve a love story, so the word love probably won't tell us much about the film itself. After fitting and transforming our movie plot descriptions, we calculate the pairwise cosine similarity score of every movie. To do this, we create a symmetric matrix for every movie plot vector and compare the score to every other movie in the matrix. The closer the score is to 1, the more similar the two movies are. Looking at the recommendations for the movie *Toy Story*, we get the below output:

| **Index** | **Title**             |
|----------:|-----------------------|
|     13574 |           Toy Story 2 |
|      5342 |  Destination Meatball |
|     15948 |           Toy Story 3 |
|      5203 |          Puny Express |
|     16619 |              Nebraska |
|     16606 |                   LUV |
|      3731 | The Barber of Seville |
|      2811 |      Girl from Havana |
|      6608 |         Mr. B Natural |
|     22145 | It's a Boy Girl Thing |

This makes a lot of sense because *Toy Story 2* and *Toy Story 3* are extremely likely to have similar plot descriptions and matches on character's names with the original movie. *Destination Meatball* sneaks into the #2 slot because the main characters are Woody Woodpecker and Buzz Buzzard, which are also the same first names as the two main characters in *Toy Story* (Woody and Buzz Lightyear).

#### Metadata-Based Model

It goes without saying that a person who loves *Toy Story* is very likely to have a thing for Disney movies. They may also prefer to watch animated movies. Unfortunately, our plot description recommender isn't able to capture all this information. Therefore the next recommendation system uses more advanced metadata, such as genres, director, and major stars. This recommender will be able to do a much better job of identifying an individual's taste for a particular director, actor, sub-genre, and so on. As we merge in the IMDB data, we are able to perform a bit more exploratory data analysis, as seen in the charts below:

![pie](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/Movie%20Genres%20Pie.png)

From the above, we can see a few takeaways:

* Drama is the single most dominant genre with over 14000 movies.
* Out of the top 5 genres, there are still many genres in the dataset. They hold 31.10% of the total genres of the movies.

![genre grid](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/Movie%20Genres%20Grid.png)

As we can see here, our `Others` category above has many subcategories within it. 

![year](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/Movie%20By%20Release%20Year.png)

Additionally, we can tell that more movies have been released in recent years due to the skewed left nature of the above chart.

The metadata-based model uses a vectorizer to build document vectors. One thing we address is that actors could have the same first name, for example such as *Tom Hanks* and *Tom Cruise*. These are clearly 2 different people but as of now, our vectorizer would just look at the name *Tom* as a separate entity. Therefore, we are going to strip the spaces between the genres, cast, and director's names. Therefore, in our example we will now have *tomhanks* and *tomcruise* to differentiate between our two actors. After doing this, we have a `soup` function which has an output such as `elijahwood ianmckellen livtyler viggomortensen seanastin orlandobloom christopherlee cateblanchett action adventure drama peterjackson`.

This recommendation function will follow the basic same process as before, however we will be using a `CountVectorizer` instead of the `TF-IDFVectorizer` because the `TF-IDFVectorizer` will give less weight to actors and directors who have been in a large number of movies. We do not want to penalize artists for appearing in additional movies. After computing our `CountVectorizer`, we get the below output:

| **Index** | **Title**                                      |
|----------:|------------------------------------------------|
|     12055 |                                    Toy Story 2 |
|     12860 |                              The Polar Express |
|      8566 |        Raggedy Ann & Andy: A Musical Adventure |
|      9994 |        Pound Puppies and the Legend of Big Paw |
|     10280 | DuckTales the Movie: Treasure of the Lost Lamp |
|     10708 |                       Tom and Jerry: The Movie |
|     11568 |                                       Hercules |
|     15110 |                                  The Wild Life |

*Toy Story 2* is obviously very close to *Toy Story*, but we can see that we do have a lot of differences from there. The second model has more children movies due to the genre, while the first model has more plot based movies.

#### Ratings Model

Now we are going to create our third type of model. One of the most basic ideas for a model is just to rank movies off of their respective ratings. However, doing a model like this has a few caveats:

* Ratings do not look at the popularity of a movie. For example, a movie with a rating of 8.0 from 10 voters will be considered "better" than a movie with a rating of 7.9 from 10,000 voters.
* This metric will also favor movies that a smaller number of voters with extremely high ratings.

Let's take a look at the top-rated movies in our dataset:

![rated](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/IMDB%2015%20Highest%20Rated.png)

And now let's look at the movies that received the most votes:

![votes](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/IMDB%2015%20Highest%20Voted.png)

Simply taking movies off of these lists are another type of recommendation engine that Apple TV+ can offer.

#### Weighted Rating Model

There is a slight relationship between the average IMDB rating and the number of IMDB votes, as we can see below: 

![ratepop](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/Rating%20and%20Popularity.png)

Due to somewhat positive relationship, we are going to come up with a weighted rating that looks at both metrics. 

The average rating for a movie in our IMDB dataset is around 6.2 on a scale of 10. If we set don't filter out any movies using the number of votes category, we get the below output:

|       |                **Title** | **averageRating** | **numVotes** | **Score** |
|------:|-------------------------:|------------------:|-------------:|----------:|
| 22505 |               Swayamvara |               9.4 |           16 |       9.4 |
| 11036 | The Shawshank Redemption |               9.3 |      2663062 |       9.3 |
| 21844 |               Kaya Taran |               9.2 |            7 |       9.2 |
| 20410 |              Hamara Ghar |               9.2 |            6 |       9.2 |
|  8039 |            The Godfather |               9.2 |      1845515 |       9.2 |

As we can see above, we have a lot of movies here that have basically no votes. Let's try this again but look at movies that have received at least 237 votes, meaning they are not in the bottom 25% of votes received. 

|       |                **Title** | **averageRating** | **numVotes** | **Score** |
|------:|-------------------------:|------------------:|-------------:|----------:|
| 11036 | The Shawshank Redemption |               9.3 |      2663062 |       9.3 |
|  8039 |            The Godfather |               9.2 |      1845515 |       9.2 |
| 24805 |                Mayabazar |               9.1 |         5149 |       9.1 |
| 10848 |         Schindler's List |               9.0 |      1348587 |       9.0 |
| 24947 |         Aha Naa Pellanta |               9.0 |         3056 |       9.0 |


Lastly, let's try this one more time but only include movies that are in the top 10% of votes received. 

|       |                **Title** | **averageRating** | **numVotes** | **Score** |
|------:|-------------------------:|------------------:|-------------:|----------:|
| 11036 | The Shawshank Redemption |               9.3 |      2663062 |       9.3 |
|  8039 |            The Godfather |               9.2 |      1845515 |       9.2 |
| 17849 |          The Dark Knight |               9.0 |      2636054 |       9.0 |
|  6130 |             12 Angry Men |               9.0 |       786416 |       9.0 |
|  8257 |    The Godfather Part II |               9.0 |      1264131 |       9.0 |

This looks significantly better. Now we have the ability to generate recommendations based on the average rating and number of votes, while taking into account a minimum number of votes needed to recommend. 

### Collaborative Filtering Recommendation Models

We are now going to move onto a new type of recommendation system using collaborative filtering. All of the previous recommender systems were content based, meaning they would recommend items based on analyzing item attributes and finding similar items. Pivoting to collaborative filtering, we will now create a recommendation system based on a user's previous behaviors. The goal of collaborative filtering systems is to provide the best user experience. Big companies such as Netflix and Amazon use these every day and are a staple of their business model.

Our dataframe contains user ratings that range from 0.5 to 5.0. Therefore, we can model this problem as an instance of supervised learning where we need to predict the rating, given a user and a movie. Although the ratings can only take in ten discrete values, we will model this as a regression problem.

We are going to split the dataframe so that 75% of a user's ratings are in the training dataset and 25% are in the testing dataset, which will require us do the `train_test_split` in a bit of an odd way. We will assume that the `userId` field is the target variable (or y) and that the other columns are the predictor variables (or X). Additionally, are going to use **root mean squared error** to assess our model performance as it is the most commonly used performance metric for regressors. 

#### Baseline Model

To start, we are going to make a baseline collaborative filter model. This model takes in a `userId` and `imdbId` as input and returns a float between 0.5 and 5.0. We make our baseline model will return a 3.0 regardless of `userId` and `imdbId`. **The RMSE returned for this model is 1.141.**

#### User-Based Mean Model

This type of filter finds users that are similar to a particular user and then recommends products that those users have liked. We will start by building a simple collaborative filter. This will take in a `userId` and `imdbId` and output the mean rating for the movie by everyone who has rated it. This filter will not distinguish between users meaning each user is assigned equal weight. 

It is also possible that some movies will only be in the test set and not the training set, therefore we will give a default rating of 3.0 like our baseline model. **After running our model, we return a RMSE of 0.963.**

#### Item-Based Models

These models are the same as the last type, except users now play the role that the items played. According to [Towards Data Science](https://towardsdatascience.com/comprehensive-guide-on-item-based-recommendation-systems-d67e40e2b75d) it was developed by Amazon in 1998 and plays a great role in Amazon's success.

At the core, item-based collaborative filters are all about finding items similar to the ones that a user has already liked. For example, let's assume that Maddie enjoyed movies 'A', 'B', and 'C'. We will then search for movies that are similar to those three movies. If we found a movie 'D', that is highly similar to one, two, or three of 'A', 'B', or 'C', we would recommend movie 'D' to Maddie because it is very similar to movies she already watched.

![item_model](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/1_9IYmsEsJJz-j4taOySkBYQ.png)

##### Cluster Models

Using clustering, it is possible to group users into a cluster and only take the users from the same clusters into consideration when predicting ratings. We will first find the k-nearest neighbors of users who have rated the movie, and then output the average rating of the nearest users for the movie. **The `KNNBasic` model has a RMSE of 0.894, the `KNNBaseline` model has a RMSE of 0.830, and the `KNNWithMeans` has a RMSE of 0.852.**

##### Singular-Value Decomposition Models

Principal Component Analysis (or PCA) transforms a *m x n* matrix into n, m-dimensional vectors (or principal components) in such a way that each component is orthogonal to the next component. It also constructs these components in such a way that the first component holds the most variance (or information), followed by the second component, and so on.

The classic version of Singular-Value Decomposition (SVD), like most other machine learning algorithms, does not work with sparse matrices. However, Simon Funk figured out a workaround for this problem, and his solution led to one of the most famous solutions in the world of recommender systems.

Funk's system took in the sparse ratings matrix, A, and constructed two dense user- and item-embedding matrices, U and V respectively. These dense matrices directly gave us the predictions for all the missing values in the original matrix, A.

**Our `SVD` model has a RMSE of 0.828 and our `SVD` with `GridSearchCV` has a RMSE of 0.824.**

### Hybrid Models

Hybrid recommenders are powerful, robust systems that combine various simpler models to give predictions. There is no correct way for a hybrid model to function - some use content and collaborative filtering techniques separately while others use content based techniques in collaborative filters.

Netflix is great example of a hybrid recommender. They have one line that typical includes a section *Because you watched This*, which is a content-based technique to show movies similar to movies you have viewed in the past. And then there can be another section such as *Top Picks*, which is a collaborative filtering technique to identify users who have liked movies that are similar to movies I have watched.

![netflix](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/T34z9GTZKh6o7JGrp9CM9k.png)

#### Hybrid Model #1

We are going to create a hybrid recommendation function that uses content and collaborative filtering techniques. First, we will pare down our dataframe to include movies that have achieved a certain score and a specific number of votes (content-based filter). From there we will load our new dataframe into `Surprise` and run a SVD package (collaborative filter). **After performing these two operations, we return a RMSE of 0.741, which is our lowest RMSE.**

|         **Type of Model**        | **RMSE** |
|:--------------------------------:|:--------:|
|    Baseline Model (User Based)   |   1.141  |
|      Mean Model (User Based)     |   0.963  |
|      KNN Basic (Item Based)      |   0.893  |
|     KNN Baseline (Item Based)    |   0.830  |
|    KNN With Means (Item Based)   |   0.852  |
|         SVD (Item Based)         |   0.828  |
| SVD With GridSearch (Item Based) |   0.824  |
|           Hybrid Model           |   0.741  |

#### Hybrid Model #2

Now that we know that using the top performing movies with more than 50k votes reduces our RMSE, we are going to use this data to hypertune a `Surprise` algorithm.

First, we are going to rotate through the below algorithms to see which one has the lowest RMSE:

|   **Algorithm** | **test_rmse** | **fit_time** | **test_time** |
|----------------:|--------------:|-------------:|--------------:|
|     KNNBaseline |      0.738610 |     0.050790 |      0.621395 |
|    KNNWithMeans |      0.747910 |     0.048250 |      0.487763 |
|   KNNWithZScore |      0.752364 |     0.059215 |      0.526507 |
|             SVD |      0.755539 |     0.401586 |      0.049812 |
|           SVDpp |      0.756807 |     3.569030 |      0.149865 |
|        KNNBasic |      0.775442 |     0.041540 |      0.451509 |
|        SlopeOne |      0.792698 |     0.014635 |      0.087261 |
|    BaselineOnly |      0.795183 |     0.006902 |      0.014330 |
|             NMF |      0.798329 |     0.493980 |      0.026299 |
|    CoClustering |      0.835452 |     0.159107 |      0.017365 |
| NormalPredictor |      1.183182 |     0.009999 |      0.018840 |

Now we know that our hybrid data (movies that have a score greater than 7.75) gives us the lowest RMSE with `KNNBaseline`, we can move on to actual findings with this information.

## Results

Utilizing the `KNNBaseline` hybrid model, we are able to generate predictions that have a RMSE of 0.739 (out of 5). By running this model through our dataset, we are able to generate predictions for each user and every movie in our dataset. When we look at the average estimated rating by title, we can determine what movies our users are most likely to resonate with based on their previous choices. 

## Recommendations

With the `KNNBaseline` model, we can predict that specific movies will be of interest to customers who are already on the platform. We would like to recommend that the movies listed below be added to Apple TV+. These well known releases should lead to more people subscribing when they want to watch one of these films. 

| **Index** |                   **Title**                   |
|----------:|:---------------------------------------------:|
|         0 |            The Shawshank Redemption           |
|       317 |                 Reservoir Dogs                |
|       448 |                   Fight Club                  |
|       666 |               The Usual Suspects              |
|       870 |                  The Departed                 |
|       977 |                   The Matrix                  |
|      1533 |                    Memento                    |
|      1692 |                Schindler's List               |
|      1912 |                     Snatch                    |
|      2005 |              Requiem for a Dream              |
|      2101 |            The Silence of the Lambs           |
|      2380 |                  Forrest Gump                 |
|      2709 | The Lord of the Rings: The Return of the King |
|      2894 |     Eternal Sunshine of the Spotless Mind     |
|      3025 |                 Spirited Away                 |
|      3112 |              Saving Private Ryan              |
|      3300 |                American Beauty                |
|      3504 |                  Pulp Fiction                 |
|      3811 |                     Fargo                     |
|      4173 |                The Dark Knight                |
|      4322 |              There Will Be Blood              |
|      4350 |            How to Train Your Dragon           |
|      4456 |                  The Pianist                  |
|      4502 |               American History X              |
|      4631 |                  Donnie Darko                 |

Additionally, after acquiring these movies, Apple TV+ should utilize the movie recommendation engines that were listed above on their platform. Some recommendation items can be specifically tailored towards a user's preferences (because you watched this, you might like this), and other recommendation items can be based on overall statistics (trending right now or highest rated movies).

## Next Steps

After acquiring these films, Apple TV+ should continue to gather information from their subscribers to see what content they interact with and what content they consume. From there, Apple TV+ can recommend content that is specifically tailored towards a user's interest.

Additionally, Apple TV+ can look at demographics for users and see where they are ahead of the curve and where they are lacking behind other OTT platforms. For example, if males ages 13-24 make up 14% of Netflix's audience and that same demo is only 5% on Apple TV+, Apple TV+ should look to figure out what content males ages 13-24 are interested in so Apple TV+ can look to add it into their library. 

![footer](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/Apple%20Logo%20Header.png)

## Credits and Relevant Resources

[Susan Li: Towards Data Science post on building and testing recommender systems](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)

[Rounak Banik: Hands-On Recommendation Systems with Python](https://learning.oreilly.com/library/view/hands-on-recommendation-systems/9781788993753/94dd3f9a-7847-47a2-9f98-b0b8f094688a.xhtml)

[Diven Sambhwani: Towards Data Science post on movie recommender systems](https://towardsdatascience.com/movie-recommender-system-part-1-7f126d2f90e2)

[Jonah Flateman: Medium post on using surprise in Python with a recommender system](https://medium.com/@jonahflateman/using-surprise-in-python-with-a-recommender-system-2d6030140926)

[Mihir Thakkar: Github repo on machine learning](https://github.com/kevgross89/Introduction-to-Machine-Learning/blob/master/Collaborative%20Filtering/Movie%20Lens%20Collaborative%20Filtering.ipynb)

[Armand Olivares: Medium post on building NLP content-based recommender systems](https://medium.com/@armandj.olivares/building-nlp-content-based-recommender-systems-b104a709c042)

[Amol Mavuduru: Medium post on building simple recommender systems with Surprise](https://towardsdatascience.com/how-you-can-build-simple-recommender-systems-with-surprise-b0d32a8e4802)