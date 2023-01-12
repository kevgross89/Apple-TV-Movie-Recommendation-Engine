# Apple TV+ Movie Recommendation Engine

![Header](https://github.com/kevgross89/OTT-Capstone-Project/blob/main/Images/1c81beceadedd19f042225269431cd84.png)

## Motivation

Apple TV+ is looking to increase their market share in the streaming marketplace. One way they think they can attract more subscribers is by increasing the amount of movies they offer on their platform. Apple TV+ would like to explore the use of recommendation engines when selecting movies to include on their platform with the end goal being to attract more subscribers. Using various Python packages, the final machine learning model (`KNNBaseline`) has a RMSE of 0.75, meaning that it is able to predict the rating (out of 5) within 0.75 points of a given movie and user. 

## Business Understanding

According to [Business Insider](https://www.businessinsider.com/major-streaming-services-compared-cost-number-of-movies-and-shows-2022-4#prime-video-has-the-most-movies-of-any-service-but-hbo-max-has-the-most-high-quality-movies-2), as of April 11, 2022, Apple TV+ only had 44 total movies available, with only 14 of those movies considered to be "high quality" (rated 7.5+ on IMDb with 300+ votes). Compared to the rest of the streaming landscape, they not only have significantly less movies than its competitors, but also less "high quality" movies.

![Image](https://github.com/kevgross89/OTT-Capstone-Project/blob/main/Images/Streaming%20Service%20Movie%20Share.png)

[KPMG](https://advisory.kpmg.us/articles/2019/consumers-video-streaming.html) conducted a survey to better understand how consumers chose video streaming services and found that:

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

This makes a lot of sense because *Toy Story 2* and *Toy Story 3* are extremely likely to have similar plot descriptions to the original movie. 

#### Metadata-Based Model

It goes without saying that a person who loves *Toy Story* is very likely to have a thing for Disney movies. They may also prefer to watch animated movies. Unfortunately, our plot description recommender isn't able to capture all this information. Therefore the next recommendation system uses more advanced metadata, such as genres, director, and major stars. This recommender will be able to do a much better job of identifying an individual's taste for a particular director, actor, sub-genre, and so on. As we merge in the IMDB data, we are able to perform a bit more exploratory data analysis, as seen in the charts below:

![pie](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/Movie%20Genres%20Pie.png)

From the above, we can see a few takeaways:

* Drama is the single most dominant genre with over 14000 movies.
* Out of the top 5 genres, there are still many genres in the dataset. They hold 31.10% of the total genres of the movies.

![genre grid](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/Movie%20Genres%20Grid.png)

As we can see here, our `Others` category above has many subcategories within it. 

![year](https://github.com/kevgross89/Apple-TV-Movie-Recommendation-Engine/blob/main/Images/Movie%20By%20Release%20Year.png)

Additonally, we can tell that more movies have been released in recent years due to the skewed left nature of the above chart.

The metadata-based model uses a vectorizer to build document vectors. One thing we address is that actors could have the same first name, for example such as *Tom Hanks* and *Tom Cruise*. These are clearly 2 different people but as of now, our vectorizer would just look at the name *Tom* as a separate entity. Therefore, we are going to strip the spaces between the genres, cast, and director's names. Therefore, in our example we will now have *tomhanks* and *tomcruise* to differentiate between our two actors. After doing this, we have a `soup` function which has an output such as `elijahwood ianmckellen livtyler viggomortensen seanastin orlandobloom christopherlee cateblanchett action adventure drama peterjackson`.

This recommendation function will follow the basic same process as before, however we will be using a `CountVectorizer` instead of the `TF-IDFVectorizer` because the `TF-IDFVectorizer` will give less weight to actors and directors who have been in a large number of movies. We do not want to penalize artists for appearing in additional movies. After computing our `CountVectorizer`, we get the below output:

| **Index** | **Title**                                      |
|----------:|------------------------------------------------|
|     12055 |                                    Toy Story 2 |
|     12860 |                              The Polar Express |
|      8566 |        Raggedy Ann & Andy: A Musical Adventure |
|      8567 |        Raggedy Ann & Andy: A Musical Adventure |
|      9994 |        Pound Puppies and the Legend of Big Paw |
|     10280 | DuckTales the Movie: Treasure of the Lost Lamp |
|     10708 |                       Tom and Jerry: The Movie |
|     11568 |                                       Hercules |
|     11569 |                                       Hercules |
|     15110 |                                  The Wild Life |

*Toy Story 2* is obviously very close to *Toy Story*, but we can see that we do have a lot of differences from there. The second model has more children movies due to the genre, while the first model has more plot based movies.

#### Ratings Model

Now we are going to create our third type of model. One of the most basic ideas for a model is just to rank movies off of their respective ratings. However, doing a model like this has a few caveats:

* Ratings do not look at the the popularity of a movie. For example, a movie with a rating of 8.0 from 10 voters will be considered "better" than a movie with a rating of 7.9 from 10,000 voters.
* This metric will also favor movies that a smaller number of voters with extremely high ratings.

Let's take a look at the top rated movies in our dataset:

## Recommendations

## Next Steps

## Credits and Relevant Resources