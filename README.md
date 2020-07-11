# Building a Recommender System in Python
In this blog, the content based filtering approach of building a recommender system is explained in detail, followed by a brief insight on how _Neural Networks_ can be implemented for the same.<br>
You can find the entire code on [GitHub](https://github.com/SinchanaVaidya/Movie-Recommender-System/blob/master/MovieRecommendationSystem.ipynb).
<br><br>
## What is a Recommender system?

I was always fascinated by how Amazon, Netflix and Spotify would always recommend me the products, movies and songs almost close to my personal taste. I did some digging regarding the same to know that the _Recommender System_ is the backbone to these almost accurate recommendations. So what actually are Recommender Systems?<br>

According to Wikipedia, [Recommender system](https://en.wikipedia.org/wiki/Recommender_system#:~:text=A%20recommender%20system%2C%20or%20a,primarily%20used%20in%20commercial%20applications.), or a recommendation system is a subclass of information filtering system that seeks to predict the _"rating"_ or _"preference"_ a user would give to an item.<br>
To put it in simple words, a recommender system is a simple algorithm which identifies patterns in the dataset and provides us with the most relevant information.

## Types of Recommendation systems

### 1. Recommending the Popular Items
This includes recommending the items which are most liked by users. There is no exact personalisation of your likes in this. The _Sort by Most Popular_ in _Myntra_ is an example for this.
### 2. Recommendation using a Classifier
Classifiers are parametric solutions so we just need to define some features of the user and the item. The problem of predicting whether a user (or a customer) will accept a specific recommendation can be modeled as a binary classification problem. The outcome can be 1 if the user likes it or 0 otherwise. An example for this is the _genre selection_ when you make a _Spotify_ account.
### 3. Recommendation Algorithms
Now let us come to the special class of algorithms which are tailor-made for solving the recommendation problem. There are typically two types of algorithms – Content Based and Collaborative Filtering algorithms.
1. **Content based algorithms:**<br>
        - Idea: If you like an item then you will also like **similar** item<br>
        - It is based on similarity of the items being recommended<br>
         <img src="https://miro.medium.com/max/656/1*BME1JjIlBEAI9BV5pOO5Mg.png" width="300" height="300">
2. **Collaborative filtering algorithms:**<br>
        - Idea: If a person A likes items 1, 2, 3 and B like 2,3,4 then they have similar interests and A should like item 4 and B should like item 1 <br>
        - This algorithm is entirely based on the past behavior and not on the context <br>
         <img src="https://miro.medium.com/max/656/1*x8gTiprhLs7zflmEn1UjAQ.png" width="300" height="300">
          <br>
## What is Correlation?
The statistical relationship between two variables is referred to as their correlation. A correlation can be:
- **Positive Correlation:** Both variables change in the same direction.
- **Neutral Correlation:** No relationship in the change of the variables.
- **Negative Correlation:** Variables change in opposite directions.

## Building a Recommender System
Here, we will be making use of the _content-based filtering_ algorithm.<br>
<br>

**Importing Libraries:**<br>
Datasets used: We are going to use the _movies_ and _ratings_ datasets available [here](https://drive.google.com/file/d/1Dn1BZD3YxgBQJSIjbfNnmCFlDW2jdQGD/view) to build a simple item similarity based recommender system.<br>
Let us dive into coding!<br>
First, let us import the required libraries and read the _ratings_ dataset. 
```ruby
import pandas as pd
df= pd.read_csv('ratings.csv')
df.head()
```
We can observe that, the dataset has 4 columns, namely: userId, movieId, rating and timestamp. <br>
- **userId** indicates the unique ID of the film critic. 
- **rating** column contains the ratings given by the critics.
- **timestamp** is the unix seconds since 1/1/1970 UTC.
- **movieId** contains the unique ID of the movie, and is linked with the other dataset.<br>
<br>
Now, let us read the _movies_ dataset , which is pretty straight-forward.

```ruby
movies= pd.read_csv('movies.csv')
movies.head()
```
<br>

Now let us merge the two data sets into one, using the column _movieId_ as the key. The _pandas_ dataframe [merge()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html) is used for the same. The _on_ parameter specifies column or index level names to join on. 
```ruby
df= pd.merge( df, movies, on='movieId')
df.head()
```
<br>

**Exploratory Data Analysis:**<br>
Let us import the libraries required for visualising data.
```ruby
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
<br>

Let's create a dataframe with average ratings. A [groupby()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) operation involves some combination of splitting the object, applying a function, and combining the results. This can be used to group large amounts of data and compute operations on these groups.
```ruby
df1= df.groupby('title')['rating']
df1.mean().sort_values(ascending=False).head()
```
<br>

Now, we compute how many ratings were given with respect to each movie.
```ruby
df1.count().sort_values(ascending=False).head()
```
<br>

Putting the mean value in the dataframe.
```ruby
ratings= pd.DataFrame( df1.mean())
ratings.head()
```
<br>

Now, we integrate the rating counts to this dataframe.
```ruby
ratings['no_of_ratings']= pd.DataFrame(df1.count())
ratings.head()
```
<br>

Let us go ahead and visualise the data, by plotting a _histogram_ with respect to the number of ratings.
```ruby
plt.figure(figsize=(10,7))
ratings['no_of_ratings'].hist(bins=100)
```
<img src="https://user-images.githubusercontent.com/66874666/87218907-b051e080-c374-11ea-87da-501ed19e8d6e.png" width="500" height="300">
<br>

Now, let us plot a _histogram_ with respect to the ratings.
```ruby
plt.figure(figsize=(10,7))
ratings['rating'].hist(bins=100)
```
<img src="https://user-images.githubusercontent.com/66874666/87218910-b3e56780-c374-11ea-95de-b33f41efda40.png" width="500" height="300"><br>
If you study the plot, you can observe a _Normal Gaussian System_. That is, it follows a **Gaussian curve** (bell-shaped curve). There are outliers, but they can be ignored.
<br><br>

Let us plot a _jointplot_ for the same. Jointplot is a combination of scatterplot and histogram.
```ruby
sns.jointplot(x='rating', y='no_of_ratings', data=ratings, alpha=0.5)
```
<img src="https://user-images.githubusercontent.com/66874666/87218968-19395880-c375-11ea-8d4e-28d4718a3825.png" width="500" height="500"><br>
We can visualise that, where the points are more dense, most number of ratings have been given there.
<br><br>
<br>

**Recommending Similar Movies:**<br>
We create a matrix or table that has user IDs on one axis, and movie titles on the other axis.<br>
Each cell will consist of ratings that the critics have given for the movie. Also, there will be a lot of _NaN_ values, because people may not have watched all of the movies listed.<br>
For those unfamiliar with pivot tables, it’s basically a table where each cell is a filtered count (another way to think of it is as a 2 or more-dimensional groupby). You can use the [pivot_table()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html) function for the same.
```ruby
movielist= df.pivot_table(index= 'userId', columns= 'title', values='rating')
movielist.head()
```
<br>

Now, we sort the values to get the most rated movie.
```ruby
ratings.sort_values('no_of_ratings', ascending=False).head()
```
<br>

We will now find the **correlation** with the pivot table and see which movie will be recommended.<br>
Let's choose 2 movies. I'll choose _Mission Impossible: Ghost Protocol_ (because Anil Kapoor!!) and _Ironman_ (because come on, who hates ironman?).
<br>

Let us see the _user ratings_ for these two movies.
```ruby
mission_ratings= movielist['Mission: Impossible - Ghost Protocol (2011)']
ironman_ratings= movielist['Iron Man (2008)']
mission_ratings.head()
```
<br>

Pandas dataframe [.corrwith()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corrwith.html) is used to compute pair wise correlation between rows or columns of two DataFrame objects. If the shape of two dataframe objects is not the same, then the corresponding correlation value will be a _NaN_ value. <br>
Note: The correlation of a variable with itself is 1.
```ruby
recommended_mission= movielist.corrwith(mission_ratings)
recommeneded_ironman= moviemat.corrwith(ironman_ratings)
```
<br>

Let us clean the data by removing the _NaN_ values, and let's store the data **(of Mission Impossible)** in a dataframe. <br>
Here, we must note that, higher the correlation, higher correlated the predicted movie is to Mission Impossible, and thus will be recommended.
```ruby
corr_mission= pd.DataFrame(recommended_mission, columns=['Correlation'])
corr_mission.dropna(inplace=True)
corr_mission.head()
```
<br>

If we sort the dataframe by correlation, we should theoretically get the most similar movies to Mission Impossible. <br>
But, we may also get some results that don't really make sense. That's because lot of movies may have been watched only once by the film critics who happen to have watched Mission Impossible too.
```ruby
corr_mission.sort_values('Correlation',ascending=False).head(10)
```
<br>

We need to fix this. This can be done by filtering out movies that have less than, say,  100 reviews (This value was chosen by studying the histogram plotted earlier).<br>
So, let's join the column containing count of ratings to the dataframe.
```ruby
corr_mission= corr_mission.join(ratings['no_of_ratings'])
corr_mission.head()
```
<br>

Now, all left to do is to sort the values, and notice how the recommendations make a lot more sense! (Maybe I should watch Jaws next!)
```ruby
corr_mission[corr_mission['no_of_ratings']>100].sort_values('Correlation',ascending=False).head()
```
<img src="https://user-images.githubusercontent.com/66874666/87219163-a29d5a80-c376-11ea-9107-fc517539fa9a.png" width="450" height="250"><br>
<br><br>

Now, let's do the same for the movie **Ironman** (How did the system know I was gonna watch Mask next?!)
```ruby
corr_ironman= pd.DataFrame(recommeneded_ironman, columns=['Correlation'])
corr_ironman.dropna(inplace=True)
corr_ironman= corr_ironman.join(ratings['no_of_ratings'])
corr_ironman[corr_ironman['no_of_ratings']>100].sort_values('Correlation',ascending=False).head()
```
<img src="https://user-images.githubusercontent.com/66874666/87219168-a5984b00-c376-11ea-918f-bac60b8dc9c6.png" width="450" height="250"><br>
<br><br>

## A Neural Network Inspired Approach for Improved Movie Recommendations
### Multivariate Movie Recommendation Model
<img src="https://user-images.githubusercontent.com/66874666/87219511-7df6b200-c379-11ea-9dfd-48c4fbd822dd.png" width="700" height="450"><br>
<br>

Multivariate approach is based on three modules:
1. Mobile app
2. Multivariate recommender
3. Web scraper
<br>

Users get recommendation services using a mobile application. _**Mobile app module**_ provides information such as _user's profile, search history and watch history_ to the recommender module.<br>
<br>
_**Recommendation Module**_ is based on deep learning _NLP_(Natural Language Processing) module and _computation module_.<br> 
**NLP Module** preprocesses the data fetched (user's reviews) using _[tokenizer](https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/), [stemmer](https://medium.com/@tusharsri/nlp-a-quick-guide-to-stemming-60f1ca5db49e), and [POStagger](https://nlp.stanford.edu/software/tagger.shtml)_, and then _semantically_ analyses the reviews, and extracts the semantic emotions about movies.<br>
**Semantic parsing** is the task of converting a natural language utterance to a logical form: a machine-understandable representation of its meaning. Semantic parser work is based on deep machine learning algorithm recurrent neural network ([RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)/[LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)) attention with user movie attention (UMA).<br>
<br>
Semantic emotion is classified to 5 major classes based on their relative _semantic scores_:
- Highly favourable
- Favourable
- Averagely favourable
- Unfavourable
- Highly unfavourable
<br>

_**Computation module**_ normalises the quantitative data (Twitter likes, votes ratings etc), and the _normalised scores_ and _semantic emotional scores_ are evaluated to generate the recommended movie list, which is generated according to the users' taste and preference.<br>
**Web scraper** fetches data (reviews, Twitter likes, votes, polls) from external sources ( like IMDB, Rotten Tomatoes) and stores them in [NoSQL databse](https://en.wikipedia.org/wiki/NoSQL) for computation.<br>
And well, yeah, that's pretty much about Neural Network implementation.
<br><br>

_Reference for the Neural Network architecture:_ _https://www.hindawi.com/journals/cin/2019/4589060/_
