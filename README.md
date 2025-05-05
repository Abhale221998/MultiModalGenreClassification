We initially started with a dataset with ~52400 movies. However, the genre distribution was skewed towards drama and comedy. To tackle this we tried a couple of options: 
1. Merge multiple datasets to reinforce underrepresented genres
2. Web Scraping
We finally used a hybrid model where we initially combined multiple (3) datasets and web scraping for the genres which were still not sufficiently represented.
Also, to make it more uniformly distributed, we capped the number of movies/genre to 8000. This makes the size of the final dataset (after the capping) ~74000
  
Attached a jupyter notebook (FINAL_DATASET) that contains the results for the genre distribution after doing the stated modifications to the initial dataset. 
It also shows the stages of genre distribution variation for reference. 

Poster Data Download:
The way we approached poster data download is by utilizing the TMDB and OMDB APIs. While downloading the poster data, a few short comings that we came across were: 
1. There were movies/movie titles in languages other than english as well. This caused error while downloading posters as these movies were not in the TMDB dataset.
2. Some old movies posters (pre 1995) were not on the TMDB or OMDB API.

We tackled this issue by training the image model on the movies which have the posters. 

Image model Selection and Training
To train an image model on the dataset we considered multiple options. We have attached a pdf file of the researched models. Finally, after researching models and understanding the pros and cons of each model carefully, we shortlisted 3 models: ResNet50, EfficientNetB4 and YOLO. 
