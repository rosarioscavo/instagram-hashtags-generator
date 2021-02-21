# InstagramHashtagGenerator

This project's main goal remains a "proof of concept" based on what I have studied through
the material of the subject "Social Media Management".

"Instagram Hashtag Generator" analyzes an image and gives back some hashtags based on a prediction of binary classifiers. It is possible to use two different classifiers, a classifier based on Logistic Regression and the other one on k-nearest neighbors.

Instagram allows its users to use hashtags, a maximum of 30 hashtags per post precisely.
Hashtags are used to make a post easier to be discovered by other users. Some hashtags are
more popular than others because they are used independently of the context. However, some are
correlated with each other; for example, a picture of two friends could be associated with the tags "people" and "friends").

The idea is to analyze some main tags and seek out the most common hashtags used with them, removing the ones known to be used to gain more likes and comments (e.g., "picoftheday").

Follow an example of the output of the program on two images.

<img src="https://drive.google.com/uc?export=view&id=1JWdRa57YL3yp7CadLpH2PR40ao_AkUj2"/>

The tags that has been analyzed are:
1. nature
2. sky
3. summer
4. sea
5. friends
6. food
7. art
8. street
9. car
10. building
11. people
12. animal
13. cute
14. happy
15. fashion
16. sunset
17. architecture
18. landscape
19. smile
20. girl

For each of them is possible to download the model (knn or lr) already trained with more than 10000 images in total.

It is present a pdf that is the relation I presented as a project, in which I explained each step of the realization of the algorithm.

## Developed by

- [Rosario Scavo](https://github.com/rosarioscavo) 