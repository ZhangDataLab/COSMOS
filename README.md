# Paths of A Million People: Extracting Life Trajectories from Wikipedia
This is the source code for paper 'Paths of A Million People: Extracting Life Trajectories from Wikipedia'.   
Here we provide
- Source code for our model **COSMOS** in ```/COSMOS```
- Source code for our tripets extraction tool in ```/preprocessing/tuple_extraction.py```
- Life Trajectories Extracted from 5% biography pages on Wikipedia ```/trajectory_data```
- A Sample from our ground truth dataset *WikiLifeTrajectory* : ```/sample_data/sample.json```

## Abstract
Notable people's life trajectories have been a focus of study -- the locations and times of various activities, such as birth, death, education, marriage, competition, work, delivering a speech, making a scientific discovery, finishing a masterpiece, and fighting a battle, and how these people interact with others, carry important messages for the broad research related to human dynamics. However, the scarcity of trajectory data in terms of volume, density, and inter-person interactions, limits relevant studies from being comprehensive and interactive. We mine millions of biography pages from Wikipedia and tackle the generalization problem stemming from the variety and heterogeneity of the trajectory descriptions. Our ensemble model COSMOS, which combines the idea of semi-supervised learning and contrastive learning, achieves an F1 score of 85.95%. For this task, we also create a hand-curated dataset, WikiLifeTrajectory, consisting of 8,852 (person, time, location) triplets as ground truth. Besides, we perform an empirical analysis on the trajectories of 8,272 historians to demonstrate the validity of the extracted results. To facilitate the research on trajectory extractions and help the analytical studies to construct grand narratives, we make our code, the million-level extracted trajectories, and the WikiLifeTrajectory dataset publicly available.

## COSMOS Model

### Requirements
See ```requirements.txt```