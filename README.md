# Paths of A Million People: Extracting Life Trajectories from Wikipedia
This is the source code for paper 'Paths of A Million People: Extracting Life Trajectories from Wikipedia'.   
Here we provide:
- Source code for our model **COSMOS** in ```/COSMOS```
- Source code for our tripets extraction tool in ```/preprocessing/tuple_extraction.py```
- Annotated data for taining and testing in ```/data```
- Life Trajectories Extracted from 5% biography pages on Wikipedia ```/trajectory_data```
- A Sample from our ground truth dataset *WikiLifeTrajectory* : ```/sample_data/sample.json```

## Abstract
The life trajectories of notable people have been studied to pinpoint the times and places of significant events such as birth, death, education, marriage, competition, work, speeches, scientific discoveries, artistic achievements, and battles. Understanding how these individuals interact with others provides valuable insights for broader research into human dynamics. However, the scarcity of trajectory data in terms of volume, density, and inter-person interactions, limits relevant studies from being comprehensive and interactive. We mine millions of biography pages from Wikipedia and tackle the generalization problem stemming from the variety and heterogeneity of the trajectory descriptions. Our ensemble model COSMOS, which combines the idea of semi-supervised learning and contrastive learning, achieves an F1 score of 85.95%. For this task, we also create a hand-curated dataset, WikiLifeTrajectory, consisting of 8,852 (person, time, location) triplets as ground truth. Besides, we perform an empirical analysis on the trajectories of 8,272 historians to demonstrate the validity of the extracted results. To facilitate the research on trajectory extractions and help the analytical studies to construct grand narratives, we make our code, the million-level extracted trajectories, and the WikiLifeTrajectory dataset publicly available.

## COSMOS Model

### Requirements
See ```requirements.txt```

## Others

We train our model on two Tesla M40. All our data shared from this work will be made FAIR[1].

[1] FORCE11. 2020. The FAIR Data principles. https://force11.org/info/the-fair-data-principles/.

## Cite
Please cite our paper if you find this code useful for your research.

```
@article{zhang2024paths,
  title={Paths of A Million People: Extracting Life Trajectories from Wikipedia},
  author={Zhang, Ying and Li, Xiaofeng and Liu, Zhaoyang and Zhang, Haipeng},
  journal={arXiv preprint arXiv:2406.00032},
  year={2024}
}
```