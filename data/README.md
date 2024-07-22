# Data for training and testing
## representative_train.pkl
The dataframe for training with shape (6004, 11).
The following is an introduction to each column of data:
| level_0 | index | raw_label | sent_list | label | paragraph | sent_len | link_num | ref_num | para_level | sample_source |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| level_0 index | index of sample | (raw triplets, label and annotation method) | sentences contain corresponding time and place entities | the label of triplet, 1 for True and 0 for False | related paragraph | the number of tokens in these sentences | the number of links in this paragraph | the number of references in this paragraph | the level of this paragraph | the title of biography page in Wikipedia |

## representative_test.pkl
The dataframe for testing with shape (2574, 11).
The following is an introduction to each column of data:
| level_0 | index | raw_label | sent_list | label | paragraph | sent_len | link_num | ref_num | para_level | sample_source |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| level_0 index | index of sample | (raw triplets, label and annotation method) | sentences contain corresponding time and place entities | the label of triplet, 1 for True and 0 for False | related paragraph | the number of tokens in these sentences | the number of links in this paragraph | the number of references in this paragraph | the level of this paragraph | the title of biography page in Wikipedia |

## regular.pkl
Dataframe for testing, shape (274, 9), annotated by humans.
The following is an introduction to each column of data:
| raw_label | sent_list | label | paragraph | sent_len | link_num | ref_num | para_level | sample_source |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| (raw triplets, label and annotation method) | sentences contain corresponding time and place entities | the label of triplet, 1 for True and 0 for False | related paragraph | the number of tokens in these sentences | the number of links in this paragraph | the number of references in this paragraph | the level of this paragraph | the title of biography page in Wikipedia |
