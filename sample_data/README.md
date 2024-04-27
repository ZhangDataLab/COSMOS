# Sample_data
## sample.json
Donald Friend's json format page contains both gpt-annotated triples and manually annotated triples.

`[person, time, location, True / False, manual]` for manually annotated triples.
`[person, time, location, True, GPT]` for gpt-annotated triples.

## GPT_annotation_sample.csv
100 GPT annotated data samples in the **Representative** dataset.

The following is an introduction to each column of data:
| source_biography_page | sent_list | triplet | GPT_annotation | manual_annotation |
|-----|-----|-----|-----|-----|
| title of biography page in Wikipedia | sentences contain corresponding time and place entities | raw triplet extracted via GPT | GPT annotation | manual annotation |


## test_set_sample.csv
This file comprises 5% of the samples from the test set of the **Representative** dataset, totaling 142 entries.

The following is an introduction to each column of data:
| source_biography_page | sent_list | raw_label | label | COSMOS | RoBERTa | BERT | CeleTrip | CNN | LSTM |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| title of biography page in Wikipedia | sentences contain corresponding time and place entities | (raw triples, label and annotation method) | label of triplet, 1 for True and 0 for False | output of COSMOS (ours) | output of RoBERTa (baseline) | output of BERT (baseline) | output of CeleTrip (baseline) | output of CNN (baseline) | output of LSTM (baseline) | 