"""
@Title: Extracting triplets from JSON format data

@refer
[1] Liu, Y., Hua, W., & Zhou, X. (2021). Temporal knowledge extraction from large-scale text corpus. World Wide Web, 24(1), 135–156. https://doi.org/10.1007/s11280-020-00836-5
"""

# others
import json
from tqdm import *
import argparse
import warnings
warnings.filterwarnings('ignore')

# nlp
import nltk
from nltk import Tree, pos_tag

import spacy
from timexy import Timexy

import benepar
benepar.download('benepar_en3')
# benepar.download('benepar_en3_large')

# nlp pipeline
nlp = spacy.load("en_core_web_trf")
config = {
    "kb_id_type": "timex3",
    "label": "timexy",
    "overwrite": False,
}
nlp.add_pipe("timexy", config=config, before="ner")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

##################################################################################################
loc_ent_list = ['EVENT', 'FAC', 'GPE', 'LOC', 'ORG']
time_ent_list = ['DATE', 'TIME', 'timexy']

def is_target_sent(doc):
    loc_exist, time_exist = False, False
    for ent in doc.ents:
        if ent.label_ in loc_ent_list:
            loc_exist = True
            continue    
        elif ent.label_ in time_ent_list:
            time_exist = True
            continue    
    return loc_exist & time_exist

def extract_verb(doc): 
    verbs = []
    for token in doc:
        if token.pos_ == "VERB" or token.pos_ == 'AUX':
            verbs.append(token.text)         
    return verbs

def generate_timex_tag_list(doc): 
    timex_tag_dict = {'timexy': [], 'DATE': []}
    for ent in doc.ents:
        if ent.label_ == "timexy":
            temp_tag_list = []
            for token in ent:
                temp_tag_list.append(token.text)
                
            timex_tag_dict['timexy'].append([temp_tag_list, "".join(ent.text)])
        if ent.label_ == "DATE":
            timex_tag_dict['DATE'].append((ent.text.split(' '), ent.text)) 
    return timex_tag_dict

def extract_person(doc): 
    persons = {'unitary': [], 'plural': []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ent_list = ent.text.split(' ')
            if len(ent_list) == 1:
                persons['unitary'].append(ent.text)
            else:
                persons['plural'].append((ent_list, ent.text))
    return persons

def extract_subjects(doc): 
    subjects = []
    
    for token in doc:
        if "subj" in token.dep_ and token.text.lower() in ['he', 'she', 'they']:
            subjects.append(token.text) 
    return subjects

def extract_loc(doc): 
    locations = {'unitary': [], 'plural': []}
    for ent in doc.ents:
        if ent.label_ in loc_ent_list:
            ent_list = ent.text.split(' ')
            if len(ent_list) == 1:
                locations['unitary'].append(ent.text)
            else:
                locations['plural'].append((ent_list, ent.text))
    return locations

##################################################################################################
##################################################################################################
def get_nltk_tree(doc): # or (doc, sent_idx)
    constituents = doc._.parse_string
    nltk_tree = Tree.fromstring(constituents)
    return nltk_tree

def calc_dist(nltk_tree, node1_str, node2_str):
    node1_pos = None
    node2_pos = None

    for pos in nltk_tree.treepositions():
        if node1_pos != None and node2_pos != None:
            lca_pos = []
            for i, (p1, p2) in enumerate(zip(node1_pos, node2_pos)):
                if p1 == p2:
                    lca_pos.append(p1)
                else:
                    break

            path1 = len(node1_pos) - i
            path2 = len(node2_pos) - i
            return lca_pos, min(path1, path2)

        node = nltk_tree[pos]
        if isinstance(node, str):
            if node == node1_str:
                node1_pos = pos
            elif node == node2_str:
                node2_pos = pos
        else:
            continue
            
    return None, 9999999999

def find_pv_match(temp_tree, person_list, verb_list):
    dist_dict = {}
    lca_dict = {}
    person_match_verb = {}

    for person in person_list['unitary']:
        person_match_verb[person] = (None, 9999999999)

        for verb in verb_list:
            lca, min_path = calc_dist(temp_tree, person, verb)
            lca_dict[(person, verb)] = lca
            dist_dict[(person, verb)] = min_path
            if min_path < person_match_verb[person][-1]:
                person_match_verb[person] = (verb, min_path)

    for person in person_list['plural']:
        person_match_verb[person[-1]] = (None, 9999999999)

        for verb in verb_list:
            lca, min_path = calc_dist(temp_tree, person[0][0], verb)
            lca_dict[(person[-1], verb)] = lca
            dist_dict[(person[-1], verb)] = min_path
            if min_path < person_match_verb[person[-1]][-1]:
                person_match_verb[person[-1]] = (verb, min_path)

    for person in person_list['subject']:
        person_match_verb[person] = (None, 9999999999)

        for verb in verb_list:
            lca, min_path = calc_dist(temp_tree, person, verb)
            lca_dict[(person, verb)] = lca
            dist_dict[(person, verb)] = min_path
            if min_path < person_match_verb[person][-1]:
                person_match_verb[person] = (verb, min_path)

    return person_match_verb, dist_dict, lca_dict

def find_vt_match(temp_tree, verb_list, timex_tag):
    dist_dict = {}
    lca_dict = {}
    verb_match_time = {}

    for verb in verb_list:
        verb_match_time[verb] = (None, 9999999999)

        for t in timex_tag['timexy']:
            lca, min_path = calc_dist(temp_tree, verb, t[0][0])
            lca_dict[(verb, t[-1])] = lca
            dist_dict[(verb, t[-1])] = min_path
            if min_path < verb_match_time[verb][-1]:
                verb_match_time[verb] = (t[-1], min_path)

        for t in timex_tag['DATE']:
            lca, min_path = calc_dist(temp_tree, verb, t[0][0])
            lca_dict[(verb, t[-1])] = lca
            dist_dict[(verb, t[-1])] = min_path
            if min_path < verb_match_time[verb][-1]:
                verb_match_time[verb] = (t[-1], min_path)

    return verb_match_time, dist_dict, lca_dict

def find_vl_match(temp_tree, verb_list, loc_list):
    dist_dict = {}
    lca_dict = {}
    verb_match_loc = {}

    for verb in verb_list:
        verb_match_loc[verb] = (None, 9999999999)

        for loc in loc_list['unitary']:
            lca, min_path = calc_dist(temp_tree, verb, loc)
            lca_dict[(verb, loc)] = lca
            dist_dict[(verb, loc)] = min_path
            if min_path < verb_match_loc[verb][-1]:
                verb_match_loc[verb] = (loc, min_path)

        for loc in loc_list['plural']:
            dist_dict[(verb, loc[-1])] = 9999999999

            for term in loc[0]:
                lca, min_path = calc_dist(temp_tree, verb, term)

                if min_path < dist_dict[(verb, loc[-1])]: 
                    lca_dict[(verb, loc[-1])] = lca
                    dist_dict[(verb, loc[-1])] = min_path

            if dist_dict[(verb, loc[-1])] < verb_match_loc[verb][-1]:
                verb_match_loc[verb] = (loc[-1], dist_dict[(verb, loc[-1])])

    return verb_match_loc, dist_dict, lca_dict

def merge_by_verb(person_match_verb, verb_match_time, verb_match_loc):
    res_tuples = []
    for person, verb_tuple in person_match_verb.items():
        if verb_tuple[0] == None:
            res_tuples.append((None, None, None))
            continue
        res_tuples.append((person, verb_tuple[0], verb_match_loc[verb_tuple[0]][0], verb_match_time[verb_tuple[0]][0]))
    
    return res_tuples

##################################################################################################
##################################################################################################
def sent_extraction(sent):
    verb_list = extract_verb(sent)
    person_list = {**extract_person(sent), **{'subject': extract_subjects(sent)}}
    loc_list = extract_loc(sent)
    time_list = generate_timex_tag_list(sent)

    nltk_tree = get_nltk_tree(sent)

    person_match_verb, _, _ = find_pv_match(nltk_tree, person_list, verb_list)
    verb_match_time, _, _ = find_vt_match(nltk_tree, verb_list, time_list)
    verb_match_loc, _, _ = find_vl_match(nltk_tree, verb_list, loc_list)
    extracted_tuples = merge_by_verb(person_match_verb, verb_match_time, verb_match_loc)
    
    res_tuples = []
    for res_tuple in extracted_tuples:
        if None not in res_tuple:
            res_tuples.append((res_tuple[0], res_tuple[3], res_tuple[2], None))
            
    return res_tuples

def paragraph_extraction(paragraph):
    target_sent_list = []
    tuple_list = []
    sent_list = []
    
    doc = nlp(paragraph)

    for sent in doc.sents:
        sent_list.append(sent.text)
        if not is_target_sent(sent):   
            continue
        res_tuples = sent_extraction(sent)
        for res_tuple in res_tuples:
            target_sent_list.append(sent.text) 
            tuple_list.append(res_tuple)

    return target_sent_list, tuple_list, sent_list

def person_extraction(person_dataset):
    # process summary
    paragraph = person_dataset['summary']['content']
    person_dataset['summary']['target_sents'], person_dataset['summary']['baseline_labels'],\
        person_dataset['summary']['content_sents'] = paragraph_extraction(paragraph)

    # process sections
    for sec_idx, section in enumerate(person_dataset['sections']):
        paragraph = section['content']
        person_dataset['sections'][sec_idx]['target_sents'], person_dataset['sections'][sec_idx]['baseline_labels'], \
            person_dataset['sections'][sec_idx]['content_sents'] = paragraph_extraction(paragraph)

    return person_dataset

def all_extraction(dataset):
    res_dataset = {}
    
    for person, person_dataset in tqdm(dataset.items()):
        
        try:
            res_dataset[person] = person_extraction(person_dataset.copy())
        except:
            pass
        
    return res_dataset

def parse_args():
    """
    Parses the arguments
    """
    
    parser = argparse.ArgumentParser(description="Run Extraction.")
    
    parser.add_argument('--input', nargs='?', default=None,
                        help='The path of input json file')
    
    parser.add_argument('--output', nargs='?', default=None,
                        help='The path of output json file')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    with open(args.input, encoding='utf-8') as f:
        json_dataset = json.load(f)
        
    json_result = all_extraction(json_dataset)
    
    print("%d pages have been processed." % (len(json_result)))

    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(json_result, file, ensure_ascii=False)