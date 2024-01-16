# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# others
import numpy as np
import pandas as pd
import random
import os
import argparse, logging
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# for BERT
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

# Definition of Model and Dataset
from Model.dataset import *
from Model.COSMOS import *
from Model.config import *

config_map = {
    'COSMOS': 'Config_COSMOS'
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def parse_args():
    """
    Parses the arguments
    """
    
    parser = argparse.ArgumentParser(description="Run Model.")
    
    parser.add_argument('--model', nargs='?', default='COSMOS',
                        help='Specify a model')
    parser.add_argument('--mode', nargs='?', default='semi_supervised',
                        help='Specify training mode')
    parser.add_argument('--gpu_id', nargs='?', default=1,
                        help='Specify GPU id')
    parser.add_argument('--stage', nargs='?', default='train',
                    help='Specify running mode')
    
    return parser.parse_args()

def load_data_cnn(args, config):
    """
    Load Data for cnn
    """
    logging.info('========== Start Loading Data ==========')
    
    WikiDataSet = WikiDataSet_simple
    
    train_dataset = WikiDataSet(config.train_path)
    test_dataset = WikiDataSet(config.test_path)
    
    if config.setting_2_dataset_path:
        setting_2_dataset = WikiDataSet(config.setting_2_dataset_path)
    else:
        setting_2_dataset = None
    
    if config.inference_dataset_path:
        inference_dataset = WikiDataSet(config.inference_dataset_path)
    else:
        inference_dataset = None
    
    logging.info('========== Data Loaded ==========')
    logging.info("Train size: " + str(len(train_dataset)))
    logging.info("Test size: " + str(len(test_dataset)))
    
    unlabel_dataset = None
    if args.mode == 'semi_supervised':
        unlabel_dataset = WikiDataSet(config.unlabel_path)
        logging.info("Unlabeled Data size:" + str(len(unlabel_dataset)))
    
    return train_dataset, test_dataset, unlabel_dataset, setting_2_dataset, inference_dataset

def load_data_bert(args, config):
    """
    Load Data for bert
    """
    
    logging.info('========== Start Loading Data ==========')
    
    train_dataset = WikiDataSet_bert(config.train_path, config)
    test_dataset = WikiDataSet_bert(config.test_path, config)

    if config.setting_2_dataset_path:
        setting_2_dataset = WikiDataSet_bert(config.setting_2_dataset_path, config)
    else:
        setting_2_dataset = None
        
    if config.inference_dataset_path:
        inference_dataset = WikiDataSet_bert(config.inference_dataset_path, config)
    else:
        inference_dataset = None
    
    logging.info('========== Data Loaded ==========')
    logging.info("Train size: " + str(len(train_dataset)))
    logging.info("Test size: " + str(len(test_dataset)))
    
    unlabel_dataset = None
    if args.mode == 'semi_supervised':
        unlabel_dataset = WikiDataSet_bert(config.unlabel_path, config)
        logging.info("Unlabeled Data size:" + str(len(unlabel_dataset)))
    
    return train_dataset, test_dataset, unlabel_dataset, setting_2_dataset, inference_dataset

def train(model, device, train_loader, val_loader, optimizer, criterion, scheduler, config):
    
    cnn_train_loader, bert_train_loader = train_loader
    
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_acc = 0.0
    early_stop = 0
    
    for epoch in range(config.epoch):
        model.train()
        train_loss = 0
        
        print("Epoch : #", epoch)
        model.current_epoch = epoch
        
        for batch, batch_data in enumerate(zip(cnn_train_loader, bert_train_loader)):
            
            # cnn input
            sent_feat = batch_data[0][0].to(device)
            tuple_feat = batch_data[0][1].to(device)
            labels = batch_data[0][2].to(device)
            cnn_input = (sent_feat, tuple_feat)
            
            # bert input
            input_ids = batch_data[1]["input_ids"].to(device)
            attention_mask = batch_data[1]["attention_mask"].to(device)
            labels = batch_data[1]["label"].to(device)
            bert_input = (input_ids, attention_mask)
            
            # forward
            outputs, hidden_feat = model(cnn_input=cnn_input, bert_input=bert_input, device=device)
            # loss
            ce_loss = criterion(outputs, labels)
            scl_loss = contrastive_loss(0.1, hidden_feat.cpu().detach().numpy(), labels)
            loss = (config.scl_coef * scl_loss) + (1 - config.scl_coef) * ce_loss
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() / len(labels)
        
        scheduler.step()
        
        _, train_acc, _, _, _ = evaluate(model, device, train_loader, criterion)
        _, val_acc, _, _, _ = evaluate(model, device, val_loader, criterion)
        
        train_loss_list.append(train_loss / (batch + 1))
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        
        # early stop
        if val_acc >= best_acc:
            best_acc = val_acc
            logging.info("Save the best model!")
            torch.save(model.state_dict(), config.best_model_path)
            early_stop = 0
        else:
            early_stop += 1
        torch.cuda.empty_cache()
        
        if early_stop > config.patience:
            logging.info("Stop epoch : " + str(epoch))
            break
        
    return train_loss_list, train_acc_list, val_acc_list

def contrastive_loss(temp, embedding, label):
    """calculate the contrastive loss
    @refer https://github.com/sl-93/SUPERVISED-CONTRASTIVE-LEARNING-FOR-PRE-TRAINED-LANGUAGE-MODEL-FINE-TUNING/blob/main/main.py
    """
    cosine_sim = cosine_similarity(embedding, embedding)
    dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    dis = np.exp(dis)
    cosine_sim = np.exp(cosine_sim)

    row_sum = []
    for i in range(len(embedding)):
        row_sum.append(sum(dis[i]))
    contrastive_loss = 0
    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        inner_sum = 0
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + np.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss

def compute_alpha(batch_num, max_batches):
    """
    for semi-trained
    """
    if batch_num < int(0.10 * max_batches):
        return 0.0
    elif batch_num > int(0.90* max_batches):
        return 1.0
    else:
        return (batch_num * 1.0) / max_batches

def semi_train(model, device, unlabel_loader, train_loader, val_loader, optimizer, criterion, scheduler, config):

    cnn_train_loader, bert_train_loader = train_loader
    cnn_unlabel_loader, bert_unlabel_loader = unlabel_loader
    
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_acc = 0.0
    early_stop = 0
    
    logging.info("epoch coef:")
    logging.info(config.epoch_coef)
    
    for epoch in range(config.epoch):
        model.train()
        train_loss = 0
        
        print("Epoch : #", epoch)
        
        max_batches = len(cnn_train_loader)
        cnn_unlabel_data_iter = iter(cnn_unlabel_loader)
        bert_unlabel_data_iter = iter(bert_unlabel_loader)
        
        model.current_epoch = epoch
        
        for batch, batch_data in enumerate(zip(cnn_train_loader, bert_train_loader)):
            
            alpha = compute_alpha(batch, max_batches) * (config.epoch_coef / (epoch+1))
            
            # cnn input
            sent_feat = batch_data[0][0].to(device)
            tuple_feat = batch_data[0][1].to(device)
            labels = batch_data[0][2].to(device)
            cnn_input = (sent_feat, tuple_feat)
            
            # bert input
            input_ids = batch_data[1]["input_ids"].to(device)
            attention_mask = batch_data[1]["attention_mask"].to(device)
            labels = batch_data[1]["label"].to(device)
            bert_input = (input_ids, attention_mask)
            
            # forward
            # labeled loss
            outputs, hidden_feat = model(cnn_input=cnn_input, bert_input=bert_input, device=device)
            labeled_ce_loss = criterion(outputs, labels)
            labeled_scl_loss = contrastive_loss(config.temperature, hidden_feat.cpu().detach().numpy(), labels)
            labeled_loss = (config.scl_coef * labeled_scl_loss) + (1 - config.scl_coef) * labeled_ce_loss
            
            # unlabeled loss
            cnn_unlabel_batch_data, bert_unlabel_batch_data = None, None
            try:
                cnn_unlabel_batch_data = next(cnn_unlabel_data_iter)
                bert_unlabel_batch_data = next(bert_unlabel_data_iter)
            except:
                cnn_unlabel_data_iter, bert_unlabel_data_iter \
                    = iter(cnn_unlabel_loader), iter(bert_unlabel_loader)
                
                cnn_unlabel_batch_data, bert_unlabel_batch_data \
                    = next(cnn_unlabel_data_iter), next(bert_unlabel_data_iter)
            
            # cnn input
            sent_feat = cnn_unlabel_batch_data[0].to(device)
            tuple_feat = cnn_unlabel_batch_data[1].to(device)
            unlabel_labels = cnn_unlabel_batch_data[2].to(device)
            cnn_input = (sent_feat, tuple_feat)
            
            # bert input
            input_ids = bert_unlabel_batch_data["input_ids"].to(device)
            attention_mask = bert_unlabel_batch_data["attention_mask"].to(device)
            unlabel_labels = bert_unlabel_batch_data["label"].to(device)
            bert_input = (input_ids, attention_mask)
            
            unlabel_outputs, _ = model(cnn_input=cnn_input, bert_input=bert_input, device=device)
            pseudo_label = torch.argmax(unlabel_outputs, dim=1)
            unlabeled_loss = criterion(unlabel_outputs, pseudo_label)
            
            loss = labeled_loss + (alpha * unlabeled_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() / len(labels)
        
        scheduler.step()
        
        _, train_acc, _, _, _ = evaluate(model, device, train_loader, criterion)
        _, val_acc, _, _, _ = evaluate(model, device, val_loader, criterion)
        
        train_loss_list.append(train_loss / (batch + 1))
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.best_model_path)
            early_stop = 0
        else:
            early_stop += 1
        torch.cuda.empty_cache()
        
        if early_stop > config.patience:
            logging.info("Stop epoch : " + str(epoch))
            break
        
    return train_loss_list, train_acc_list, val_acc_list

@torch.no_grad()
def evaluate(model, device, dataloader, criterion):
    
    cnn_val_loader, bert_val_loader = dataloader
    
    model.eval()
    val_loss = 0
    total_pred = []
    total_labels = []
    total_source = []
    
    for batch, batch_data in enumerate(zip(cnn_val_loader, bert_val_loader)):
        
        # cnn input
        sent_feat = batch_data[0][0].to(device)
        tuple_feat = batch_data[0][1].to(device)
        labels = batch_data[0][2].to(device)
        cnn_input = (sent_feat, tuple_feat)
        
        # bert input
        input_ids = batch_data[1]["input_ids"].to(device)
        attention_mask = batch_data[1]["attention_mask"].to(device)
        labels = batch_data[1]["label"].to(device)
        bert_input = (input_ids, attention_mask)
            
        # forward
        outputs, _ = model(cnn_input=cnn_input, bert_input=bert_input, device=device)
        # loss
        loss = criterion(outputs, labels)
        # predict
        _, predicted = torch.max(outputs.data, 1)
        
        # concat the result
        total_pred += list(predicted.cpu().numpy().reshape(-1))
        total_labels += list(labels.cpu().numpy().reshape(-1))
        val_loss += loss.item()
        
        # source
        total_source += batch_data[1]['source']
    
    score = accuracy_score(total_labels, total_pred)
    
    return val_loss / (batch + 1), score, total_labels, total_pred, total_source

    
if __name__ == '__main__':
    
    ################################# init ###############################
    os.chdir('/xxx/')
    seed_everything(42)
    args = parse_args()
    
    # init config
    config = eval(config_map[args.model])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(int(args.gpu_id))
    
    # init logging
    logging.basicConfig(filename=config.log_file, filemode='a', level=logging.DEBUG, \
    format='[%(asctime)s][%(levelname)s] ## %(message)s')
    
    # init tokenizer
    VOCAB = 'vocab.txt'
    config.BERT_config.tokenizer = BertTokenizer.from_pretrained(os.path.join(config.BERT_config.bert_model_name, VOCAB))
    
    logging.info('+++++++++++++++++++++++++++++++ Begin to run +++++++++++++++++++++++++++++++')
    # load data
    cnn_train_dataset, cnn_test_dataset, cnn_unlabel_dataset, \
        cnn_setting_2_dataset, cnn_inference_dataset = load_data_cnn(args, config.CNN_config)
    bert_train_dataset, bert_test_dataset, bert_unlabel_dataset, \
        bert_setting_2_dataset, bert_inference_dataset = load_data_bert(args, config.BERT_config)
    # random split train/val
    logging.info('========== Split val set from train set ==========')
    train_idx, val_idx = train_test_split([i for i in range(len(cnn_train_dataset))], test_size = 0.2, random_state = 42)
    logging.info('The length of final train set: ')
    logging.info(len(train_idx))
    logging.info('The length of val set: ')
    logging.info(len(val_idx))
    ################################# end of init ###############################

    # get data_loader
    batch_size = config.batch_size
    
    cnn_unlabel_loader = DataLoader(
        cnn_unlabel_dataset,
        batch_size = batch_size,
        drop_last = False
    )
    
    cnn_train_loader = DataLoader(
        cnn_train_dataset,
        sampler = train_idx,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)

    cnn_val_loader = DataLoader(
        cnn_train_dataset,
        sampler = val_idx,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)

    cnn_test_loader = DataLoader(
        cnn_test_dataset,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)
    
    cnn_setting_2_dataset_loader = DataLoader(
        cnn_setting_2_dataset,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)
        
    cnn_inference_dataset_loader = DataLoader(
        cnn_inference_dataset,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)
        
    bert_unlabel_loader = DataLoader(
        bert_unlabel_dataset,
        batch_size = batch_size,
        drop_last = False
    )
    
    bert_train_loader = DataLoader(
        bert_train_dataset,
        sampler = train_idx,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)

    bert_val_loader = DataLoader(
        bert_train_dataset,
        sampler = val_idx,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)

    bert_test_loader = DataLoader(
        bert_test_dataset,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)
    
    bert_setting_2_dataset_loader = DataLoader(
        bert_setting_2_dataset,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)
        
    bert_inference_dataset_loader = DataLoader(
        bert_inference_dataset,
        batch_size = batch_size,
        drop_last = False)
        # shuffle = True)
    
    ################################# training ###############################
    # init model
    logging.info("===== Model: {} =====".format(args.model))
    model = eval(args.model + '(config)')
    
    if torch.cuda.is_available() == True:
        model = model.to(device)
        
    CRLoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.lr_scheduler_gamma)
    
    if args.stage == 'train':
    
        logging.info("======= Training Mode: Semi-Supervised =======")
        
        unlabel_loader = (cnn_unlabel_loader, bert_unlabel_loader)
        train_loader = (cnn_train_loader, bert_train_loader)
        val_loader = (cnn_val_loader, bert_val_loader)
        
        train_loss_list, train_acc_list, val_acc_list = semi_train(model, device, unlabel_loader, train_loader, \
            val_loader, optimizer, CRLoss, scheduler, config)
    
    ################################# end of training ###############################
    
    ################################# evaluate ###############################
    logging.info("======= Start to evaluate  =======")
    # load best model
    model.load_state_dict(torch.load(config.best_model_path))
    # test on test set
    
    test_loader = (cnn_test_loader, bert_test_loader)
    
    test_loss, test_acc_s1, test_labels_s1, test_predict_s1, test_label_source = evaluate(model, device, test_loader, CRLoss)
    
    setting_1_result_df = pd.DataFrame({
        "label": test_labels_s1,
        "predict": test_predict_s1,
        "source": test_label_source
    })
    
    setting_1_rep_all = classification_report(test_labels_s1, test_predict_s1, digits=6, target_names=['non trip location','trip location'])
    logging.info('#'*20)
    logging.info('[Setting_1_all] after epoch training : ')
    logging.info('\n' + setting_1_rep_all)
    logging.info('#'*20)
    # For GPT labeled data
    setting_1_gpt = setting_1_result_df[setting_1_result_df["source"] == "GPT"]
    setting_1_rep_gpt = classification_report(setting_1_gpt["label"], setting_1_gpt["predict"], digits=6, target_names=['non trip location','trip location'])
    logging.info('[Setting_1_gpt] after epoch training : ')
    logging.info('\n' + setting_1_rep_gpt)
    # For manual labeled data
    setting_1_manual = setting_1_result_df[setting_1_result_df["source"] == "manual"]
    setting_1_rep_manual = classification_report(setting_1_manual["label"], setting_1_manual["predict"], digits=6, target_names=['non trip location','trip location'])
    logging.info('[Setting_1_manual] after epoch training : ')
    logging.info('\n' + setting_1_rep_manual)
    
    setting_2_dataset_loader = (cnn_setting_2_dataset_loader, bert_setting_2_dataset_loader)
    test_loss, test_acc, test_labels, test_predict, test_label_source = evaluate(model, device, setting_2_dataset_loader, CRLoss)
    
    setting_2_result_df = pd.DataFrame({
        "label": test_labels,
        "predict": test_predict,
        "source": test_label_source
    })
    
    setting_2_rep_all = classification_report(test_labels, test_predict, digits=6, target_names=['non trip location','trip location'])
    logging.info('[Setting_2_all] after epoch training : ')
    logging.info('\n' + setting_2_rep_all)
    grouped_setting_2 = setting_2_result_df.groupby("source")
    acc_list = []
    for name, df in grouped_setting_2:
        tmp_acc = accuracy_score(df["label"], df["predict"])
        acc_list.append(tmp_acc)
    s2_mean = np.mean(acc_list)
    s2_std = np.std(acc_list)
    logging.info('[Setting_2_stat] : %f ± %f ', s2_mean, s2_std)
    
    ################################# end of evaluate ###############################
        
    torch.cuda.empty_cache()
    logging.info('+++++++++++++++++++++++++++++++ Finish +++++++++++++++++++++++++++++++')