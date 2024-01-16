# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

# model
class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, config.num_classes)
        
    def forward(self, input_ids, attention_mask, device='cuda'):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits, x

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(config.in_channels, config.out_channels, (config.kernel_heights[0], config.embedding_length), config.stride, config.padding)
        self.conv2 = nn.Conv2d(config.in_channels, config.out_channels, (config.kernel_heights[1], config.embedding_length), config.stride, config.padding)
        self.conv3 = nn.Conv2d(config.in_channels, config.out_channels, (config.kernel_heights[2], config.embedding_length), config.stride, config.padding)
        self.tuple_conv3 = nn.Conv2d(config.in_channels, config.out_channels, (config.kernel_heights[3], config.embedding_length), config.stride, config.padding)
        self.dropout = nn.Dropout(config.keep_probab)
        self.fusion_lin = nn.Linear(len(config.kernel_heights)*config.out_channels, config.lin_size)
        self.label = nn.Linear(config.lin_size, config.output_size)
    
    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        
        return max_out
    
    def forward(self, input_sentences, tuple_feat, device='cuda'):
        input_sentences = input_sentences.to(device)
        input_sentences = input_sentences.float().unsqueeze(1)
        
        tuple_feat = tuple_feat.to(device)
        tuple_feat = tuple_feat.float().unsqueeze(1)

        max_out1 = self.conv_block(input_sentences, self.conv1)
        max_out2 = self.conv_block(input_sentences, self.conv2)
        max_out3 = self.conv_block(input_sentences, self.conv3)
        max_out_tuple = self.conv_block(tuple_feat, self.tuple_conv3)
        
        all_out = torch.cat((max_out1, max_out2, max_out3, max_out_tuple), 1)
        fc_in = self.dropout(all_out)
        fusion_feat = F.relu(self.fusion_lin(fc_in))
        
        logits = self.label(fusion_feat)
        return logits, fusion_feat
    
class COSMOS(nn.Module):
    def __init__(self, config):
        super(COSMOS, self).__init__()
        self.cnn = CNN(config=config.CNN_config)
        self.bert = BERT(config=config.BERT_config)
        
        self.linear_cnn = nn.Linear(128, config.fused_dim, bias=True)
        self.linear_bert = nn.Linear(768, config.fused_dim, bias=True)
        
        self.att_liner = nn.Linear(2*config.fused_dim, config.hidden_feat_dim)
        
        self.classifier = nn.Linear(4, 2, bias=True)
        
        self.current_epoch = 0
        
        self.bert.register_full_backward_hook(self.bert_backward_hook)
    
    def forward(self, cnn_input, bert_input, device='cuda'):
        sent_feat, tuple_feat = cnn_input
        cnn_output, cnn_feat = self.cnn(sent_feat, tuple_feat)
        cnn_feat = self.linear_cnn(cnn_feat)
        
        input_ids, attention_mask = bert_input
        bert_output, bert_feat = self.bert(input_ids, attention_mask)
        bert_feat = self.linear_bert(bert_feat)
        
        fused_output = torch.cat([cnn_output, bert_output], dim=-1)
        logits = self.classifier(fused_output)
        
        hidden_feat = torch.cat([cnn_feat, bert_feat], dim=-1)
        hidden_feat = torch.tanh(self.att_liner(hidden_feat))
        
        return logits, hidden_feat
        
    def bert_backward_hook(self, module, grad_input, grad_output):
        if self.current_epoch > 3:
            return (None,)
        else:
            return grad_input