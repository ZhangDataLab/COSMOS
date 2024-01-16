# --------------------------------------------------------------- #
class Config_CNN:
    
    # dataset
    unlabel_path = None
    train_path = None
    test_path = None
    setting_2_dataset_path = None
    inference_dataset_path = None
    
    # model paramaters
    embedding_length = 768
    in_channels = 1
    out_channels = 16
    stride = 1
    padding = 0
    keep_probab = 0.9
    kernel_heights = [3, 4, 5, 3]
    
    sent_input_size = 768
    tuple_input_size = 768
    lin_size = 128
    output_size = 2
    
# --------------------------------------------------------------- #
class Config_bert:
    
    # dataset
    unlabel_path = None
    train_path = None
    test_path = None
    setting_2_dataset_path = None
    inference_dataset_path = None
    
    # model paramaters
    bert_model_name = "/.../bert-base-uncased"
    num_classes = 2
    max_length = 128
    tokenizer = None
    
#------------------------------------------------------------------------#
class Config_COSMOS:
    CNN_config = Config_CNN
    BERT_config = Config_bert
    
    # hyperparamaters #
    # during training
    epoch = 100
    device = 'cuda'
    step_size = 10
    lr_scheduler_gamma = 0.5
    patience = 15
    weight_decay = 5e-4
    learning_rate = 5e-5
    batch_size = 32
    epoch_coef = 0.85
    scl_coef = 0.1
    temperature = 0.1
    
    # for model
    fused_dim = 64
    hidden_feat_dim = 32
    
    # save
    best_model_path = None
    save_result_path = None
    save_prob_path = None
    final_best_model_path = None
    log_file = None