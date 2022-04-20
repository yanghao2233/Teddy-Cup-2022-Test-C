
"""
T1_yy_context_type_path 类
    储存第一题中所需的模型、暂存向量、原始数据、目标结果的相对路径类
    ->  _all_data_: 原始数据
        _title/text_model_: 训练模型路径
        _title/text_keyvecs_: 关键词向量暂存文件路径
        _result_: 结果文件路径
"""
T1_19_all_data_path = './data/2018-2019茂名（含自媒体）.xlsx'
T1_20_all_data_path = './data/2020-2021茂名（含自媒体）.xlsx'

T1_19_title_model_path = './model/Test1_2019_titles_w2v_128d.model'
T1_20_title_model_path = './model/Test1_2020_titles_w2v_128d.model'
T1_19_text_model_path = './model/Test1_2019_texts_w2v_128d.model'
T1_20_text_model_path = './model/Test1_2020_texts_w2v_128d.model'

T1_19_title_keyvecs_path = './model/Test1_2019_title_keyvec.csv'
T1_20_title_keyvecs_path = './model/Test1_2020_title_kevec.csv'
T1_19_text_keyvecs_path = './model/Test1_2019_text_keyvec.csv'
T1_20_text_keyvecs_path = './model/Test1_2020_text_keyvec.csv'

T1_stopwords_path = './data/hit_stopwords.txt'
T1_keywords_path = './data/keywords.txt'