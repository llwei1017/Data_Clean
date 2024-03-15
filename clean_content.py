import re
import os
import pandas as pd
import numpy as np
from get_data import data_get

# def remove_emojis(text):
#     # 定义表情符号的正则表达式模式
#     emoji_pattern = re.compile("["
#                                "\U0001F600-\U0001F64F"  # 表情符号 (emoticons)
#                                "\U0001F300-\U0001F5FF"  # 图形符号 (symbols & pictographs)
#                                "\U0001F680-\U0001F6FF"  # 交通与地图符号 (transport & map symbols)
#                                "\U0001F1E0-\U0001F1FF"  # 国旗 (flags)
#                                "]+", flags=re.UNICODE)
    
#     text_without_emojis = re.sub(emoji_pattern, "", text)
    
#     return text_without_emojis

# def read_keywords_from_file(file_path):
#     keywords = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             keyword = line.strip()
#             keywords.append(keyword)
#     return keywords

# def clean_quote_client(sentence):
#     keyword = '【引用: 】'
#     if keyword in sentence:
#         quote_index = sentence.find('【引用: 】')
#         result = sentence[quote_index + 7:]
#         sentence = result
#     return sentence

# def clean_quote_service(data, index, sentence):
#     keyword = '这是一条引用/回复消息，'
#     if keyword in sentence:
#         df = data[max(0, index - 20):index + 21]
#         content_list = []
#         for content in df['content']:
#             sub_contents = content.split("------")
#             content_list.extend(sub_contents)
#         matching_sentence = None
#         for _sentence in content_list:
#             if sentence in _sentence:
#                 matching_sentence = _sentence
#                 break
#         if matching_sentence:
#             sentence = sentence.replace(matching_sentence, "")
#     return sentence

# def clean_sentence(data):
#     keywords = read_keywords_from_file('/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/My_Data_process/data_in/keyword.txt')
#     for index, row in data.iterrows():
#         if row['msg_type'] == 'text' or row['msg_type'] == 'voice':
#             contents = str(row['content']).split("\n------\n")
#             new_content = []
#             for content in contents:
#                 if row['role'] == '客户':
#                    # content = clean_quote_client(content)
#                     content = remove_emojis(content)
#                     delete_or_not = any(keyword in str(row['content']) for keyword in keywords)
#                     if not delete_or_not:
#                         new_content.append(content)
#                     else:
#                         # new_content.append(np.nan)
#                         continue

#                 else:
#                     #content = clean_quote_service(data, index, content)
#                     content = remove_emojis(content)
#                     delete_or_not = any(keyword in str(row['content']) for keyword in keywords)
#                     if not delete_or_not:
#                         new_content.append(content)
#                     else:
#                         # new_content.sappend(np.nan)
#                         continue

#             #new_content.append(content)

#             new_content_str = '\n------\n'.join(new_content)
#             # if new_content==[]:
#             #     new_content=np.nan
#             data.at[index, 'clean_content'] = new_content_str if new_content_str else np.nan

#         else:
#             data.at[index, 'clean_content'] = np.nan

#     return data

def remove_emojis(text):
    # 定义表情符号的正则表达式模式
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"  # 表情符号 (emoticons)
                               "\U0001F300-\U0001F5FF"  # 图形符号 (symbols & pictographs)
                               "\U0001F680-\U0001F6FF"  # 交通与地图符号 (transport & map symbols)
                               "\U0001F1E0-\U0001F1FF"  # 国旗 (flags)
                               "]+", flags=re.UNICODE)

    text_without_emojis = re.sub(emoji_pattern, "", text)

    return text_without_emojis




def read_keywords_from_file(file_path):
    keywords = []
    with open(file_path, 'r') as file:
        for line in file:
            keyword = line.strip()
            keywords.append(keyword)
    return keywords

def clean_quote_client(sentence):
    keyword = '【引用: 】'
    if keyword in sentence:
        quote_index = sentence.find('【引用: 】')
        result_pre=sentence[:quote_index]
        result_ = sentence[quote_index + 8:]
        sentence =  "【引用:"+result_pre+" 】"+result_
    return sentence

def separate_sentence(sentence):
    pattern = r'([\w（）]+[^a-zA-Z0-9\s\.]+|\w+$)'
    separated_sentence = re.findall(pattern, sentence)
    return separated_sentence


def clean_quote_service(sentence):
        keyword = '这是一条引用/回复消息，'
        if keyword in sentence:
            parts = sentence.split('，')
            sentence= "，".join(parts[2:])#包含原句子的对话
            sentence="【引用 】"+sentence
        return sentence


def clean_sentence(data):
    keywords = read_keywords_from_file('/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/My_Data_process/data_in/keyword.txt')
    for index, row in data.iterrows():
        if row['msg_type'] == 'text' or row['msg_type'] == 'voice':
            contents = str(row['content']).split("\n------\n")
            new_content = []
            for content in contents:
                if row['role'] == '客户':
                    content = clean_quote_client(content)
                    content = remove_emojis(content)
                    delete_or_not = any(keyword in str(row['content']) for keyword in keywords)
                    if not delete_or_not:
                        new_content.append(content)
                    else:
                        continue

                else:
                    content = clean_quote_service(content)
                    content = remove_emojis(content)
                    delete_or_not = any(keyword in str(row['content']) for keyword in keywords)
                    if not delete_or_not:
                        new_content.append(content)
                    else:
                        continue

            #new_content.append(content)

            new_content_str = '\n------\n'.join(new_content)
            data.at[index, 'clean_content'] = new_content_str if new_content_str else np.nan

        else:
            data.at[index, 'clean_content'] = np.nan

    return data

if __name__ == "__main__":
    # file_path = '/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/My_Data_process/群聊不同id-3150个文件/wrLg07EQAA_2sMy01YlIeXGZr4SZTMtg.xlsx'
    # output_file_path = '/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/My_Data_process/群聊不同id-3150个文件-清洗后/wrLg07EQAA_2sMy01YlIeXGZr4SZTMtg.xlsx'
    base_in_dir = "/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/My_Data_process/群聊不同id-3150个文件"
    base_out_dir = "/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/My_Data_process/群聊不同id-3150个文件-清洗后"
    for file in os.listdir(base_in_dir):
        # if str(file) =="wrLg07EQAA_YFBCYNosPJ59DDdF0kNNQ.xlsx":
        file_in_path = os.path.join(base_in_dir,file)
        file_out_path = os.path.join(base_out_dir,file)
        columns_name = ['role', 'content', 'role_id', 'msg_type', 'msg_time', 'msg_id', 'room_id']
        data = data_get(file_in_path, None, columns_name)
        data = data.get_data()
        new_data = clean_sentence(data)
        new_data.to_excel(file_out_path, index=False)
        # 不要content列 只要clean_content列
        # for txt in list(new_data['clean_content']):
        #     print(txt)
        filnal_new_data = new_data.dropna(subset=['clean_content'])
        filnal_new_data = filnal_new_data.drop(columns=["content"])
        filnal_new_data.to_excel(os.path.join(base_out_dir.replace("清洗后","清洗后的最终"),file),index=False)
        # break
