import os
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import re
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Union
from torch import cuda
import pandas as pd
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
from transformers.generation import GenerationConfig
# from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
import subprocess



local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

class InvalidQClassifier(nn.Module):
    def __init__(self, model_load_path):
        super(InvalidQClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_load_path, num_hidden_layers=12)
        self.dropout = nn.Dropout(float(0.3))
        self.classifier = nn.Linear(768, int(2))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, ids, mask, token_type_ids, y=None):
        _, pooler_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.dropout(pooler_output)
        output = self.classifier(output)
        if y is not None:
            loss = self.loss(output, y)
            return loss
        return output

class DataCleaner:
    def __init__(self):
        self.device = None
        self.invalid_q_tokenizer = None 
        self.invalid_q_model = None
        self.Qwen_tokenizer = None
        self.Qwen_model = None 
        self.Qwen_generation_config = None
        self.instruction_template = None
        os.environ['OSS_ACCESS_KEY_ID'] = ''
        os.environ['OSS_ACCESS_KEY_SECRET'] = ''
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
        self.bucket = oss2.Bucket(auth, 'https://oss-cn-shenzhen.aliyuncs.com', 'xxxxxxxx-ai-policy-information')
        

    def _init_invalid_q_model(self):

        model_load_path = '/data/disk4/tmp/project_oriented_model/bert-base-chinese-classify_invalid_questions'
        tokenizer = BertTokenizer.from_pretrained(model_load_path)

        model = InvalidQClassifier(model_load_path)
        model.load_state_dict(torch.load('/data/disk4/tmp/project_oriented_model/bert-base-chinese-classify_invalid_questions/model.pth'))
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _init_Qwen_model(self):
        model_path = r"/data/disk4/tmp/project_oriented_model/Qwen-14B-chat_extract_person_product_name"
        # model_path = r"/data/disk4/tmp/project_oriented_model/microsoft_phi-2-chat-entity_name"
        
        instruction_template = """你是命名实体识别专家，提取下列语句中人名；提取下列语句中保险产品的完整产品名,蜗牛是我们公司的名字，不是产品。返回json格式：{{'人名'：[],'产品名'：[]}}。问题如下：\n```{question}``` """

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            pad_token_id=tokenizer.pad_token_id,
            device_map="auto",
            trust_remote_code=True).eval()
        generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)
        return tokenizer, model, generation_config, instruction_template
    
    
    # Qwen-14B 提取 人名[PERSON_NAME] 产品名[PRODUCT_NAME]
    def _extract_name_from_q(self, response, input_text):
        try:
            response_dict = json.loads(response)
        except:
            print("--------实体命名结果json格式有误--------")
            return input_text
        if response_dict.get('产品名') is not None:
            product_name_list = response_dict['产品名']
            for product_name in product_name_list:
                input_text = input_text.replace(product_name, '[PRODUCT_NAME]')
        if response_dict.get('人名') is not None:
            people_name_list = response_dict['人名']
            for people_name in people_name_list:
                input_text = input_text.replace(people_name, '[PERSON_NAME]')
        # print(input_text)
        return input_text
    

    # True: 无效  False: 有效    label_name_dict = {0: '无效', 1: '有效'}
    def _question_is_invalid(self, input_text): 
        inputs = self.invalid_q_tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=20,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device, dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device, dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0).to(self.device, dtype=torch.long)

        pooler_output = self.invalid_q_model(ids, mask, token_type_ids)
        pred_label = torch.argmax(pooler_output, axis=1).tolist()[0]
        _is_invalid_bool = True if pred_label == 0 else False
        return _is_invalid_bool


    def _clean_text(self, input_text):
        # 使用正则表达式匹配表情包[xx]
        # input_text = re.sub(r'\[(.*?)\]', '', input_text)
        # 匹配客户** **
        input_text = re.sub(r'\*\*(.*?)\*\*', r'\1', input_text)
        # 使用正则表达式匹配url
        url_pattern = r'\{(.*?)\}'
        input_text = re.sub(url_pattern, lambda match: self._clean_url(match.group(1)), input_text)
        # 使用正则表达式匹配"xxxx"
        input_text = re.sub(r'\"(.*?)\"', r'\1', input_text)

        return input_text

    def _clean_url(self, matched_text):
        # 清洗链接title，浏览后修改此部分做指定清洗
        clean_list = ['产品详情', '了解蜗牛', '蜗牛保险']
        matched_text = matched_text.replace('""', '"').replace("'", '"').replace("：", ":")
        
        try:
            json_dict = json.loads('{' + matched_text + '}')
            if json_dict.get('title') is not None:
                if json_dict['title'] in clean_list:
                    return '【链接：' + json_dict['description'].replace('\n', ',') + '】'
                else:
                    return '【链接：' + json_dict['title'] + '】'
            else:
                return ''
        except:
            return matched_text

    #_clean_data_4_labelStudio in:file <- conv out:file <- clean_conv
    def _clean_data_4_labelStudio(self, input_file_path, output_path):
        result_strings = []

        with open(input_file_path, 'r', encoding='utf-8') as file:
            input_text = file.read()
            input_text = input_text.replace(":", "：")
            messages = re.split(r'\n(?=客户：|经纪人：)', input_text)
            # 合并连续同一人留言
            last_author = ''

            for message in messages:
                author, text = message[:message.find('：')], message[message.find("：") + 1:]
                if text.startswith(" "):
                    text = text[1:]
                text = text.replace('\n', '，')
                text = self._clean_text(text)
                if text:
                    if author == last_author:
                        result_strings[-1]["text"] += f'\n{text}'
                    else:
                        result_strings.append({"text": text, "author": author})
                        last_author = author
        save_json = {"data":{"content": result_strings}}
        with open(output_path,'w',encoding='utf-8') as file:
            json.dump(save_json, file, indent=4, ensure_ascii=False)

    # 对话问题提取，清洗
    # 读取json文件
    def _read_json_data(self, input_q_path:str) -> list:
        with open(input_q_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        return json_data
    
    # 清洗引用部分
    def _clean_quote(self, text:str) -> str:
        text = text.replace("- - - - - - - - - - - - - - -", " @#$ ").replace("---------------", " @#$ ").replace("------", " @#$ ")
        text_split = text.split(" @#$ ")
        if len(text_split) == 1:
            return text
        return_text = f'【引用: {text_split[0][text_split[0].rfind("：")+1:]}】 {text_split[1]}'
        if len(text_split) == 2:
            return return_text
        else:
            temp = f"{return_text} @#$ {' @#$ '.join([x for x in text_split[2:]])}"
            return self._clean_quote(temp)

    # 获取源一对多问题提取数据
    def _get_source_q_data(self, q_data:list) -> pd.DataFrame:
        df = pd.DataFrame(columns=["input", "original_q", "modify_q"])
        df_index = 0
        for contents in q_data:
            input_txt = ""
            for content in contents['data']['content']:
                if content['author'] == '客户':
                    if "- - - - - - - - - - - - - - -" in content['text'] or '------' in content['text']:
                        mess = self._clean_quote(content['text'])
                        input_txt += f"{mess}\n"
                        continue
                    input_txt += f"{content['text']}\n"
            try:
                # 空问题导入
                results = contents['annotations'][0]['result']
                if len(results) == 0 or (len(results) == 1 and results[0]['type'] == 'choices'):
                    if input_txt != '':
                        new_list = [input_txt, "", ""]
                        df.loc[df_index] = new_list
                        df_index += 1
                        continue
                q_id_dict = dict()
                relation_list = []
                for question in results:
                    # if question['type'] == 'labels':
                    if question['type'] == 'paragraphlabels':
                        q_id_dict[question['id']] = {"original_q":question['value']['text'], "modify_q":""}
                    elif question['type'] == 'textarea':
                        q_id_dict[question['id']]["modify_q"] = question['value']['text'][0]
                    elif question['type'] == 'relation' and question['from_id'] != question['to_id']:
                        flag = 1
                        for index,start_id in enumerate(relation_list):
                            if question['from_id'] in start_id:
                                relation_list[index].append(question['to_id'])
                                flag = 0
                                continue
                            if question['to_id'] in start_id:
                                relation_list[index].append(question['from_id'])
                                flag = 0
                                continue
                        if flag:
                            relation_list.append([question['from_id'],question['to_id']])
                                
                for from_id in relation_list:
                    try:
                        modify_q = q_id_dict[from_id[0]]['modify_q'] if q_id_dict[from_id[0]]['modify_q'] != "" else q_id_dict[from_id[-1]]['modify_q']
                        temp = ""
                        flag_temp = set()
                        for next_id in from_id[1:]:
                            if next_id not in flag_temp:
                                temp += f"\n{q_id_dict[next_id]['original_q']}"
                                del q_id_dict[next_id]
                            flag_temp.add(next_id)
                        q_id_dict[from_id[0]]['modify_q'] = modify_q
                        q_id_dict[from_id[0]]['original_q'] = f"{q_id_dict[from_id[0]]['original_q']}{temp}"
                    except:
                        continue
                for value in q_id_dict.values():

                    df.loc[df_index] = [input_txt, value['original_q'], self.clean_sentence(value['modify_q'])]
                    df_index += 1
            except:
                df.loc[df_index] = [input_txt, '', '']
                df_index += 1
        return df
    
    
    # 合并问题，/n -> ，
    def _merge_question(self, input_txt:str, original_q:str) -> Union[str, str]:
        q_list = re.split(r'\\n|\n', original_q)

        with ThreadPoolExecutor(max_workers=30) as executor:
            q_list = list(executor.map(self.clean_sentence, q_list))
        # q_list = [self.clean_sentence(i) for i in q_list]

        input_list = re.split(r'\n', input_txt)

        with ThreadPoolExecutor(max_workers=30) as executor:
            input_list = list(executor.map(self.clean_sentence, input_list))
        # input_list = [self.clean_sentence(i) for i in input_list]

        return_input_txt = '\n'.join([i for i in input_list if i!=""])

        return_original_q = self.clean_sentence(original_q)

        if len(q_list) > 1:
            del_start = -1
            return_original_q = ""
            for i in q_list:
                if i != "":
                    for index,one_input in enumerate(input_list):
                        if i in one_input:
                            if del_start == -1:
                                del_start = index
                            return_original_q += f"{i}，"
                            input_list.remove(one_input)
            return_original_q = return_original_q.strip("，")
            input_list.insert(del_start, return_original_q)
            return_input_txt = '\n'.join([i for i in input_list if i!=""])
        return return_input_txt, return_original_q
    
    # 清洗df文件
    def _clean_df_data(self, df: pd.DataFrame) -> pd.DataFrame:
        temp_df = df
        continue_list = ["嗯嗯", "噢噢", "好的", "哦哦", "嗯", "噢", "哦", "谢谢", "你好", "好的谢谢"]

        del_flag = []
        for key,rows in temp_df.iterrows():
            if rows['original_q'] in continue_list:
                del_flag.append(key)
                continue
            if len(rows['original_q'].strip()) < 5 and rows['original_q'] == '' and rows['input'].count('\n') < 2:
                del_flag.append(key)
                continue
            input_txt, original_q = self._merge_question(temp_df['input'][key], rows['original_q'])
            temp_df.loc[temp_df['input'] == temp_df['input'][key], 'input'] = input_txt
            temp_df.loc[key, "original_q"] = original_q
        # 去除没用信息的行
        temp_df = temp_df.drop(index=del_flag).reset_index(drop=True)
        return temp_df
    
    def _get_pred_data(self, question_list:list) -> list: # in:问题列表；out:推理数据
        return_pred = []
        for value in question_list:
            return_pred.append({
                "instruction": "你是命名实体识别专家，提取下列语句中人名；提取下列语句中保险产品的完整产品名,蜗牛是我们公司的名字，不是产品。返回json格式：{'人名'：[],'产品名'：[]}",
                "input": value,
                "output": value
            })
        return return_pred
    
    # data_info_backup_path = os.path.join(data_info_dir_path, "data_info_backup.json")
    # 

    def _get_pred_result(self, pred_data, llama_dir_path, output_dir_path):
        os.chdir(llama_dir_path)
        pred_data_path = os.path.join(output_dir_path, "pred_data")
        if not os.path.exists(pred_data_path):
            os.mkdir(pred_data_path)
        # 存入推理数据
        with open(os.path.join(pred_data_path, "pred_data.json"), 'w', encoding='utf-8') as save_file:
            json.dump(pred_data, save_file, indent=4, ensure_ascii=False)
        # 修改llama_factory里的dataset_info.json
        data_info_path = fr"{llama_dir_path}/data/dataset_info.json"
        with open(data_info_path, "r", encoding="utf-8") as file:
            data_info = json.load(file)
        #add_dataset
        data_info["custom_pred_dataset"] = {
            "file_name": os.path.join(pred_data_path, "pred_data.json"),
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": ""
            }}
        with open(data_info_path, "w", newline="", encoding="utf-8") as file:
            json.dump(data_info, file, ensure_ascii=False, indent=4)
        # 开始推理
        pred_sh_path = llama_dir_path.replace("LLaMA-Factory", "ai-project/data_cleaning/extract_questions/predict.sh")
        python_path = fr"{llama_dir_path}/src/train_bash.py"
        subprocess.run(['sh', pred_sh_path, pred_data_path, python_path])
        
        pred_result_dict = {}
        with open(fr"{pred_data_path}/generated_predictions.jsonl", 'r', encoding="utf-8") as file:
            for line in file:
                json_object = json.loads(line)
                pred_result_dict[json_object['label']] = json_object['predict']
                
        return pred_result_dict
        
    def _pred_data_2_df(self, df_data, pred_result_dict):
        for index, row in df_data.iterrows():
            row_input_list = row['input'].split('\n')
            row_dialogue = ''
            for row_question in row_input_list:
                if row_question in pred_result_dict:
                    response = pred_result_dict[row_question]
                    row_dialogue += self._extract_name_from_q(response, row_question) + '\n'
                else:
                    row_dialogue += row_question + '\n'
            df_data.loc[index, "input"] = row_dialogue
            if row['original_q'] != '':
                row_original_q = self._extract_name_from_q(pred_result_dict[row['original_q']], row['original_q'])
                df_data.loc[index, "original_q"] = row_original_q
            if row['modify_q'] != '':
                row_modify_q = self._extract_name_from_q(pred_result_dict[row['modify_q']], row['modify_q'])
                df_data.loc[index, "modify_q"] = row_modify_q
        return df_data

    
    def _get_chat(self, text: str) -> str:
        question =f'''你是命名实体识别专家，提取下列语句中人名；提取下列语句中保险产品的完整产品名,蜗牛是我们公司的名字，不是产品。返回json格式：{{'人名'：[],'产品名'：[]}}
        问题如下：{text}
        '''
        response, history = self.Qwen_model.chat(self.Qwen_tokenizer, question, history=None)
        return response

    # 流程合并
    def _clean_data_4_training(self, llama_path:str, output_dir_path:str, json_path:str, output_path:str) -> None:
        q_json_data = self._read_json_data(json_path)   # 读取json
        df_data = self._get_source_q_data(q_json_data)  # 获取源df
        df_data = self._clean_df_data(df_data)          # 清洗源df
        q_list = []
        for key,rows in df_data.iterrows():
            if rows['original_q'] != '':
                q_list.append(rows['original_q'])
            if rows['modify_q'] != '':
                q_list.append(rows['modify_q'])
        pred_data = self._get_pred_data(q_list) # 获取推理数据
        # 跑推理模型
        pred_result_dict = self._get_pred_result(pred_data,llama_path,output_dir_path)
        df_data = self._pred_data_2_df(df_data, pred_result_dict)
        # df_data = self._mask_sentence(df_data)  # mask产品名，人名
        df_data.to_excel(output_path, index=False)      # 输出xlsx


    def batch_clean_data_4_training(self, input_data_dir_path, output_dir_path, llama_path):
        subfolders_names = [f for f in os.listdir(input_data_dir_path) if f.endswith('.json')]

        for file in tqdm(subfolders_names, desc='batch_clean_data_4_training', position=0):
            json_file = os.path.join(input_data_dir_path, file)
            output_path = os.path.join(output_dir_path, file.replace('.json', '.xlsx'))
            self._clean_data_4_training(llama_path, output_dir_path, json_file, output_path)
            bucket_file_path = f"label_studio_train_data/{file.replace('.json', '.xlsx')}"
            self.bucket.put_object_from_file(bucket_file_path, output_path)
                
    
    def batch_clean_data_4_lableStudio(self, input_data_dir_path, output_dir_path):
        subfolders_names = [f for f in os.listdir(input_data_dir_path) if f.endswith(".txt")]
        save_data = []
        
        for txt_file in tqdm(subfolders_names,total=len(subfolders_names)):
            input_file_path = os.path.join(input_data_dir_path, txt_file) 
            save_data.append(self._clean_data_4_labelStudio(input_file_path).copy())
        save_path = fr"{output_dir_path}/merge.json"
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(save_data, file, indent=4, ensure_ascii=False)
            
    
    def clean_sentence(self, text, mask_sentence=False) -> str:
        if not isinstance(text, (int, float, str)):
            raise TypeError(f"输入类型{type(text)}错误，必须是int、float或str类型")
        text = str(text)
        replacements = {
            '\\n':'\n',
            ":":"：",
            "我已经添加了你，现在我们可以开始聊天了。": "",
            "我通过了你的联系人验证请求，现在我们可以开始聊天了": "",
            "*":"",
            "客户：":"\n",
            "经纪人：":"\n"
            }
        for old_word, new_word in replacements.items():
            text = text.replace(old_word, new_word)
        text_split = [x for x in text.split('\n') if x != ""]
        return_text = ""
        for value in text_split:
            value = value.lstrip("-，")
            value = value.strip()   # 前后空格、换行
            value = value.strip(":：‘’“”\'\"")      # 前后无用标点
            value = re.sub(r"https：//(.*)", '【网址】',value)      # 网址
            value = re.sub(r"\[(.*?)\]", "", value)     # [标签]
            value = re.sub(r"(\d{15}$|\d{18}$|\d{17}(\d|X|x))", "【身份证号码】", value)    # 身份证号码
            value = re.sub(r"1[3456789]\d{9}", "【手机号码】", value)     # 手机号码
            value = self._clean_quote(value)    # 处理引用
            if value != "":
                return_text += f"{value}，" if value[-1] not in ",，？！。；?!;" else value

        # if self._question_is_invalid(return_text):
        #     return ''
        # return_text = self._extract_name_from_q(return_text)
        return_text = return_text.rstrip("，")
        if mask_sentence:
            if self.device == None:
                self.device = 'cuda' if cuda.is_available() else 'cpu'
                self.invalid_q_tokenizer, self.invalid_q_model = self._init_invalid_q_model()
                self.Qwen_tokenizer, self.Qwen_model, self.Qwen_generation_config, self.instruction_template = self._init_Qwen_model()
            pred = self._get_chat(return_text)
            return_text = self._extract_name_from_q(pred, return_text)
        return return_text
    
    def entity_name_from_qlist(self, q_list:list, llama_path:str, output_dir_path): # in:问题列表；out:推理结果xlsx
        q_list = [self.clean_sentence(q) for q in q_list]
        # 获取推理数据
        pred_data = self._get_pred_data(q_list) 
        # 跑推理模型
        pred_result_dict = self._get_pred_result(pred_data,llama_path,output_dir_path)
        result = []
        for key,value in pred_result_dict.items():
            result.append(self._extract_name_from_q(value, key))
        # df = pd.DataFrame(result, columns=['pred_result'])
        # df.to_excel(fr"{output_dir_path}/out.xlsx", index=False)\
        return result

    def clean_paragraph(self, text: str, mask_sentence=False) -> str:
        if not isinstance(text, (int, float, str)):
            raise TypeError(f"输入类型{type(text)}错误，必须是int、float或str类型")
        text = str(text)
        text_split = [x for x in text.split('\n') if x != '']
        for index,value in enumerate(text_split):
            value = self.clean_sentence(value, mask_sentence)
            text_split[index] = value
        return '\n'.join([x for x in text_split if x != ""])
        
      

if __name__ == "__main__":
    data_cleaner = DataCleaner()
    print('mask后：',data_cleaner.clean_sentence('艳姐，妈咪宝贝和小小神兽哪个好呀啊', mask_sentence=True))
    
    # input_data_dir_path = '/data/disk4/home/yongsen/data/lable_studio_json/平安e生保'
    # output_dir_path = '/data/disk4/home/yongsen/data/output/平安e生保'
    # LLAMA_FACTORY_DIR_PATH=r"/data/disk4/home/yongsen/llm_code/LLaMA-Factory"
    # data_cleaner.batch_clean_data_4_training(input_data_dir_path, output_dir_path, LLAMA_FACTORY_DIR_PATH)

    # request_list = ['13：46','@7 73','1069395595301','11月3日星期五【中汇人寿】尊敬的李晓妍']
    # LLAMA_FACTORY_DIR_PATH=r"/data/disk4/home/yongsen/llm_code/LLaMA-Factory"
    # # 结果输出目录，需要修改
    # output_dir_path = r"/data/disk4/home/yongsen/data/output" 
    # result = data_cleaner.entity_name_from_qlist(request_list, LLAMA_FACTORY_DIR_PATH, output_dir_path)
    # print(result)