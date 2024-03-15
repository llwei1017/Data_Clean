import pandas as pd
import os
import re
import pandas as pd
import numpy as np
import Levenshtein
from pandas.core.frame import DataFrame
from extract_data_from_DB import DBdata

class FindErrorException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class GetData:
    def __init__(self, in_dir, op_dir):
        self.base_in_dir = in_dir
        self.base_op_dir = op_dir
    
    def _delete_error_label(self, q_l_id_local_filename):
        """去除跑问题标签数据中，异常标签的数据->另存new-data_q_l_id.xlsx"""
        data_q_l_id = pd.read_excel(os.path.join(self.base_in_dir, q_l_id_local_filename))
        # 去除nan
        data_q_l_id = data_q_l_id.dropna(subset="user_id")
        groupping = data_q_l_id.groupby(by="question")
        # 统计同一个问题 分配不同或者相同标签的数据条数
        count_same_q_diff_l = 0
        same_q_diff_l_df = pd.DataFrame(columns=['question', 'label', 'user_id', 'msg_time', '原始未合并问题'])
        count_same_q_same_l = 0
        # drop标记列表
        del_index_list = []

        for group_name,groups in groupping:
            # 同一问题有不同label, 全部删除
            if len(list(groups['label']))>1 and len(set(list(groups['label'])))>1:
                count_same_q_diff_l += 1
                same_q_diff_l_df = pd.concat([groups,same_q_diff_l_df]) if not same_q_diff_l_df.empty else groups.copy()
                for idx in list(groups.index):
                    del_index_list.append(idx)
            # 同一问题有同个label, 只保留一份
            if len(list(groups['label']))>1 and len(set(list(groups['label'])))==1:
                count_same_q_same_l += 1
                for idx in list(groups.index)[:-1]:
                    del_index_list.append(idx)

        print(f"same_q_with_same_l条数: {count_same_q_diff_l}\n")
        print(f"same_q_with_diff_l条数: {count_same_q_same_l}\n")

        # 备份标签异常的数据
        same_q_diff_l_df = same_q_diff_l_df.reset_index()
        same_q_diff_l_df.to_excel(os.path.join(self.base_op_dir,"same_q_diff_l_df.xlsx"), index=False)

        # 删除
        print(f"删除错误label前，共{len(data_q_l_id)}条数据\n")
        new_data_q_l = data_q_l_id.drop(index=del_index_list).reset_index(drop=True)
        op_path = os.path.join(self.base_op_dir,"new-data_q_l_id.xlsx")
        new_data_q_l.to_excel(op_path, index=False)
        print(f"删除错误label后，共{len(new_data_q_l)}条数据\n")
        return op_path

    def _generate_sameID(self, db_file_name, chat_local_filename):
        """对比mysql和本地chat_120s, 提取chat_120在库中的所有客户服务群聊id->另存重复群聊id.xlsx"""
        db_data = pd.read_csv(os.path.join(self.base_op_dir,db_file_name),sep=',')
        chat_120s_data = pd.read_excel(os.path.join(self.base_in_dir,chat_local_filename))

        groupID = list(db_data.loc[:,"group_id"].drop_duplicates(keep='first'))
        roomID = list(chat_120s_data.loc[:,'room_id'].drop_duplicates(keep='first'))
        sameID = list(set(roomID).intersection(set(groupID)))
        print(f"近一年库里总共有{len(groupID)}个不同的客户服务群，共计{len(chat_120s_data)}条消息\n")
        print(f"过滤之前，本地chat_120s共有{len(roomID)}个不同的客户服务群\n")

        # 保存
        dict_data ={"重复群聊id" : sameID}
        df_data = DataFrame(dict_data)
        op_path = os.path.join(self.base_op_dir,"重复群聊id.xlsx")
        df_data.to_excel(op_path,index=False)
        
        return sameID

    def _groupby_sameID(self, db_file_name, chat_local_filename):
        """根据重复群聊id.xlsx文件，先过滤一遍，在根据群聊id分组->另存到base_op_dir下群聊不同id-{xxx}个文件"""
        sameID_list = self._generate_sameID(db_file_name,chat_local_filename)
        chat_120s_data = pd.read_excel(os.path.join(self.base_in_dir,chat_local_filename))
        # 只保留重复id的记录
        del_list = []
        for idx, row in chat_120s_data.iterrows():
            del_list.append(idx) if row['room_id'] not in sameID_list else None
            # delflag = 0
            # for sameid in sameID_list:
            #     if row['room_id'] == sameid:
            #         delflag = 1
            #         break
            # if not delflag:
            #     # room_id没有在same_id列表中
            #     del_list.append(idx)

        new_chat_120s_data = chat_120s_data.drop(index=del_list).reset_index(drop=True)
        new_chat_120s_data.to_excel(os.path.join(self.base_op_dir,"final"+chat_local_filename), index=False)
        print(f"过滤之后，本地chat_120s共有{len(sameID_list)}个不同的客户服务群，共计{len(new_chat_120s_data)}条消息\n")

        # 按照room_id分组 
        os.makedirs(os.path.join(self.base_op_dir,f"群聊不同id-{len(sameID_list)}个文件"), exist_ok=True)
        op_dir_path = os.path.join(self.base_op_dir,f"群聊不同id-{len(sameID_list)}个文件")
        groupbying = new_chat_120s_data.groupby(by="room_id")
        # self.groupby_dir = op_dir_path
        for groupname,groups in groupbying:
            op_path = os.path.join(op_dir_path, groupname + ".xlsx")
            groups.to_excel(op_path,index=False)
        
        return op_dir_path
    
    def _filter_words(self,singel_q):
        words = ['嗯','嗯嗯','嗯好',\
                '好','好吧','好的','好嘞','好啊','好叭','好的～', '好喔','好的呢','好的，谢谢了','好的谢谢','好的，谢谢',\
                '哦','哦哦',
                '感谢～','谢谢','收到了，谢谢','好谢谢','好谢谢',\
                '对','对的']
        for word in words:
            if singel_q == word:
                return True
        return False
    
    def _extract_ID(self):
        """
            ID_dict: 存取usr_id对应的群聊id
        """
        ID_df = pd.read_excel(os.path.join(self.base_in_dir,"userID-2-roomID.xlsx")) # roleIDd对应roomID存放的地址
        ID_dict = dict()
        for id,row in ID_df.iterrows():
            ID_dict[row['role_id']] = row['room_id']
        self.ID_dict = ID_dict
    
    #######################对group_by后的几千个文件的数据清洗逻辑################################
    def _remove_emojis(self,text):
        # 定义表情符号的正则表达式模式
        emoji_pattern = re.compile("["
                                "\U0001F600-\U0001F64F"  # 表情符号 (emoticons)
                                "\U0001F300-\U0001F5FF"  # 图形符号 (symbols & pictographs)
                                "\U0001F680-\U0001F6FF"  # 交通与地图符号 (transport & map symbols)
                                "\U0001F1E0-\U0001F1FF"  # 国旗 (flags)
                                "]+", flags=re.UNICODE)

        text_without_emojis = re.sub(emoji_pattern, "", text)

        return text_without_emojis
    
    def _clean_quote_client(self,sentence):
        keyword = '【引用: 】'
        if keyword in sentence:
            quote_index = sentence.find('【引用: 】')
            result_pre=sentence[:quote_index]
            result_ = sentence[quote_index + 8:]
            sentence =  "【引用:"+result_pre+" 】"+result_
        return sentence
    def _clean_quote_service(self,sentence):
        keyword = '这是一条引用/回复消息，'
        if keyword in sentence:
            parts = sentence.split('，')
            sentence= "，".join(parts[2:])#包含原句子的对话
            sentence="【引用 】"+sentence
        return sentence
    
    def _clean_(self,message,role,keywords):
        message=self._remove_emojis(message)
        if role =="客户":
            message=self._clean_quote_client(message)
            return message
        else :
            message = self._clean_quote_service(message)
            delete_or_not = any(keyword in message for keyword in keywords)
            if not delete_or_not:
                return message
    
    def _clean_chat_data(self,keyword_file,file_path):
        data=pd.read_excel(file_path)       
        keyword_filepath = os.path.join(self.base_in_dir,keyword_file)
        keywords=[]
        with open (keyword_filepath,'r') as file :
            for line in file :
                keyword = line.strip()
                keywords.append(keyword)
        for index,row in data.iterrows():
            if not(row['msg_type'] == 'text' or row['msg_type'] == 'voice'):
                data.at[index, 'clean_content'] = ''
            
            else:
                contents = str(row['content']).split("\n------\n")
                new_content = []
                
                for content in contents:
                    content=self._clean_(content,row['role'],keywords)
                    if content is not None:
                        new_content.append(content)

                new_content_str = '\n------\n'.join(new_content)
                # print(new_content_str)

                data.at[index, 'clean_content'] = new_content_str if new_content_str else ''
        
        data.replace("", np.nan, inplace=True)
        return data

    def chat_data_process(self, db_file_name, keyword_filename,chat_local_filename):
        """对group_by得到的几千个文件，遍历每个文件，保存清洗后和清洗后的最终文件"""
        groupby_dir = self._groupby_sameID(db_file_name, chat_local_filename)
        os.makedirs(os.path.join(groupby_dir,"清洗后"),exist_ok=True)
        os.makedirs(os.path.join(groupby_dir,"清洗后最终"),exist_ok=True)
        #######################group_by每个文件的清洗逻辑################################
        for chat_file in os.listdir(groupby_dir):
            if str(chat_file).endswith(".xlsx"):
                chat_path = os.path.join(groupby_dir,chat_file)
                # 保留清洗前`content`和清洗后的`clean_content`
                clean_groupby_op_path = os.path.join(groupby_dir,"清洗后",chat_file)
                # 仅保留清洗后的`clean_content`
                final_clean_op_path = os.path.join(groupby_dir,"清洗后最终",chat_file)
                df = self._clean_chat_data(keyword_filename, chat_path)
                df.to_excel(clean_groupby_op_path,index=False)
                final_df = df.dropna(subset=['clean_content'], how='all')
                final_df.to_excel(final_clean_op_path,index=False)

        self.groupby_dir = os.path.join(groupby_dir,"清洗后最终")   


    def extract_history(self, room_id, question, q_concat):
        # 根据room_id找群聊文件
        temp_history = [] 
        history = [] # [temp_history,distance]
        single_chat_120s = pd.read_excel(os.path.join(self.groupby_dir, room_id+".xlsx")) # 每个单独群聊文件存放的地址
        for id,row in single_chat_120s.iterrows():
            # 定位问题出现的行
            if question in row['clean_content']:
                content_lis = row['clean_content'].split("\n------\n")
                content = "".join(content_lis)
                distance = Levenshtein.distance(q_concat, content) # 记录疑似question出现的位置和与q_concat的编辑距离
                history.append(["".join(temp_history) + row['role'] + ": \n" + content, distance])
            else:
                # 留存客服与客户对话
                chat = row['role'] + ": \n" + row['clean_content'] + "\n\n"
                temp_history.append(chat)
        sorted_history_list = sorted(history, key=lambda x:x[-1]) # 按照编辑距离升序排序
        return sorted_history_list[0][0] # 取最小值

    def construct_data(self, data_q_l_id_filename):
        self._extract_ID()
        new_filename = self._delete_error_label(data_q_l_id_filename)
        new_data_q_l = pd.read_excel(os.path.join(self.base_op_dir,new_filename))

        # 针对concat_q分割不同的单个问题
        new_id = 0
        no_find_usrids = pd.DataFrame(columns=['question','label','user_id','msg_time','原始未合并问题'])
        no_find_usrids_op_path = os.path.join(self.base_op_dir,"new-data_q_l_id中无法溯源的数据.xlsx")

        # 溯源
        new_idx = 0
        singleQ_label_id_history = pd.DataFrame(columns=["singele_q","label","user_id","history"])
        singleQ_label_id_history_op_path = os.path.join(self.base_op_dir,"new-data_q_l_id中可溯源的数据.xlsx")

        for id,row in new_data_q_l.iterrows():
            concat_q = new_data_q_l.loc[id,'question']
            q_list = new_data_q_l.loc[id,'原始未合并问题'].split("\n------\n")
            extract_history_flag=0
            history_list=[]
            label = new_data_q_l.loc[id,'label']
            user_id = new_data_q_l.loc[id,"user_id"]
            # 枚举每一个问题
            for idx,single_q in enumerate(q_list):
                single_q = single_q.replace("\n","").replace(" ","")
                if single_q and self._filter_words(single_q) is not True:
                    try:roomid = self.ID_dict[user_id] # 根据usr_id查找群聊id
                    except Exception as e:
                        print(f"{e.message}: 找不到该usr_id对应的群聊\n")
                        pass
                    try:history = self.extract_history(roomid, single_q, concat_q)
                    except Exception as e:
                        print(f"{e.message}: 无法回溯该问题{single_q}的历史聊天记录\n")
                        pass
                    if len(history):
                        extract_history_flag += 1
                        history_list.append("".join(history))

            if extract_history_flag and len(set(history_list))>=1:
                history_text = (sorted(set(history_list), key=lambda x:len(x)))[-1] # 备选history中挑最长的
                # 保存新行内容[singele_q,label,user_id,history]
                singleQ_label_id_history.loc[new_idx] = [concat_q,label,user_id,history_text]
                new_idx += 1
            elif extract_history_flag == 0:
                no_find_usrids.loc[new_id]=row
                new_id+=1
        # 保存
        print(f"现在singleQ_label_id_history一共有{new_idx}条数据\n")
        singleQ_label_id_history.to_excel(os.path.join(self.base_op_dir,"singleQ_label_id_history-2.xlsx"),index=False)
        # 保存无法溯源的question和对应user_id
        no_find_usrids.to_excel(os.path.join(self.base_op_dir,"no_find_usrids.xlsx"),index=False)

if __name__ == "__main__":
    base_in_dir = r"/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/My_Data_process/data/in"
    base_op_dir = r"/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/My_Data_process/data/out"
    keyword_file_name = "keyword.txt"
    db_file_name = "近一年客户服务群id抽取.csv"
    chat_local_filename = "chat_120s_merged.xlsx"
    data_q_l_id_filename = "data_q_l_id.xlsx"
    getdata = GetData(base_in_dir,base_op_dir)
    getdata.chat_data_process(db_file_name,keyword_file_name,chat_local_filename)
    getdata.construct_data(data_q_l_id_filename)