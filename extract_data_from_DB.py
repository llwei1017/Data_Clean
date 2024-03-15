import pandas as pd
import psycopg2
import mysql.connector
import pandas as pd
import os


class DBdata():
    def __init__(self):
        self.db_config = {
        'host': '',
        'user': '',
        'password': '',
        'database': '', # 真实生产环境，测试环境
        'port': 
        }
        self.data_op_dir = r"/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/My_Data_process/data/out"
        self.db_connection = psycopg2.connect(**self.db_config)
        
    # 释放连接
    def _release_connection(self):
        self.db_connection.close()

     # 定位企微群聊链接
    def db_extract(self, sql_order):
        # origin_header = ['acctdt', 'user_id', 'user_vx_name', 'user_vx_id', 'first_channel', 'user_label_list', 'group_id'\
        #         'group_name','group_member_list','group_create_time','group_owner_id','group_owner_name','broker_list'\
        #         'action_time','action_name','json_content']
        header = ['group_name','group_id','acctdt']
        update_query = sql_order
        cursor = self.db_connection.cursor()
        cursor.execute(update_query)
        results = cursor.fetchall()
        one_data = pd.DataFrame(results,columns=header)
        one_data.to_csv(os.path.join(self.data_op_dir,"近一年客户服务群id抽取.csv"),sep=',',index=False)
        cursor.close()
        self._release_connection()
    

if __name__ == "__main__":
    db = DBdata()
    sql_order = """
    select distinct(group_name),group_id,acctdt
    from public.ads_enterprise_group_chat_detail
    where acctdt > '2023-01-01' and group_name LIKE '%客户服务群%'
    """
    db.db_extract(sql_order)