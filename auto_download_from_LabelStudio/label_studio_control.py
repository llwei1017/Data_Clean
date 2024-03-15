import requests
import os
import time

class DataLoad:
    
    def __init__(self):
        self.cookies = {
            'sessionid': ''}
        self.authorization_token = ''
        # self.html_style = {
        #     '问题提取': '<View>\n  <Header value="完成该篇聊天记录的标注时，请点击下方勾选"/>\n           <Choices name="complete" toName="content" choice="single">\n                <Choice value="我已完成该篇标注"/>\n           </Choices>\n\n  <View className="main_text" style="height: 700px; overflow: auto;">\n   <Paragraphs name="content" value="$content" background="#166a45" layout="dialogue"/>\n  </View>\n  <Style> \n\n       .main_text {\n            white-space: pre-wrap;\n        }\n\n    </Style>\n           <ParagraphLabels name="question" toName="content">\n            <Label value="问题" background="#166a45" hotkey="alt+1"></Label>\n        </ParagraphLabels>\n\n\n      <View visibleWhen="region-selected" whenLabelValue="问题">\n                <Header value="修改问题"/>\n       <TextArea name="modify_quetion" toName="content" perRegion="true" placeholder="修改问题" ></TextArea>\n    </View>\n </View>',
        #     '问题分类': '<View>\n  <View style="display: flex">\n    <View style="height: 270px; overflow: auto; width: 70%">\n     <Paragraphs style="height: 140px" name="question" value="$question" layout="dialogue"/>\n     <Textarea style="height: 40px;" name="input" toName="question" placeholder="改进原问题"/>\n    </View>  \n\n      <View>\n\t<Text name="text-1" value="$question" granularity="word" highlightColor="#ff0000" />\n \t <Taxonomy name="taxonomy" toName="text-1" minWidth = "350px"  style="height: 100px;margin-bottom: 50px;" placeHolder="点击选择" apiUrl="https://insnai-ai-policy-information.oss-cn-shenzhen.aliyuncs.com/labels.json" />\n</View>\n      \n </View>\n  <View style="height: 500px; overflow: auto;">\n   <Paragraphs name="content" value="$content" layout="dialogue"/>\n  </View>\n</View>'
        # }
        # self.logger = self.logger_config()
        
    def _get_label_studio_info(self): # 获取label studio列表信息
        label_studio_info = []
        for page in range(1, 3):
            params = {
                'page': page,
                'page_size': '100',
                'include': 'id,title,created_by,created_at,color,is_published,assignment_settings',
            }
            response_json = requests.get('http://label.xxxx.com/api/projects', params=params, cookies=self.cookies, verify=False).json()
            # print(response_json)
            label_studio_info.extend(response_json['results'])
        ls_dict = dict()
        for value in reversed(label_studio_info):
            ls_dict[value['title']] = value
        return ls_dict
    
    def _download_data(self, request_list, ls_dict, output_dir_path): # in:请求下载列表，label studio信息列表； out:输出json路径目录
        for value in request_list:
            print(f'------------下载 "{value}" 中------------')
            if value not in ls_dict:
                print(f"{value}有误，获取不到相关信息。")
                continue
            project_id = ls_dict[value]['id']
            project_name = ls_dict[value]['title']
            save_path = os.path.join(output_dir_path, ls_dict[value]['title']+'.json')
                
            try:
                response = requests.get(f'http://label.xxx.com/api/projects/{project_id}/export', params={'exportType': "JSON"}, cookies=self.cookies)
                with open(save_path, 'wb') as file:
                    file.write(response.content)
                print(f'下载 "{project_name}" 成功')
            except:
                print(f'error,下载{project_name}异常,链接：http://label.xxx.com/api/projects/{project_id}/export')


    # output_dir_path 文件输出的路径，page_total label studio上的总页数
    def download_data_from_studio(self, request_list, output_dir_path):
        ls_dict = self._get_label_studio_info()
        timestamp = int(time.time())
        os.mkdir(os.path.join(output_dir_path, str(timestamp)))
        self._download_data(request_list, ls_dict, os.path.join(output_dir_path, str(timestamp)))
        return os.path.join(output_dir_path, str(timestamp))

if __name__ == "__main__":
    label = DataLoad()
    output_path = r"/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/ai_project"
    request = ["中意一生_问题提取","金禧一生_问题提取"]
    label.download_data_from_studio(request, output_path)