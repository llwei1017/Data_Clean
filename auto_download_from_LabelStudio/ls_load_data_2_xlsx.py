from clean_data import DataCleaner
from label_studio_control import DataLoad
import torch

def get_data_2_xlsx_from_ls(request_list, output_path, llama_path): # in:请求下载列表(项目名称)；out:输出xlsx路径目录
    data_load = DataLoad()
    data_clean = DataCleaner()
    xlsx_outpiut_path = data_load.download_data_from_studio(request_list, output_path)
    data_clean.batch_clean_data_4_training(xlsx_outpiut_path, xlsx_outpiut_path, llama_path)

def json_data_2_xlsx(input_dir_path, output_dir_path): # in:包含本地下载的json文件目录；out:输出xlsx的路径目录
    data_clean = DataCleaner()
    data_clean.batch_clean_data_4_training(input_dir_path, output_dir_path)



if __name__ == "__main__":
    # request_list = ["平安e生保_5_问题提取"]   # 请求列表
    torch.cuda.set_device(7)
    # 本地的LLaMA Factory地址，需要修改
    data_clean = DataCleaner()
    # request_list = ['13：46','@7 73','1069395595301']
    request_list = ["中意一生_问题提取","金禧一生_问题提取"]
    LLAMA_FACTORY_DIR_PATH=r"/data/disk4/home/weilili/LLaMA-Factory"
    # 结果输出目录，需要修改
    output_dir_path = r"/data/disk4/home/weilili/ai-project/data_cleaning/extract_questions/output"   
    get_data_2_xlsx_from_ls(request_list, output_dir_path, LLAMA_FACTORY_DIR_PATH)    # 自动化下载
    # result = data_clean.entity_name_from_qlist(request_list, LLAMA_FACTORY_DIR_PATH, output_dir_path)
    # input_dir_path = r"/data/disk4/home/zihan/data_cleaning/extract_questions/source_data/output/test"    # 输入json目录
    # json_data_2_xlsx(input_dir_path, output_dir_path)   # 本地下载
    # print(result)