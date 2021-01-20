import json
import os

def read_json_model_info(file_path):
    if os.path.exists(file_path):
        f=open(file_path,'r')
        content=f.read()
        model_info=json.loads(content)
        f.close()
    else:
        model_info={
            'best_MAE':1000,
            'best_epoch':0,
            'checkpoint_MAE':1000,
            'checkpoint_epoch':0
        }
    return model_info

def write_json_model_info(file_path,model_info):
    f=open(file_path,'w+')
    content=json.dumps(model_info)
    f.write(content)
    f.close()