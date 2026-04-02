import getpass
import json
import os.path

def load_key(keyname: str) -> object:
    file_name = "Keys.json"
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            Key = json.load(file)
        if keyname in Key and Key[keyname]:
            return Key[keyname]
        else:
            keyval = getpass.getpass("配置文件无相应就，请输入配置信息").strip()
            Key[keyname] = keyval
            with open(file_name, "w") as file:
                json.dump(Key,file,indent=4)
            return keyval
    else:
        keyval = getpass.getpass("配置文件无相应就，请输入配置信息").strip()
        Key = {
            keyname:keyval
        }
        with open(file_name, "w") as file:
                json.dump(Key,file,indent=4)
        return keyval
    
if __name__ == "__main__":
     print(load_key("API_KEY"))