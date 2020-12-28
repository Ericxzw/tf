'''
@Time : 2020/12/9 10:13
@Author : xzw
@File : get_data.py
@Desc : 
'''
from seq2seq.utils.DBConn import OracleConn
from seq2seq.sql import wt_df
from pandas import DataFrame
import numpy as np

def save_data(file_name, data):
    '''
    保存数据
    :param file_name: 保存的文件名
    :param data: 需要保存的数据
    :return:
    '''
    f = open(file_name, "w", encoding='utf-8')
    for d in range(len(data)):
        tmp = str(data[d]).replace('\n', '').replace('\t', '').replace(' ', '').strip()
        f.write(tmp + '\n')
    f.close()

if __name__ == '__main__':
    __host, __port, __user, __password, __service_name = 'ip_host', 'port', 'username', 'password', 'service_name'

    # 1、建立连接
    conn = OracleConn(host=__host, port=__port, username=__user, password=__password, service_name=__service_name)

    # 2、获取数据
    data = conn.get_data(sql=wt_df)
    data = DataFrame(data, columns=['zxnr', 'dfnr'])
    zxnr = list(data['zxnr'])
    dfnr = list(data['dfnr'])
    print(len(zxnr), len(dfnr))

    # 3、保存数据
    save_data('./samples/question', zxnr)
    save_data('./samples/answer', dfnr)

    # 4、关闭资源
    conn.close_conn()
