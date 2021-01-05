'''
@Time : 2020/8/13 15:02
@Author : xzw
@File : DBConn.py
@Desc : 用于连接数据库的类
'''

import cx_Oracle
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class OracleConn(object):
    '''
    连接Oracle数据库获取数据
    '''

    def __init__(self, host, port, username, password, service_name):
        '''
        初始化方法
        :param host: 地址
        :param port: 端口号
        :param username: 用户名
        :param password: 密码
        :param service_name: service名称
        '''
        self.conn = cx_Oracle.connect(username, password, host + ':' + port + '/' + service_name)
        self.cur = self.conn.cursor()

    def get_data(self, sql):
        '''
        获取数据
        :param sql: SQL
        :return: 返回数据集
        '''
        self.cur.execute(sql)
        result = self.cur.fetchall()

        return result

    def close_conn(self):
        '''
        关闭连接
        :return:
        '''
        self.cur.close()
        self.conn.close()

