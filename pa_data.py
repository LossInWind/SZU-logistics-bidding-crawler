import requests
import chardet
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import re


# 递归遍历节点，获取所有文本
def get_all_text(node):
    # 如果当前节点是文本节点，直接返回文本
    if isinstance(node, str):
        return node.strip()
    else:
        # 否则遍历子节点
        text = ''
        for child in node.children:
            text += get_all_text(child)
        return text.strip()


# 创建一个空数据帧
df = pd.DataFrame(columns=['i', 'url', 'result', 'time', 'money'])

# 遍历 id
for i in range(1, 7450):
    # 构造 URL
    url = f'https://www1.szu.edu.cn/bids/view.asp?id={i}'

    # 发送 HTTP 请求
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(url, headers=headers)

    # 使用 BeautifulSoup 解析 HTML
    encoding = chardet.detect(response.content)['encoding']
    try:
        soup = BeautifulSoup(response.content.decode(encoding, errors='ignore'), 'html.parser')
    except UnicodeDecodeError:
        continue

    # 获取所有文本
    text = get_all_text(soup)

    # 输出文本
    print(i)
    print(f'URL: {url}')
    # print(text)
    print("!!!!!!!!!!!!!!!!!!!!!!")

    # 查找文本中的关键词
    if '[返回列表]' in text:
        start = text.index('[返回列表]') + len('[返回列表]')
        end = text.find('深大新闻网')
        result = text[start:end]
        print(result)
        print("!!!!!!!!!!!!!!!!!!!!!!")
        # 获取时间
        if '（本文更新于' in result:
            start = result.index('（本文更新于') + len('（本文更新于')
            end = start
            while end < len(result) and result[end] != '）':
                end += 1
            time_str = result[start:end]
            print(time_str)

            # 获取金额或预算
            budget = None
            nums = []

            if "成交价格：" in result:
                start = result.index("成交价格：") + len("成交价格：")
                c = start
                while c < len(result) and (
                        result[c].isdigit() or result[c] == '.' or result[c] == '万'):
                    c += 1
                end = c
                budget_str = result[start:end]
                if budget_str.endswith('万'):
                    num = float(budget_str[:-1]) * 10000
                elif budget_str.isdigit() or '.' in budget_str:
                    num = float(budget_str)
                else:
                    num = None
                if num is not None:
                    nums.append(num)
            else:
                keywords = ['￥', '经费为', '最高限价', '采购预算', "合计：", "总价：", "总金额：", "总价为", "总金额为",
                            "总价为人民币", "总金额为人民币", "总价人民币", "总金额人民币","中标金额为","成交价格：¥","成交金额为","价格：¥","成交金额为","成交金额：¥","成交金额：","费用上限为"]
                for keyword in keywords:
                    if keyword in result:
                        start = result.index(keyword) + len(keyword)
                        c = start
                        while c < len(result) and (
                                result[c].isdigit() or result[c] == '.' or result[c] == '万'):
                            c += 1
                        end = c
                        budget_str = result[start:end]
                        if budget_str.endswith('万'):
                            num = float(budget_str[:-1]) * 10000
                        elif budget_str.isdigit() or '.' in budget_str:
                            num = float(budget_str)
                        else:
                            num = None
                        if num is not None:
                            nums.append(num)

            if nums:
                budget = min(nums)
                print(budget)


        # 将值添加到数据帧中
        df = df.append({'i': i, 'url': url, 'result': result, 'time': time_str, 'money': budget}, ignore_index=True)
    else:
        print('[返回列表] not found')

    print("====================================")

# 将数据帧写入Excel文件
df.to_excel('output_1.xlsx', index=False)
