import json
import re
from pathlib import Path

data_path = "./data/chinese-poetry/唐诗"
max_length = 128

def sentenceParse(para):
    para = re.sub(r"（.*?）", "", para)
    para = re.sub(r"\(.*?\)", "", para)
    para = re.sub(r"{.*?}", "", para)
    para = re.sub(r"《.*?》", "", para)
    para = re.sub(r"[\[\]]", "", para)
    para = "".join([s for s in para if s not in "0123456789-"])
    para = re.sub(r"。。", "。", para)
    para = re.sub(r"？。", "。", para)
    para = re.sub(r"？", "。", para)
    if len(para) < 24 or len(para) > max_length:
        return ""
    return para


def parseRawData(author=None, constrain=None):
    def handleJson(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        rst = []
        for poetry in data:
            if author and poetry.get("author") != author:
                continue

            paragraphs = poetry.get("content")

            if any(
                len(tr) != constrain and len(tr) != 0
                for s in paragraphs
                for tr in re.split("[，！。]", s)
                if constrain is not None
            ):
                continue

            pdata = "".join(paragraphs)
            pdata = sentenceParse(pdata)
            segments = [segment for segment in re.split(r"[，。]", pdata) if segment]

            # 仅训练五言绝句和七言律诗
            if any(len(segment) not in [5] for segment in segments):
                continue
            # 去除含有错误字符的诗句
            if "□" in pdata:
                continue
            if pdata:
                print(pdata)
                rst.append(pdata)
        return rst

    poems = []
    src_path = Path(data_path)
    for file_path in src_path.glob("data3*"):
        poems.extend(handleJson(file_path))
    return poems


poems = (
    parseRawData(author="李白")
    + parseRawData(author="李商隐")
    + parseRawData(author="刘禹锡")
)
poems = set(poems)
poems = list(poems)


with open("poems.json", "w", encoding="utf-8") as f:
    json.dump(list(poems), f, ensure_ascii=False)

# 构建词汇表
word_to_index = {}
for poem in poems:
    for word in poem:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
word_to_index["<EOP>"] = len(word_to_index)
word_to_index["<START>"] = len(word_to_index)

# 保存词汇表到JSON文件
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(word_to_index, f, ensure_ascii=False)
