from tqdm import tqdm
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
from threading import Thread
from keybert import KeyBERT
from konlpy.tag import Okt
from datetime import date
from queue import Queue
import pandas as pd
import numpy as np
import requests
import pymysql
import time
import ast
import re


# -- Extract Functions --#


def loadLibBook(libCode: int, startDate: str) -> pd.DataFrame:
    """

    도서관 정보나루 API에서 도서관의 장서 데이터를 불러온다.
    해당 함수는 extractAllLibBooksMultiThread에서 사용된다.

    """
    libname = {
        111003: "강남",
        111004: "강동",
        111005: "강서",
        111006: "개포",
        111007: "고덕",
        111008: "고척",
        111009: "구로",
        111010: "남산",
        111022: "노원",
        111011: "도봉",
        111012: "동대문",
        111013: "동작",
        111014: "마포",
        111016: "서대문",
        111030: "송파",
        111015: "양천",
        111018: "영등포",
        111019: "용산",
        111020: "정독",
        111021: "종로",
    }
    from datetime import timedelta, datetime

    day = datetime.strptime(startDate, "%Y-%m-%d").date()

    # 해당 달의 첫째날 구하기 => 7월
    first_day = day.replace(day=1)

    # 전달의 마지막 날 구하기 => 6월 30일
    dayLast = first_day - timedelta(days=1)

    # 전달의 첫째 날 구하기 => 6월 1일
    dayStart = dayLast.replace(day=1)
    # print(dayStart, " - ", dayLast)

    libUrl = f"http://data4library.kr/api/itemSrch?libCode={libCode}&startDt={dayStart}&endDt={dayLast}&authKey=7123eacb2744a02faca2508a82304c3bf154bf0b285da35c2faa2b8498b09872&pageSize=10000"
    libHtml = requests.get(libUrl)
    libSoup = BeautifulSoup(libHtml.content, features="xml")

    libBookname = list(map(lambda x: x.string, libSoup.find_all("bookname")))
    libAuthor = list(map(lambda x: x.string, libSoup.find_all("authors")))
    libPublisher = list(map(lambda x: x.string, libSoup.find_all("publisher")))
    libISBN = list(map(lambda x: x.string, libSoup.find_all("isbn13")))
    libClassNum = list(map(lambda x: x.string, libSoup.find_all("class_no")))
    libRegDate = list(map(lambda x: x.string, libSoup.find_all("reg_date")))
    libBookImageURL = list(map(lambda x: x.string, libSoup.find_all("bookImageURL")))

    data = pd.DataFrame(
        [
            libBookname,
            libAuthor,
            libPublisher,
            libISBN,
            libClassNum,
            libRegDate,
            libBookImageURL,
        ]
    ).T
    data.columns = ["도서명", "저자", "출판사", "ISBN", "주제분류번호", "등록일자", "이미지주소"]
    sortedData: pd.DataFrame = data[~data["주제분류번호"].isna()].copy()
    sortedData["지역"] = libname[libCode]
    return sortedData


##최신버전
def extractAllLibBooksMultiThread(startDate: str, thread_num: int = 10) -> pd.DataFrame:
    """

    loadLibBook를 multithread로 구현해서 Extract 속도를 향상시켰다.
    queue 와 thread library를 사용했다.

    """

    q = Queue()
    dataframes = []
    seoulLibCode = [
        111003,
        111004,
        111005,
        111006,
        111007,
        111008,
        111009,
        111010,
        111022,
        111011,
        111012,
        111013,
        111014,
        111016,
        111030,
        111015,
        111018,
        111019,
        111020,
        111021,
    ]

    for code in seoulLibCode:
        q.put(code)

    def process():
        while True:
            code = q.get()
            item = loadLibBook(code, startDate)
            dataframes.append(item)
            q.task_done()
            print(f"{code} 완료 | {len(item)} 개 추출")

    for _ in range(thread_num):
        worker = Thread(target=process)
        worker.daemon = True
        worker.start()

    q.join()

    # result = pd.concat(dataframes).drop_duplicates(subset="ISBN")
    result = pd.concat(dataframes)
    val = result["주제분류번호"].astype(str)
    BM = val.str.contains("004.") | val.str.contains("005.")
    result: pd.DataFrame = result[BM].reset_index(drop=True)
    return result


def extractKyobo(ISBN: int) -> list:
    """

    새롭게 확보한 도서의 데이터를 교보문고에서 크롤링한다.
    목차, 도서소개, 추천사 등 가용한 텍스트 모두를 수집한다.
    해당 함수는 kyoboSaveMultiThread에서 사용된다.

    """
    kyoboUrl = f"http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode={ISBN}"
    kyoboHtml = requests.get(kyoboUrl)
    kyoboSoup = BeautifulSoup(kyoboHtml.content, "html.parser")

    try:
        """
        book_title, book_toc, book_intro, publisher 추출하기

        """
        book_title: str = kyoboSoup.find(class_="prod_title").string

        # 도서 toc 추출, <br/>을 가지고 목차를 구분하므로 split 사용
        book_toc: str = kyoboSoup.find(class_="book_contents_item")
        book_toc = str(book_toc).split("<br/>")[1:-1]

        # 도서 소개 추출, <br/>을 가지고 목차를 구분하므로 split 사용, tag제거
        book_intro: str = kyoboSoup.find_all("div", "info_text")[-1]
        book_intro_list = str(book_intro).split("<br/>")

        book_intro = []
        for i in book_intro_list:
            book_intro += re.sub("<.*>", "", i).split(".")

        publisher: str = kyoboSoup.find(class_="book_publish_review")

        if publisher is not None:
            publisher = publisher.p.get_text()
            publisher = publisher.split(".")
        else:
            publisher = []

        itemList = [book_title]

        for chars in [book_toc, book_intro, publisher]:
            # ''제거
            chars = list(filter(None, chars))

            # 한국어 영어 제외하고 모두 제거
            chars = [
                re.sub("[^A-Za-z\u3130-\u318F\uAC00-\uD7A3]", " ", i) for i in chars
            ]

            # 공백 하나로 줄이기 ex) '  ' -> ' '
            chars = [re.sub(" +", " ", i).strip(" ") for i in chars]

            itemList.append(chars)

    except:
        itemList = ["skip", ISBN]
        print("Skip :", ISBN)

    return itemList


# 최신버전
def extractKyoboMultiThread(ISBNs: list, thread_num: int = 6) -> list:
    """

    kyobosave를 multithread로 구현해서 Extract 속도를 향상시켰다.
    수집 된 리스트는 형태소분석을 거쳐 w2v학습과 keyBert 키워드 추출에 활용된다.

    """
    result: list = []

    q = Queue()
    for ISBN in ISBNs:
        q.put(ISBN)

    def process():
        for num in range(9999):
            ISBN = q.get()
            item = extractKyobo(ISBN)
            result.append(item)
            q.task_done()
            if (num + 1) % 5 == 0:
                time.sleep(0.5)
            if num % 10 == 0:
                print(f"{num} 번째")

    for _ in range(thread_num):
        worker = Thread(target=process)
        worker.daemon = True
        worker.start()

    q.join()
    print("kyoboBooks 추출완료")

    return result


def changeStringToList(strList):
    """

    dataframe안에 list를 통으로 넣으면 str으로 저장된다.
    ast 라이브러리를 쓰면 원래 ㅣist 이지만 str 타입으로 표현된 값을 다시 list 타입으로 바꿔준다.

    """
    return ast.literal_eval(strList)


def saveText(newdf):
    """

    교보문고에서 추출한 신규 도서의 text를 rawBookInfo.csv에 저장한다.
    도서별로 추출되는 text 개수가 다르다보니 도서별 column 개수 차이가 크다.
    도서별 추출 된 column 개수가 db에 설정한 column 개수보다 많아질 수 있으므로 오류 방지차 csv로 저장한다.

    """

    # -- transform and change column name --#
    a = list(map(changeStringToList, newdf["keywords"]))
    newBookInfo = pd.DataFrame(a)
    newBookInfo.columns = [f"col{i}" for i in range(len(newBookInfo.columns))]

    # -- concat with old one and save --#

    # load a previous file (혹시 파일 잘못된 경우 bookinfo12.csv로 백업 해놨음)
    bookInfo = pd.read_csv("backend/assets/dodomoa/rawBookInfo.csv", index_col=0)

    # concat new items with previous items
    concatNewBookInfo = pd.concat([bookInfo, newBookInfo])

    # -- drop duplicates by ISBN --#
    concatNewBookInfo["col1"] = concatNewBookInfo["col1"].astype(int)
    concatNewBookInfo = concatNewBookInfo.drop_duplicates(subset="col1")

    # -- save --#
    concatNewBookInfo.to_csv("backend/assets/dodomoa/rawBookInfo.csv")

    return None


def extract(date: str) -> pd.DataFrame:
    """

    이 함수를 실행하면 extract 과정이 진행된다.
    도서 정보를 수집한 다음 db 저장된 도서리스트와 비교한 뒤 db에 없는 새로운 도서만 추출한다.
    새롭게 추출된 도서 정보를 크롤링 한다.

    """
    rawbookinfo = extractAllLibBooksMultiThread(date)
    df = rawbookinfo.drop_duplicates(subset="ISBN")

    # -- load bookinfo in DB --#
    cursor.execute("SELECT ISBN FROM backend_dodomoabookinfo")
    result = cursor.fetchall()
    libinfo = pd.DataFrame(result)

    # -- compare extracted book info with previous book info and get new books   --#

    # extract ISBNs of new books and compare them with previous books
    ISBNs = df["ISBN"].tolist()
    BM = np.in1d(ISBNs, libinfo["ISBN"])

    # extract new book's info by crowaling kyobobooks site
    ISBNs = df[~BM]["ISBN"]
    docs = extractKyoboMultiThread(ISBNs, thread_num=10)

    # merge the extracted texts with book info
    orderlist = [i[1] for i in docs]
    k = pd.DataFrame([orderlist, docs]).T
    k.columns = "ISBN", "keywords"

    # make them as a dataframe
    newdf = df[~BM]
    newdf = newdf.merge(k, left_on="ISBN", right_on="ISBN")

    # save texts of newdf as a csv format
    saveText(newdf)  # return None

    return rawbookinfo, newdf


# -- Transform Functions --#


def findAlphabet(text: str) -> str:
    """
    - 알파벳을 추출한다.
    """
    # han = re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', text)
    alpha = re.findall("[a-zA-Z]+", text)
    return alpha


def removeStopwords(text: list, stopwords: list) -> str:
    """
    - stopwords를 제거한다.
    """

    word = [word for word in text if word not in stopwords]
    result = " ".join(word)
    return result


def extractKeywords(doc: str, stopwords: list, keyBertModel) -> list:
    """

    keyBert로 keyWord를 추출하는 절차
    사전 학습된 모델을 사용하기 때문에 개별 도서의 형태소를 넣으면 키워드가 추출된다.
    1. okt로 형태소 분석
    2. stopwords 제거
    3. okt에서 추출한 영문 키워드는 keyBert를 거치지 않고 저장된다.
       CS 용어가 모두 영문이다보니 도서의 키워드를 파악할 때 영문이 중요하다.
       사용자가 영문으로 검색할 때 대비용이기도 하다.
    4. 20개 한글 키워드가 추출되며 영문 키워드 개수에 따라 도서 별 총 키워드 개수는 달라진다.

    """
    doc: str = (
        re.sub("\d[.]|\d|\W|[_]", " ", doc)
        .replace("머신 러닝", "머신러닝")
        .replace("인공 지능", "인공지능")
    )
    text = list(filter(None, doc.split(" ")))
    removedoc: str = removeStopwords(text, stopwords)

    # 문서 정보 추출
    Okt = Okt()
    hanNouns: list = Okt.nouns(removedoc)
    words: str = " ".join(hanNouns)

    docResult: list = keyBertModel.extract_keywords(
        words, top_n=10, keyphrase_ngram_range=(3, 3), use_mmr=True, diversity=0.1
    )
    result = list(map(lambda x: x[0], docResult))

    items = []
    for item in result:
        items.extend(item.split(" "))

    # remove Stopwords
    items: str = removeStopwords(items, stopwords)
    hanNouns: str = removeStopwords(hanNouns, stopwords)

    bertInfo = pd.DataFrame(items.split(" "))
    keyWordInfo = pd.DataFrame(hanNouns.split(" "))

    keyWords = (
        pd.concat([bertInfo, keyWordInfo], axis=0)
        .groupby(by=0)
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    keyWords = list(filter(lambda a: a if len(a) > 1 else None, keyWords))

    engList = (
        pd.DataFrame(findAlphabet(removedoc))
        .value_counts()
        .sort_values(ascending=False)[:20]
        .index.tolist()
    )

    engList = list(map(lambda x: x[0], engList))

    result: list = keyWords[:20]
    result.extend(engList)
    return result


def transformkeyBert(newdf: pd.DataFrame, keyBertModel) -> pd.DataFrame:
    """

    도서별로 키워드를 추출한다.
    extractKeywords를 활용한다.

    """

    stopwords = pd.read_csv("backend/assets/dodomoa/englist.csv").T.values.tolist()[0]

    # keywords before extracting
    docs = newdf["keywords"]

    # extract keywords
    keywords = []
    for doc in docs:
        doc = "".join(doc)
        val: list = extractKeywords(doc, stopwords=stopwords, keyBertModel=keyBertModel)
        keywords.append(val)

    # bookinfo without keywords
    df = newdf.drop(columns=["keywords"])

    # merge result to df
    df["keywords"] = keywords

    return df


# change rows to lists and join them in a string
def mergeListToString(item: pd.Series):
    """

    list를 하나의 str로 바꾼다.
    Okt 추출에 알맞은 형태로 바꾸기 위해서 사용한다.

    """
    wordList = item.astype(str).tolist()
    str_list = [re.sub("\d", "", str(a)) for a in wordList]
    str_list = list(filter(None, str_list))
    result = " ".join(str_list)
    return result


def trainW2V():
    """

    extract 단계에서 저장한 csv를 활용해 word2vec을 학습한다.
    형태소 분석 단계를 거쳐 학습한 뒤 w2v로 저장된다.

    """
    # load items to make same condtion of rows
    bookinfo = pd.read_csv("backend/assets/dodomoa/rawBookInfo.csv", index_col=0)

    # load corpus analyzer
    okt = Okt()

    # iterate all rows of bookinfo
    wordsList = [mergeListToString(i[1]) for i in bookinfo.iterrows()]
    print("Complete wordsList Load!!")

    # analyze corpus
    print("konlpy 실행 중... 평균 9분 소요")
    konlpyWords = list(map(lambda x: okt.nouns(x), wordsList))
    print("Complete konlpyWords")

    # train Word2vec
    embedding_model = Word2Vec(
        sentences=konlpyWords, window=2, min_count=50, workers=7, sg=1
    )

    # save
    embedding_model.wv.save_word2vec_format("w2v")

    return None


# -- Load Function --#


def load(rawbookinfo, newdf, dfWithKeywords):
    """

    rawbookinfo = 정보나루 API에서 추출한 도서관 별 도서 데이터
    newdf = 기존 db에 없는 새로운 도서 목록
    dfwithkeywords = 새로운 도서 목록의 keyword
    rawbookinfo, newdf, dfwithkeywords는 각각 table에 저장된다.

    """

    from sqlalchemy import create_engine

    db_connection_str = "mysql+pymysql://root@localhost:3306/dash"
    db_connection = create_engine(db_connection_str)

    ###### backend_dodomoalibinfo
    """

    backend_dodomoalibinfo 컬럼 = ISBN, 지역
    사용자가 선택한 도서관의 도서목록을 추출할 때 사용한다.

    """

    # select ISBN and 지역 column of rawbookinfo
    booklist = rawbookinfo[["ISBN", "지역"]]

    # save to backend_dodomoalibinfo table
    booklist.to_sql(
        name="backend_dodomoalibinfo",
        con=db_connection,
        if_exists="append",
        index=False,
    )

    ###### backend_dodomoabookinfo
    """

    backend_dodomoabookinfo 컬럼 = `도서명`,`저자`,`출판사`,`ISBN`,`주제분류번호`,`등록일자`,`이미지주소`

    """
    backend_dodomoabookinfo = newdf[
        ["도서명", "저자", "출판사", "ISBN", "주제분류번호", "등록일자", "이미지주소"]
    ]

    backend_dodomoabookinfo["출판사"] = backend_dodomoabookinfo["출판사"].fillna("-")
    backend_dodomoabookinfo["주제분류번호"] = "00" + backend_dodomoabookinfo["주제분류번호"].astype(
        str
    )

    # save to backend_dodomoabookinfo table
    backend_dodomoabookinfo.to_sql(
        name="backend_dodomoabookinfo",
        con=db_connection,
        if_exists="append",
        index=False,
    )

    ###### backend_dodomoakeyword2
    """

    backend_dodomoakeyword2 컬럼 = `ISBN`,`keywords`
    사용자 검색 키워드와 도서 키워드를 비교하기 위해 keywords column에 fulltext index 설정

    """

    backend_dodomoakeyword2 = dfWithKeywords[["ISBN", "keywords"]]
    backend_dodomoakeyword2["keywords"] = list(
        map(lambda x: " ".join(x), backend_dodomoakeyword2["keywords"])
    )

    backend_dodomoakeyword2.columns = ["ISBN", "keyword"]

    # save to backend_dodomoakeyword2 table
    backend_dodomoakeyword2.to_sql(
        name="backend_dodomoakeyword2",
        con=db_connection,
        if_exists="append",
        index=False,
    )
    return None


if __name__ == "__main__":

    # load the model
    keyBertModel = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")

    # connect to mysql
    conn = pymysql.connect(
        host="localhost", port=int(3306), user="root", passwd="", db="dash_test"
    )
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # Extract
    date = date.today().strftime("%Y-%m-%d")
    rawbookinfo, newdf = extract(date)

    # transform
    dfWithKeywords = transformkeyBert(newdf, keyBertModel)
    trainW2V()  # no return

    # Load
    load(rawbookinfo, newdf, dfWithKeywords)  # no return
