"""
KOSPI200 구성종목 코드 조회.
quant_cnn_chart/src/data/kospi200_universe.py 로직을 그대로 이식 (자기완결적 레포 유지).

⚠️ pykrx 인자 순서 주의: get_index_portfolio_deposit_file()은 (ticker, date) 순서 --
ticker(지수 코드)가 먼저, date는 선택적으로 뒤. 순서를 뒤집으면 pykrx가 지수 코드를
날짜로 오인해서 엉뚱한 에러(또는 빈 리스트)를 반환함.

설치:
    pip install pykrx python-dotenv

사전 준비:
    레포 루트에 .env 파일 (KRX_ID/KRX_PW) -- quant_xgboost와 동일.
    pykrx가 2025년 12월 KRX 회원제 전환 이후 로그인 방식으로 바뀌었으므로,
    load_dotenv()는 반드시 pykrx import보다 먼저 실행해야 함.
"""

from dotenv import load_dotenv

load_dotenv()
from pykrx import stock

KOSPI200_INDEX_CODE = "1028"  # KRX 지수 코드 (코스피 200)


def get_kospi200_tickers(date: str = None) -> list[str]:
    """
    KOSPI200 구성종목의 6자리 코드 리스트를 반환.
    date: YYYYMMDD 형식. None이면 pykrx가 내부적으로 최근 영업일을 사용.
    """
    if date is None:
        tickers = stock.get_index_portfolio_deposit_file(KOSPI200_INDEX_CODE)
    else:
        tickers = stock.get_index_portfolio_deposit_file(KOSPI200_INDEX_CODE, date)

    if not tickers:
        raise RuntimeError(f"KOSPI200 구성종목 조회 실패 (빈 리스트 반환, date={date})")

    print(f"KOSPI200 구성종목 {len(tickers)}개 조회 완료" + (f" (기준일: {date})" if date else ""))
    return tickers


if __name__ == "__main__":
    tickers = get_kospi200_tickers()
    print(tickers[:10], "...")