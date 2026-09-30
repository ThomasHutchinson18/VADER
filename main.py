from vader.load_data import VaderData
import logging
logging.basicConfig(level=logging.INFO)

if __name__=="__main__":
    logger = logging.getLogger(__name__)
    vader_data = VaderData(start="2002-01-01")
    vader = vader_data.vader
    vader.to_csv('vader.csv')