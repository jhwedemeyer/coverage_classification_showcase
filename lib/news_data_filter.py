import pandas as pd

class NewsFilter:
    
    config = dict({
        "welt":{
            "upper_date_limit": "2021-12-31",
            "lower_date_limit": "1995-05-16",
        },
        "spiegel":{
            "upper_date_limit": "2021-12-31",
            "lower_date_limit": "1956-01-01",
        },
        "tagesspiegel":{
            "upper_date_limit": "2021-12-31",
            "lower_date_limit": "1997-01-01",
        },
        "zeit":{
            "upper_date_limit": "2021-12-31",
            "lower_date_limit": "1946-01-01",
        },
        "berlinermorgenpost":{
            "upper_date_limit": "2021-12-31",
            "lower_date_limit": "2002-07-01",
        },
        "hamburgerabendblatt":{
            "upper_date_limit": "2021-12-31",
            "lower_date_limit": "1949-01-01",
        },
        "focus":{
            "upper_date_limit": "2021-12-31",
            "lower_date_limit": "2002-01-01",
        },
        "zeit_offline":{
            "upper_date_limit": "2021-12-31",
            "lower_date_limit": "1948-01-01",
        },
        "default":{
            "upper_date_limit": "2021-12-31",
            "lower_date_limit": "2000-01-01",
        }
    })
    
    @staticmethod
    def apply(data):

        # work on a copy, the filters below assign to their input
        data = data.copy()

        # apply general filters
        data = GeneralFilter.apply(data)

        # apply journal specific filters; groups are iterated so that
        # JournalFilter.apply() still sees the 'journal' column
        groups = [JournalFilter.apply(group.copy()) for _, group in data.groupby("journal")]
        data = pd.concat(groups).reset_index(drop=True) if groups else data

        return data

"""
Class of filters which are applied on all data independent of the journal
"""
class GeneralFilter:

    # publication time zone of the German newspapers in the corpus
    TIMEZONE = "Europe/Berlin"

    @staticmethod
    def convert_dates(data):
        # The stored dates mix several ISO formats ('%Y-%m-%d', with seconds and
        # with microseconds). A small number of records carry a UTC offset;
        # these are converted to local time and only then made naive, so that
        # round_to_day() below assigns every article to its local publication
        # day. Records without an offset are already local times.
        raw = data["date"].astype("string")
        has_offset = raw.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True, na=False)

        dates = pd.to_datetime(raw.where(~has_offset), format="ISO8601")
        if has_offset.any():
            localised = pd.to_datetime(raw.where(has_offset), format="ISO8601", utc=True)
            localised = localised.dt.tz_convert(GeneralFilter.TIMEZONE).dt.tz_localize(None)
            dates = dates.where(~has_offset, localised)
            print("INFO: Converting " + str(int(has_offset.sum())) +
                  " dates with a UTC offset to " + GeneralFilter.TIMEZONE + ".")

        data["date"] = dates
        print("INFO: Converting dates to format 'datetime'.")
        return data

    @staticmethod
    def round_to_day(data):
        data.date = data.date.dt.floor(freq="D")
        print("INFO: Rounding dates to frequency 'day'.")
        return data
    
    @staticmethod
    def remove_non_text_articles(data):
        non_text_articles = data[(data['text'] == "") | (data['text'] == " ") | (data['text'] == "NULL")]
        if len(non_text_articles) > 0:
            data = data[(data['text'] != "") & (data['text'] != " ") & (data['text'] != "NULL")]
            print("INFO: Ignoring " + str(len(non_text_articles)) + " articles without content.")
        return data
            
    @staticmethod
    def apply(data):
        data = GeneralFilter.convert_dates(data)
        data = GeneralFilter.round_to_day(data)
        return data
                       
"""
Classes of filters specific for each journal
"""         
class WeltFilter:
    
    @staticmethod
    def remove_newsticker(data):
        data["category"] = data.link.apply(lambda l: l.split("/")[3])
        data = data[data.category != "newsticker"]
        data.drop(columns=["category"],inplace=True)
        return data
    
    @staticmethod
    def apply(data):
        data = WeltFilter.remove_newsticker(data)
        return data
    
class SpiegelFilter:
    
    @staticmethod
    def remove_special_editions(data):
        data = data[data.date != "2021-09-27"]
        data = data[data.date != "2020-10-26"]
        data = data[data.date != "2019-09-24"]
        return data
    
    @staticmethod
    def apply(data):
        data = SpiegelFilter.remove_special_editions(data)
        return data
    
class ZeitFilter:
    
    @staticmethod
    def remove_incomplete_data(data):
        data = data[(data.date < "2006-01-01") | (data.date > "2008-12-31")]
        data = data[data.date != "1958-01-02"]
        data = data[data.date != "1971-05-14"]
        data = data[data.date != "2020-12-08"]       
        return data
    
    @staticmethod
    def remove_special_editions(data):
        data = data[data.date != "2016-02-15"]
        data = data[data.date != "2017-12-04"]
        data = data[data.date != "2018-12-03"]
        data = data[data.date != "2019-12-03"]
        return data
    
    
    @staticmethod
    def apply(data):
        data = ZeitFilter.remove_incomplete_data(data)
        data = ZeitFilter.remove_special_editions(data)
        return data  
    
class BerlinerMorgenpostFilter:
    
    @staticmethod
    def remove_incomplete_data(data):
        data = data[data.date > "2002-06-01"]
        return data
    
    @staticmethod
    def apply(data):
        data = BerlinerMorgenpostFilter.remove_incomplete_data(data)
        return data  

class HamburgerAbendblattFilter:
    
    @staticmethod
    def remove_incomplete_data(data):
        data = data[data.date.dt.year != 1993]       
        return data
    
    @staticmethod
    def apply(data):
        data = HamburgerAbendblattFilter.remove_incomplete_data(data)
        return data  
    
class TagesspiegelFilter:
    
    @staticmethod
    def remove_incomplete_data(data):
        data = data[(data.date >= "1999-08-01")]
        data = data[(data.date < "2002-02-01") | (data.date > "2002-04-30")]
        return data
    
    @staticmethod
    def apply(data):
        data = TagesspiegelFilter.remove_incomplete_data(data)
        return data    

class HandelsblattFilter:

    @staticmethod
    def remove_incomplete_data(data):
        data = data[(data.date < "2007-10-08") | (data.date > "2007-10-14")]
        return data

    @staticmethod
    def apply(data):
        data = HandelsblattFilter.remove_incomplete_data(data)
        return data

"""
Class of filters which are applied on all data dependent on the journal
"""                 
class JournalFilter:
         
    @staticmethod
    def set_upper_date_limit(data, upper_limit):
        data = data[data.date <= upper_limit]
        return data
    
    @staticmethod
    def set_lower_date_limit(data, lower_limit):
        data = data[data.date >= lower_limit]
        return data
                 
    @staticmethod
    def apply(data):
        journal = data.journal.unique()
        if len(journal) != 1:
            raise ValueError("Can't apply journal filter on mixed data sets.")
        journal = journal[0]
        
        config = NewsFilter.config.get(journal)
        if config is None:
            config = NewsFilter.config.get("default")
        
        if journal == "welt":
            data = WeltFilter.apply(data)
            
        if journal == "spiegel":
            data = SpiegelFilter.apply(data)
            
        if journal == "zeit":
            data = ZeitFilter.apply(data)
            
        if journal == "tagesspiegel":
            data = TagesspiegelFilter.apply(data)
        
        if journal == "berlinermorgenpost":
            data = BerlinerMorgenpostFilter.apply(data)
            
        if journal == "hamburgerabendblatt":
            data = HamburgerAbendblattFilter.apply(data)

        if journal == "handelsblatt":
            data = HandelsblattFilter.apply(data)
            
            
        return data
    
    
class ExampleFilter:
    
    @staticmethod
    def apply(data):
        return data
