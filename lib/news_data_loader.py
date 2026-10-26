import sqlite3
import pandas as pd
import datetime
import hashlib
import matplotlib.pyplot as plt
from news_data_filter import NewsFilter


class NewsData:
    NORM_PAGE_LENGTH = 1800
    GENERAL_BAG_OF_WORDS = "der OR die OR das OR ein OR eine OR und"
        
    def __init__(self, database):
        self.database = database
    
    def load(self, columns = ["title", "date", "journal", "link", "meta", "text"], constraints = None, verbose=False, round_to_day=True):
        SQL_QUERY = "SELECT "+ ', '.join(columns) +" FROM news"
        if constraints is not None:
            SQL_QUERY += " WHERE " + constraints

        if verbose: print("Load data with querry: " + SQL_QUERY)
        allData= self.query(SQL_QUERY)

        # convert to pandas DataFrame
        allData_df = pd.DataFrame(allData,columns=columns)
        del allData
       
        allData_df = NewsFilter.apply(allData_df)
        
        if verbose:
            print("Sucessfully imported {:,} news articles as Pandas Dataframe 'allData_df':".format(len(allData_df)))
        return allData_df
    
    """
    do full text search for a keyword
    """
    def search(self,keyword: str, columns = ["title", "date", "link", "journal"], constraints = None, verbose=False, round_to_day=True):
        
        SQL_QUERRY = "SELECT "+", ".join(columns)  +" FROM news_fts WHERE news_fts MATCH '"+keyword+"'"
        if constraints is not None:
            SQL_QUERRY += " AND (" + constraints + ")"
                        
        if verbose:
            print("Search with querry: " + SQL_QUERRY)
        
        FTSdatetime= self.query(SQL_QUERRY,verbose)
        
        if verbose:
            print("Imported data from FTS: " + str(len(FTSdatetime)) + " articles.")

        FTSdatetime_df = pd.DataFrame(FTSdatetime,columns=columns)
        
        FTSdatetime_df = NewsFilter.apply(FTSdatetime_df)
        
        return FTSdatetime_df
    
    """
    evaluate sql query on the database
    """
    def query(self,query: str, verbose=False):

        con = sqlite3.connect(self.database)
        cur = con.cursor()
       
        if verbose:
            print("Connected to database " + self.database)
        
        allData = cur.execute(query).fetchall()
        if verbose:
            print("Fetched data successfull")
            
        con.close()
        
        return allData
    
    """
    fill in a default value for time points where no article was published
    """
    def fill_timeseries(self, original_timeseries, key, time_resolution, start_date, end_date ,  default_value = 0):
        # create empty time series
        journals = original_timeseries["journal"].unique()
        timeseries = pd.concat([pd.DataFrame({"date":pd.date_range(start_date, end_date, freq=time_resolution), "journal": jnl}) for jnl in journals])
        timeseries.date = timeseries.date.dt.floor(freq="D")

        # save original publication data into time series
        original_timeseries.set_index(["journal","date"],inplace=True)
        timeseries.set_index(["journal","date"],inplace=True)
        timeseries.loc[original_timeseries.index,key] = original_timeseries[key]
        timeseries.fillna(default_value,inplace=True)
        timeseries.reset_index(inplace=True)
        
        return timeseries
    
    """
    calculate the number of publications over time
    """
    def get_timeseries(self, keyword: str, data = None, start_date: str = "2021-01-01", end_date: str = "2021-12-31", resolution: str ="W", constraints = None, normalize: bool = True, fill_up = True, verbose=False):
        
        # default keyword if none is given and no search will be done
        if keyword is None: keyword = "title"
        
        # load the publication data into *FTSdatetime_df* and the normalisation data into *normalization*
        if data is None:
            if keyword == "title":
                FTSdatetime_df = self.load(columns = ["title", "link", "date", "journal", "meta"], constraints=constraints, verbose=verbose)
            else:
                FTSdatetime_df = self.search(keyword, verbose=verbose, constraints=constraints)
                
            if normalize:
                normalization = self.load(columns=['title', 'link', "journal", "date", "meta"], constraints=constraints, verbose=verbose)
        else:
            if normalize:
                if len(data) != 2:
                    raise ValueError("Got unexpected number of data frames.")
                FTSdatetime_df = data[0]
                normalization = data[1]
            else:
                FTSdatetime_df = data

        # round all date fields on the "day"-level
        FTSdatetime_df.date = FTSdatetime_df.date.dt.floor(freq="D")
        if normalize:
            normalization.date = normalization.date.dt.floor(freq="D")
        
        # count articles per time
        if verbose: print("Calc publication frequency...")
        news_timeseries = FTSdatetime_df.set_index("date")
        news_timeseries = news_timeseries.groupby(["journal",pd.Grouper(freq=resolution)]).title.count().to_frame(keyword).reset_index()
        news_timeseries = news_timeseries[(news_timeseries.date >= start_date) & (news_timeseries.date <= end_date)]
        if fill_up:
            if verbose: print("Fill missing time stamps...")
            news_timeseries = self.fill_timeseries(news_timeseries, key=keyword, time_resolution=resolution, start_date=start_date, end_date=end_date)
        
        # normalize with number of all articles
        if normalize:
            if verbose: print("Loading data for normalization...")   
            
            # count total number of published articles
            normalization = normalization.set_index("date")
            normalization = normalization.groupby(["journal",pd.Grouper(freq=resolution)]).title.count().to_frame("norm").reset_index()
            normalization = normalization[(normalization.date >= start_date) & (normalization.date <= end_date)]
            if fill_up:
                normalization = self.fill_timeseries(normalization, key='norm', time_resolution=resolution, start_date=start_date, end_date=end_date)
            
            # normalize *news_timeseries* with the total number of published articles
            normalization.set_index(["journal", "date"], inplace=True)
            news_timeseries.set_index(["journal", "date"], inplace=True)
            news_timeseries = news_timeseries[keyword]/normalization.norm
            news_timeseries.fillna(0,inplace=True) # handle division through zero
            news_timeseries = news_timeseries.to_frame(keyword).reset_index()

        return news_timeseries
    
    """
    remove articles in the database with a given link
    """
    def remove_by_link(self, urls, verbose = True):
        UBScon = sqlite3.connect(self.database)
        UBScur = UBScon.cursor()
        for link in urls:
            if verbose:
                print("Remove article " + link)
            sql_querry = "DELETE FROM news WHERE link='" + link + "';"
            try:
                UBScur.execute(sql_querry)
            except sqlite3.OperationalError as e:
                print(sql_querry)
                raise e
            UBScon.commit()
        UBScon.close()
        if verbose:
            print("Successfully removed "+str(len(urls))+" articles")
            
    """
    add articles in the database
    """
    def add(self, articles, verbose = True):
        UBScon = sqlite3.connect(self.database)
        UBScur = UBScon.cursor()
        articles.to_sql("news",UBScon,index=False, if_exists="append", method="multi")
        UBScon.close()
        if verbose:
            print("Successfully added "+str(len(articles))+" articles")
        
    """
    update the article property *entry* for a given articles. the article is identified by its link. 
    """
    def update_articles(self, articles, entry, verbose=True):
        if entry == "link":
            raise ValueError("Parameter 'entry' is not allowed to be 'link'.")
        UBScon = sqlite3.connect(self.database)
        UBScur = UBScon.cursor()
        for i, article in articles.iterrows():
            if verbose: print("Update article " + article.link)
            if "'" in str(article[entry]) or "'" in article.link:
                sql = 'UPDATE news SET '+ entry +'="' + str(article[entry]) + '" WHERE link="' + article.link + '";'
            else:
                sql = "UPDATE news SET "+ entry +"='" + str(article[entry]).replace('"',"'") + "' WHERE link='" + article.link + "';"
            try:
                UBScur.execute(sql)
            except sqlite3.OperationalError as e:
                print(sql)
                raise e
            UBScon.commit()
        UBScon.close()
        print("Successfully updated "+str(len(articles))+" articles")
        
    def update_urls(self, url_dict):
        UBScon = sqlite3.connect(self.database)
        UBScur = UBScon.cursor()
        for url in url_dict:
            print("Update article " + url + " to " + url_dict[url])
            sql = "UPDATE news SET link='" + url_dict[url] + "' WHERE link='" + url + "';"
            try:
                UBScur.execute(sql)
            except sqlite3.OperationalError as e:
                print(sql)
                UBScon.close()
                raise e
            UBScon.commit()
        UBScon.close()

    """
    get the tails of a given distribution
    """    
    def get_tails(self, distribution, q=0.999, verbose=True):
        right_qtl = distribution.quantile(q=q)
        distribution *= -1
        left_qtl = distribution.quantile(q=q) * -1
        distribution *= -1
        
        if verbose:
            print("left quantile = " + str(left_qtl))
            print("right quantile = " + str(right_qtl))

        left_tail_index = distribution[distribution <= left_qtl].index
        right_tail_index = distribution[distribution >= right_qtl].index
        
        return left_tail_index, right_tail_index
        
    """
    find articles with same text corpus AND same publication date
    """
    def find_duplicates(self, journal, data=None, verbose=True):
        
        # load data
        if data is None:
            if verbose: print("Load articles from " + journal)
            data = self.load(columns=["title","date", "text" , "link"], constraints="journal='"+ journal +"'")
        
        # hash text
        data['hash'] = data.text.apply(hash)
        
        # find duplicates in text and time
        if verbose: print("Find duplicated articles with same time ...")
        duplicates = data.groupby(["hash","date"]).title.count()

        return data.set_index(["hash","date"]).loc[duplicates[duplicates>1].index]
    
    def find_missing_data(self, journal, data = None):
        timeseries = self.get_timeseries(None, data=data, start_date=data.date.min(), end_date=data.date.max(), resolution="W", normalize=False, constraints="journal='"+journal+"'")
        return timeseries[timeseries.title == 0].date
        
    def overview_plot(self, journal):
        data = self.load(columns=["title", "date", "journal", "link", "LENGTH(text)"], constraints="journal='"+journal+"'")
        pub_timeseries = self.get_timeseries(None, data=data, start_date=data.date.min(), end_date=data.date.max(), resolution="M", normalize=False)
        pub_timeseries.set_index(["journal","date"], inplace=True)
        pub_timeseries_w = self.get_timeseries(None, data=data, start_date=data.date.min(), end_date=data.date.max(), resolution="W", normalize=False)
        pub_timeseries_w.set_index(["journal","date"], inplace=True)
        data.set_index("date",inplace=True)
        data["LENGTH(text)"] /= self.NORM_PAGE_LENGTH
        
        data_stop_words = self.search(keyword = self.GENERAL_BAG_OF_WORDS, constraints="journal='"+journal+"'")
        pub_timeseries_general = self.get_timeseries(self.GENERAL_BAG_OF_WORDS,data=data_stop_words, start_date=data.index.min(), end_date=data.index.max(), resolution="M", normalize=False)
        pub_timeseries_general.set_index(["journal","date"], inplace=True)
        
        
        FTSdatetime_df = self.search(keyword = "Klimawandel OR Klimakrise", constraints="journal='"+journal+"'")
        climatenews_timeseries_rel = self.get_timeseries("Klimawandel OR Klimakrise", data=(FTSdatetime_df,data.reset_index()), start_date=data.index.min(), end_date=data.index.max(), resolution="M")
        climatenews_timeseries = self.get_timeseries("Klimawandel OR Klimakrise", data=FTSdatetime_df, start_date=data.index.min(), end_date=data.index.max(), resolution="M", normalize=False)
        climatenews_timeseries_w = self.get_timeseries("Klimawandel OR Klimakrise", data=FTSdatetime_df, start_date=data.index.min(), end_date=data.index.max(), resolution="W", normalize=False)
        climatenews_timeseries_rel.set_index(["journal","date"], inplace=True)
        climatenews_timeseries.set_index(["journal","date"], inplace=True)
        climatenews_timeseries_w.set_index(["journal","date"], inplace=True)

        fig, axs = plt.subplot_mosaic([['a)', 'a)', 'c)'], ['b)','b)', 'd)'], ['e)', 'f)', 'g)']],
                                  constrained_layout=True)

        fig.set_size_inches(fig.get_size_inches()*1.5)
        
        fig.suptitle(journal)

        for label, ax in axs.items():
            ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

            if label == "a)":
                # Time Series of Monthly Publications
                ax.set_title('Publications (total)', fontstyle='italic')
                pub_timeseries.loc[journal,"title"].plot(ax=ax)
                ax.set_xlabel(None)
                ax.set_ylabel("article number")
            elif label == "b)":
                # Time Series of Climate Coverage
                ax.set_title('Climate Publications (share)', fontstyle='italic')
                climatenews_timeseries_rel.loc[journal,"Klimawandel OR Klimakrise"].plot(ax = ax)
                ax.set_xlabel(None)
                ax.set_ylabel("article share")
            elif label == "c)":
                # Distribution of Monthly Publications
                ax.set_title('Publications (monthly)', fontstyle='italic')
                pub_timeseries.loc[journal,"title"].plot.hist(ax=ax, bins=50, logy=True)
                ax.set_xlabel("article number")
                if ax.get_xlim()[0] > 0: ax.set_xlim(xmin=0)
            elif label == "d)":
                # Distribution of Monthly Climate Coverage
                ax.set_title('Climate Publ. (monthly)', fontstyle='italic')
                climatenews_timeseries_rel.loc[journal,"Klimawandel OR Klimakrise"].plot.hist(ax=ax, bins=50, logy=True)
                ax.set_xlabel("article share")
                if ax.get_xlim()[0] > 0: ax.set_xlim(xmin=0)
            elif label == "e)":
                # Distribution of Weekly Publications
                ax.set_title('Publications (weekly)', fontstyle='italic')
                pub_timeseries_w.loc[journal,"title"].plot.hist(ax=ax, bins=50, logy=True)
                ax.set_xlabel("article number")
                if ax.get_xlim()[0] > 0: ax.set_xlim(xmin=0)
            elif label == "f)":
                # Distribution of Weekly Climate Coverage
                ax.set_title('Climate Publ. (weekly)', fontstyle='italic')
                climatenews_timeseries_w.loc[journal,"Klimawandel OR Klimakrise"].plot.hist(ax=ax, bins=50, logy=True)
                ax.set_xlabel("article number")
                if ax.get_xlim()[0] > 0: ax.set_xlim(xmin=0)
            elif label == "g)":
                # Distribution of Article Length
                ax.set_title('Article Length', fontstyle='italic')
                
                bla = data.reset_index().set_index("link")
                bla = bla.loc[data_stop_words.link]
                
                bla["LENGTH(text)"].plot.hist(ax=ax, bins=50, logy=True)
                ax.set_xlabel("length [unit page]")
                if ax.get_xlim()[0] > 0: ax.set_xlim(xmin=0)
                
        return fig, axs
    
