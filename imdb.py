from calendar import c
from ctypes import cdll
from this import d
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from annotated_text import annotated_text
import scipy.stats as stat
from numerize import numerize
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import hydralit_components as hc
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures


st.set_page_config(page_title="IMDb Apps", layout="wide")

# specify the primary menu definition
menu_data = [
        {'icon': "fas fa-tachometer-alt", 'label':"Dashboard", 'ttip':"Interactive Dashboard"},
        {'icon': "far fa-copy", 'label':"Quiz", 'ttip':"Let's Get Some Quiz"},
        {'icon': "far fa-chart-bar", 'label':"Apps", 'ttip':"Movie Recommendation and Prediction"},
        {'icon': "bi bi-hand-thumbs-up", 'label':"Summary", 'ttip':"Summary and Recommendation"},
        {'icon': "far fa-address-book", 'label':"Contact", 'ttip':"Contact Me"},
]

over_theme = {'txc_inactive': '#FFFFFF','menu_background':'#87CEFA','txc_active':'black','option_active':'white'}
menu_id = hc.nav_bar(menu_definition=menu_data, home_name='Home', override_theme=over_theme)

with st.sidebar:
    st.write(
    '''
    # IMDb App
    Made using **streamlit** by Willy Rizkiyan
    '''
    )
    with st.expander('Disclaimer'):
        st.write(
            '''
            This data is web scrapped from IMDb site filtering by top 10.000 movies.
            '''
        )


data = pd.read_csv('imdb processed.csv')
data['star'] = data['star'].str.lower()


if menu_id == 'Home':
    '''
    # Home
    '''
    annotated_text(('Hello','everyone,', '#8ef'), ('I Am', 'Willy','#faa'),)
    '''
    ***
    '''

    st.subheader('Background')
    st.info('''
    The COVID-19 pandemic has impacted almost every aspect of our lives, and the film industry is no exception. It is estimated that the US box office alone will lose billions of dollars.
    ''')

    st.info('''The Covid-19 pandemic has forced the desire to watch movies in theaters to be "suppressed" because many cinemas are not operating.
    However, hobby of watching movies does not have to be "buried", because there are many choices of streaming platforms that can quench the thirst for watching movies.
    ''')

    st.info('This dashboard and apps can be used by people who wants to get recommendation for watching movies and also for film maker to get insight about movies that want to be produced.')

if menu_id == 'Dashboard':
    '''
    # Dashboard
    '''
    annotated_text(('Hello','everyone,', '#8ef'), ('I Am', 'Willy','#faa'),)
    '''
    ***
    '''
    data_movie = pd.DataFrame(data['year'].value_counts()).reset_index().rename(
    columns={'index': 'year', 'year':'total'}).sort_values('year')    
    
    st.header('Total Movie Every Year')
    movie1, movie2= st.columns([2,1])
    with movie1:
        fig = px.line(data_movie, x='year', y='total',
              labels={'year':'Year','total':'Total Film'}, 
              markers=True, height=400, width = 900)
        st.write(fig)

    with movie2:
        st.subheader('\n')
        st.subheader('\n')
        st.subheader('\n')
        st.info('''
        Film production relatively continues to increase from year to year.

        However, there has been a decline in the number of times during the COVID-19 pandemic.

        If there is no pandemic, it is predicted that film production will continue to increase.
        ''')


    st.subheader('\n')
    st.header('Movie Duration Distribution')
    scatter1, scatter2, scatter3 = st.columns([1,1,1])
    with scatter1:
        fig = px.scatter(data, x='year', y='duration', width = 450, height=450,
                 labels={'year':'Year','duration':'Duration'})
        st.write(fig)

    with scatter2:
        fig = px.histogram(data, x='duration', width = 450, height=450,
                   labels={'duration':'Duration'},
                   nbins=40)
        st.write(fig)

    with scatter3:
        st.subheader('\n')
        st.subheader('\n')
        st.subheader('\n')
        st.subheader('\n')
        st.subheader('\n')
        st.subheader('Duration of movies mostly around 80 - 120 mins')

    st.subheader('\n')
    st.header('Movie Genre Distribution')

    time1, time2, time3, time4, time5 = st.columns([3,2,3,2,9])

    with time1:
        min = st.number_input('From year', min_value=1913, max_value=2022, value=1913)

    with time3:
        max = st.number_input('To year', min_value=1913, max_value=2022, value=2022)

    list_genre = ['Action', 'Adult', 'Adventure', 'Animation',
        'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Horror', 'History',
        'Music', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']

    tabel = []

    for i in list_genre:
        a = []
        x = data['title'][(data['is_'+i] == 1) & (data['year']>=min) & (data['year']<=max)].count()
        a.append(i)
        a.append(x)
        tabel.append(a)

    table = pd.DataFrame(tabel, columns=['Genre', 'Total'])
    table = table.sort_values('Total', ascending=False).reset_index(drop=True).head(5)
    table.index = table.index + 1

    table_genre1, table_genre2 = st.columns([1,1])
    with table_genre1:
        table = table.sort_values(by='Total', ascending=False).head(5)
        fig = px.bar(table, x='Total', y='Genre',
                    text='Genre',
                    color='Genre',
                    labels={'Total':'Total'},
                    width=650, height=325)
        fig.update_traces(textfont_size=16, textangle=0, textposition='inside',insidetextanchor ='start')
        fig.update_yaxes(visible=False)
        fig.update_layout(showlegend=False)
        st.write(fig)

    with table_genre2:
        a = table.iloc[0,0]
        b = table.iloc[1,0]
        c = table.iloc[2,0]
        d = table.iloc[3,0]
        e = table.iloc[4,0]

        st.subheader('\n')
        st.subheader('\n')
        st.subheader('\n')
        st.info(f'''
        **{a}**, **{b}**, **{c}**, **{d}**, and **{e}** are the most widely produced film from **{min}** until **{max}**.

        Next, we will see what the best films are based on ratings for those genre.
        ''')

    list_genre = ['Action', 'Adult', 'Adventure', 'Animation',
    'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Horror', 'History',
    'Music', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']



    tabel = []

    for i in list_genre:
        tab_year = 0
        for j in range(1913,2023):
            tab = []
            x = data['title'][(data['is_'+i] == 1) & (data['year'] == j)].count()
            tab_year = x + tab_year
            tab.append(i)
            tab.append(j)
            tab.append(tab_year)
            j = j+1
            tabel.append(tab)

    table = pd.DataFrame(tabel, columns=['Genre','Year', 'Total'])

    race = st.checkbox('Race Bar Chart')
    if race:
        fig = px.bar(table, x='Total', y='Genre',
             title='Movie Genre',
             color='Genre',
             animation_frame='Year',
             range_x=[0,6000],
             height=600,
             width =1000)

        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 100
        fig.update_yaxes(categoryorder='total ascending')
        fig.update_layout(showlegend=False)
        st.write(fig)

    st.subheader('\n')
    st.header(f"Top Movie Based on Genre from {min} until {max}")

    movie1, movie2, movie3, movie4, movie5 = st.columns([1,1,1,1,1])
    with movie1:
        st.subheader(a)
        data_genre = data.sort_values('rate', ascending=False)
        data_genre = data_genre[(data_genre['is_'+a]==1) & (data['year']>=min) & (data['year']<=max)].head(5)
        list_genre = data_genre['title'].reset_index(drop=True)
        list_genre.index = list_genre.index + 1
        st.table(list_genre)
        
    with movie2:
        st.subheader(b)
        data_genre = data.sort_values('rate', ascending=False)
        data_genre = data_genre[(data_genre['is_'+b]==1) & (data['year']>=min) & (data['year']<=max)].head(5)
        list_genre = data_genre['title'].reset_index(drop=True)
        list_genre.index = list_genre.index + 1
        st.table(list_genre)

    with movie3:
        st.subheader(c)
        data_genre = data.sort_values('rate', ascending=False)
        data_genre = data_genre[(data_genre['is_'+c]==1) & (data['year']>=min) & (data['year']<=max)].head(5)
        list_genre = data_genre['title'].reset_index(drop=True)
        list_genre.index = list_genre.index + 1
        st.table(list_genre)

    with movie4:
        st.subheader(d)
        data_genre = data.sort_values('rate', ascending=False)
        data_genre = data_genre[(data_genre['is_'+d]==1) & (data['year']>=min) & (data['year']<=max)].head(5)
        list_genre = data_genre['title'].reset_index(drop=True)
        list_genre.index = list_genre.index + 1
        st.table(list_genre)

    with movie5:
        st.subheader(e)
        data_genre = data.sort_values('rate', ascending=False)
        data_genre = data_genre[(data_genre['is_'+e]==1) & (data['year']>=min) & (data['year']<=max)].head(5)
        list_genre = data_genre['title'].reset_index(drop=True)
        list_genre.index = list_genre.index + 1
        st.table(list_genre)


    st.subheader('\n')
    st.header('Most Played Actor')

    actor1 = data['star1'].tolist()
    actor2 = data['star2'].tolist()
    actor3 = data['star3'].tolist()
    actor4 = data['star4'].tolist()

    actor = actor1 + actor2 + actor3 + actor4

    data_actor = pd.DataFrame(actor, columns =['Actor']) 

    value_counts = data_actor['Actor'].value_counts(dropna=True, sort=True)

    data_counts = pd.DataFrame(value_counts)
    data_counts = data_counts.reset_index()
    data_counts.columns = ['Actor', 'Total']
    data_counts.index = data_counts.index + 1

    most_actor = data_counts.sort_values('Total', ascending=False)['Actor'].head(1).values[0]
    count1, count2, count3 = st.columns([2,1,2])
    with count1:
        st.table(data_counts.head(10))
        st.info(f'Most played actor is {most_actor}')
    
    with count3:
        st.info(f'Top film acted by {most_actor}')
        most_actor = most_actor.lower()
        movie_actor = data[data['star'].str.contains(most_actor)]
        movie_actor = movie_actor.sort_values('rate', ascending=False)
        movie_title = movie_actor[['title','genre']].reset_index(drop=True).head(10)
        movie_title.index = movie_title.index + 1
        st.table(movie_title)


if menu_id == 'Quiz':
    '''
    # Quiz
    Made using **streamlit** by Willy Rizkiyan'''
    annotated_text(('Hello','everyone,', '#8ef'), ('I Am', 'Willy','#faa'),)
    '''
    ***
    '''

    st.header('Budget and Gross Sales')
    st.subheader('**Which one is the most successful movie**')
    success = st.radio('', ('Pick One :','Avengers: Infinity War', 'Avengers: Endgame', 'Avatar'), horizontal=True)
    img1, img2, img3, img4, img5, img6 = st.columns(6)
    with img1:
        st.image('https://m.media-amazon.com/images/M/MV5BMjMxNjY2MDU1OV5BMl5BanBnXkFtZTgwNzY1MTUwNTM@._V1_.jpg', width=200)
    with img2:
        st.image('https://m.media-amazon.com/images/M/MV5BMTc5MDE2ODcwNV5BMl5BanBnXkFtZTgwMzI2NzQ2NzM@._V1_.jpg', width=200)
    with img3:
        st.image('https://m.media-amazon.com/images/M/MV5BZDA0OGQxNTItMDZkMC00N2UyLTg3MzMtYTJmNjg3Nzk5MzRiXkEyXkFqcGdeQXVyMjUzOTY1NTc@._V1_.jpg', width=200)
    if success == 'Avatar':
        st.success('You Are Right')
        bud1, bud2, bud3 = st.columns([1,1,1])

        with bud1:
            data_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
            data_budget = data_budget.sort_values(by='budget', ascending=False)

            fig = px.bar(data_budget, x='budget', y='title',
                        title='Top Movies Budget',
                        text='title',
                        color='title',
                        labels={'budget':'Budget','title':'Film'})
            fig.update_traces(textfont_size=15, textangle=0, textposition='inside',insidetextanchor ='start')
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False, width=500)
            st.write(fig)

        with bud2:
            data_gross = data.sort_values('worldwide_gross', ascending=False).head(10).reset_index(drop=True)
            data_gross = data_gross.sort_values(by='worldwide_gross', ascending=False)

            fig = px.bar(data_gross, x='worldwide_gross', y='title',
                        title='Top Movies Gross',
                        text='title',
                        color='title',
                        labels={'worldwide_gross':'Gross','title':'Film'})
            fig.update_traces(textfont_size=15, textangle=0, textposition='inside',insidetextanchor ='start')
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False, width=500)
            st.write(fig)

        with bud3:
            data_profit = data.copy()
            data_profit['profit'] = data_profit['worldwide_gross'] - data_profit['budget']
            data_profit = data_profit.sort_values('profit', ascending=False).head(10).reset_index(drop=True)
            data_profit = data_profit.sort_values(by='profit', ascending=False)

            fig = px.bar(data_profit, x='profit', y='title',
                        title='Top Movies Profit',
                        text='title',
                        color='title',
                        labels={'profit':'Profit','title':'Film'})
            fig.update_traces(textfont_size=15, textangle=0, textposition='inside',insidetextanchor ='start')
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False, width=500)
            st.write(fig)

        max1, max2, max3 = st.columns([1,1,1])
        with max1:
            max_budget = 'USD ' + numerize.numerize(data['budget'].max())
            highest_budget = data.sort_values('budget', ascending=False)['title'].head(1).values[0]
            st.info(f'Movie with highest budget is **{highest_budget}** with **{max_budget}**')
        with max2:
            max_gross = 'USD ' + numerize.numerize(data['worldwide_gross'].max())
            highest_gross = data.sort_values('worldwide_gross', ascending=False)['title'].head(1).values[0]
            st.info(f'Movie with highest gross is **{highest_gross}** with **{max_gross}**')
        with max3:
            max_profit = 'USD '  + numerize.numerize(data_profit['profit'].max())          
            highest_profit = data_profit.sort_values('profit', ascending=False)['title'].head(1).values[0]
            st.info(f'Movie with highest profit is **{highest_profit}** with **{max_profit}**')
            st.write('\n')
        st.subheader(f'{highest_profit} is the most successful movie until now')
    
    elif success == 'Avengers: Infinity War':
        st.error('Sorry')
        bud1, bud2, bud3 = st.columns([1,1,1])

        with bud1:
            data_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
            data_budget = data_budget.sort_values(by='budget', ascending=False)

            fig = px.bar(data_budget, x='budget', y='title',
                        title='Top Movies Budget',
                        text='title',
                        color='title',
                        labels={'budget':'Budget','title':'Film'})
            fig.update_traces(textfont_size=15, textangle=0, textposition='inside',insidetextanchor ='start')
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False, width=500)
            st.write(fig)

        with bud2:
            data_gross = data.sort_values('worldwide_gross', ascending=False).head(10).reset_index(drop=True)
            data_gross = data_gross.sort_values(by='worldwide_gross', ascending=False)

            fig = px.bar(data_gross, x='worldwide_gross', y='title',
                        title='Top Movies Gross',
                        text='title',
                        color='title',
                        labels={'worldwide_gross':'Gross','title':'Film'})
            fig.update_traces(textfont_size=15, textangle=0, textposition='inside',insidetextanchor ='start')
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False, width=500)
            st.write(fig)

        with bud3:
            data_profit = data.copy()
            data_profit['profit'] = data_profit['worldwide_gross'] - data_profit['budget']
            data_profit = data_profit.sort_values('profit', ascending=False).head(10).reset_index(drop=True)
            data_profit = data_profit.sort_values(by='profit', ascending=False)

            fig = px.bar(data_profit, x='profit', y='title',
                        title='Top Movies Profit',
                        text='title',
                        color='title',
                        labels={'profit':'Profit','title':'Film'})
            fig.update_traces(textfont_size=15, textangle=0, textposition='inside',insidetextanchor ='start')
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False, width=500)
            st.write(fig)
 
        max1, max2, max3 = st.columns([1,1,1])
        with max1:
            max_budget = 'USD ' + numerize.numerize(data['budget'].max())
            highest_budget = data.sort_values('budget', ascending=False)['title'].head(1).values[0]
            st.info(f'Movie with highest budget is **{highest_budget}** with **{max_budget}**')
        with max2:
            max_gross = 'USD ' + numerize.numerize(data['worldwide_gross'].max())
            highest_gross = data.sort_values('worldwide_gross', ascending=False)['title'].head(1).values[0]
            st.info(f'Movie with highest gross is **{highest_gross}** with **{max_gross}**')
        with max3:
            max_profit = 'USD '  + numerize.numerize(data_profit['profit'].max())          
            highest_profit = data_profit.sort_values('profit', ascending=False)['title'].head(1).values[0]
            st.info(f'Movie with highest profit is **{highest_profit}** with **{max_profit}**')
            st.write('\n')
        st.subheader(f'{highest_profit} is the most successful movie until now')

    elif success == 'Avengers: Endgame':
        st.error('Sorry')
        bud1, bud2, bud3 = st.columns([1,1,1])

        with bud1:
            data_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
            data_budget = data_budget.sort_values(by='budget', ascending=False)

            fig = px.bar(data_budget, x='budget', y='title',
                        title='Top Movies Budget',
                        text='title',
                        color='title',
                        labels={'budget':'Budget','title':'Film'})
            fig.update_traces(textfont_size=15, textangle=0, textposition='inside',insidetextanchor ='start')
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False, width=500)
            st.write(fig)

        with bud2:
            data_gross = data.sort_values('worldwide_gross', ascending=False).head(10).reset_index(drop=True)
            data_gross = data_gross.sort_values(by='worldwide_gross', ascending=False)

            fig = px.bar(data_gross, x='worldwide_gross', y='title',
                        title='Top Movies Gross',
                        text='title',
                        color='title',
                        labels={'worldwide_gross':'Gross','title':'Film'})
            fig.update_traces(textfont_size=15, textangle=0, textposition='inside',insidetextanchor ='start')
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False, width=500)
            st.write(fig)

        with bud3:
            data_profit = data.copy()
            data_profit['profit'] = data_profit['worldwide_gross'] - data_profit['budget']
            data_profit = data_profit.sort_values('profit', ascending=False).head(10).reset_index(drop=True)
            data_profit = data_profit.sort_values(by='profit', ascending=False)

            fig = px.bar(data_profit, x='profit', y='title',
                        title='Top Movies Profit',
                        text='title',
                        color='title',
                        labels={'profit':'Profit','title':'Film'})
            fig.update_traces(textfont_size=15, textangle=0, textposition='inside',insidetextanchor ='start')
            fig.update_yaxes(visible=False)
            fig.update_layout(showlegend=False, width=500)
            st.write(fig)

        max1, max2, max3 = st.columns([1,1,1])
        with max1:
            max_budget = 'USD ' + numerize.numerize(data['budget'].max())
            highest_budget = data.sort_values('budget', ascending=False)['title'].head(1).values[0]
            st.info(f'Movie with highest budget is **{highest_budget}** with **{max_budget}**')
        with max2:
            max_gross = 'USD ' + numerize.numerize(data['worldwide_gross'].max())
            highest_gross = data.sort_values('worldwide_gross', ascending=False)['title'].head(1).values[0]
            st.info(f'Movie with highest gross is **{highest_gross}** with **{max_gross}**')
        with max3:
            max_profit = 'USD '  + numerize.numerize(data_profit['profit'].max())          
            highest_profit = data_profit.sort_values('profit', ascending=False)['title'].head(1).values[0]
            st.info(f'Movie with highest profit is **{highest_profit}** with **{max_profit}**')
            st.write('\n')
        st.subheader(f'{highest_profit} is the most successful movie until now')

    else:
        st.info('Please pick one first')



if menu_id == 'Apps':
    '''
    # Apps
    '''
    annotated_text(('Hello','everyone,', '#8ef'), ('I Am', 'Willy','#faa'),)
    '''
    ***
    '''

    st.subheader('Movie Recommendation by Genre and Actor')
    met1, met2 = st.columns(2)
    with met1:
        genre = st.selectbox('Select genre ', ('Action', 'Adult', 'Adventure', 'Animation',
        'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Horror', 'History',
        'Music', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western'))

    with met2:
        actor= st.text_input('Input actor')
        actor1 = actor.lower()
        
    data_sort = data.sort_values('rate', ascending=False)
    data_sort = data_sort[(data_sort['is_'+genre]==1)
        & data_sort['star'].str.contains(actor1)][['title', 'year', 'star', 'oscar_wins', 'total_wins']].head(10).reset_index(drop=True)
    list_data = data_sort['title'].to_list()

    result1, result2, result3 = st.columns(3)
    with result1:
        genre = st.radio(
            "Which one do you wanna know?",
            (list_data))

    with result2:
        if genre:
            st.subheader(genre)
            result = data['plot'][data['title']==genre].reset_index(drop=True)
            st.write(result.values[0])
        else:
            st.write('No Data')

    with result3:
        if genre:
            result_img = data['url_image'][data['title']==genre].reset_index(drop=True)
            result_img = result_img.values[0]
            st.image(result_img, width=200)
        else:
            st.write('')


    data_actor = pd.read_csv('actor rank.csv', sep='\t')

    data_actor['actor'] = data_actor['actor'].str.lower()

    st.write('\n')
    st.write('\n')
    st.write('\n')

    imdb_joblib = joblib.load('imdb_joblib')

    with st.form("Prediction Form"): 
        st.subheader('Worldwide Gross Prediction')
        st.write('\n')
        st.write('\n')
        st.write('Please specify the variable :')
        act1, act2, act3, act4 = st.columns(4)
        with act1:
            a1 = st.text_input('Input 1st Actor :')
        with act2:
            a2 = st.text_input('Input 2nd Actor :')
        with act3:
            a3 = st.text_input('Input 3rd Actor :')
        with act4:
            a4 = st.text_input('Input 4th Actor :')

        a1_value = data_actor['rank_actor'].loc[data_actor['actor'].str.contains(a1)].to_list()
        if a1 == '':
            a1_value = 1
        elif a1_value:
            a1_value = 1001 - a1_value[0]
        else:
            a1_value = 1
        a2_value = data_actor['rank_actor'].loc[data_actor['actor'].str.contains(a2)].to_list()
        if a2 == '':
            a2_value = 1        
        elif a2_value:
            a2_value = 1001 - a2_value[0]
        else:
            a2_value = 1
        a3_value = data_actor['rank_actor'].loc[data_actor['actor'].str.contains(a3)].to_list()
        if a3 == '':
            a3_value = 1
        elif a3_value:
            a3_value = 1001 - a3_value[0]
        else:
            a3_value = 1
        a4_value = data_actor['rank_actor'].loc[data_actor['actor'].str.contains(a4)].to_list()
        if a4 == '':
            a4_value = 1       
        elif a4_value:
            a4_value = 1001 - a4_value[0]
        else:
            a4_value = 1
        total_value = a1_value + a2_value + a3_value + a4_value

        set1, set2, set3, set4, set5 = st.columns(5)
        with set1:
            violence = st.select_slider('Choose Violence:', ('None', 'Mild', 'Moderate','Severe'))
            if violence == 'Mild':
                violence = 1
            elif violence == 'Moderate':
                violence = 2
            elif violence == 'Severe':
                violence = 3
            else:
                violence = 0
        with set2:
            nudity = st.select_slider('Choose Nudity:', ('None', 'Mild', 'Moderate','Severe'))
            if nudity == 'Mild':
                nudity = 1
            elif nudity == 'Moderate':
                nudity = 2
            elif nudity == 'Severe':
                nudity = 3
            else:
                nudity = 0
        with set3:
            profanity = st.select_slider('Choose Profanity:', ('None', 'Mild', 'Moderate','Severe'))
            if profanity == 'Mild':
                profanity = 1
            elif profanity == 'Moderate':
                profanity = 2
            elif profanity == 'Severe':
                profanity = 3
            else:
                profanity = 0
        with set4:
            alcohol = st.select_slider('Choose Alcohol:', ('None', 'Mild', 'Moderate','Severe'))
            if alcohol == 'Mild':
                alcohol = 1
            elif alcohol == 'Moderate':
                alcohol = 2
            elif alcohol == 'Severe':
                alcohol = 3
            else:
                alcohol = 0
        with set5:
            frightening = st.select_slider('Choose Frightening:', ('None', 'Mild', 'Moderate','Severe'))
            if frightening == 'Mild':
                frightening = 1
            elif frightening == 'Moderate':
                frightening = 2
            elif frightening == 'Severe':
                frightening = 3
            else:
                frightening = 0

        dur, bud = st.columns(2)
        with dur:
            duration = st.number_input('Movie Duration (in minutes):', min_value=30, max_value=300, value=120)
            duration = pd.to_numeric(duration)
        with bud:
            budget = st.number_input('Movie Budget (in USD million):', min_value = 1, max_value=500, value=200)
            budget = int(budget)*1000000

        data_predict = {'duration':duration, 'nudity':nudity, 'violence':violence, 'profanity':profanity,
                    'alcohol':alcohol, 'frightening':frightening, 'budget':budget, 'is_Action':1,
                    'is_Adventure':0, 'is_Comedy':1, 'is_Drama':0,
                    'is_Romance':1, 'is_Other':0,
                    'total_star_score':total_value
                    }

        data_predict= pd.DataFrame(data_predict, index=[0])
        poly_reg = PolynomialFeatures(degree = 2)
        data_predict = poly_reg.fit_transform(data_predict)

        submitted = st.form_submit_button("Predict")
        if submitted:
            res1, res2, res3 = st.columns(3)
            with res1:
                predict = imdb_joblib.predict(data_predict)
                predict = int(predict[0])
                predict = '$ ' + numerize.numerize(predict)
                st.write('**Worldwide Gross Prediction :**')
                st.info(predict)


if menu_id == 'Summary':
    '''
    # Summary
    '''
    annotated_text(('Hello','everyone,', '#8ef'), ('I Am', 'Willy','#faa'),)
    '''
    ***
    '''

    st.subheader('About this project')
    st.info('''
    This project includes web scraping, data cleaning and manipulation, exploratory data analysis, hypothesis testing, modelling machine learning until deployment into web app. The contents of this web app include interactive imdb data dashboard, imdb quiz and imdb gross prediction.
    
    Dataset from this project is 10.000 movies scraped from www.imdb.com site on August, 3rd 2022 using BeautifulSoup. There are 2 ways of scraping used in getting this dataset, using html and using the json file format which is then tidied up in Pentaho apps. For actor ranking dataset using top stars at the worldwide box office from www.the-numbers.com site on August, 5th 2022. After getting the dataset then proceed with data wrangling, EDA and machine learning modelling using some regression models to predict movie worldwide gross.
    ''')

    st.subheader('Linear Regression Interpretation')
    st.info('''
    **While the other features are kept fixed :**
    * An increase of 1 minute in duration is associated with an increase of USD 732244.4 in gross.
    * An increase of USD 1 in budget is associated with an increase of USD 2.85 in gross.
    * An increase of 1 point in total_star_score is associated with an increase of USD 20609.42 in gross.
    ''')
    
    st.subheader('Rooms For Improvement')
    st.info('''
    * Use more dataset out of 10.000 top movies to make more representative.
    * Need more exploration and feature engineering to get higher accuracy.
    ''')

if menu_id == 'Contact':
    '''
    # Contact
    '''
    annotated_text(('Hello','everyone,', '#8ef'), ('I Am', 'Willy','#faa'),)
    '''
    ***
    '''

    st.header('Author')
    st.info('''
    Hello, My name is Willy Rizkiyan. Thank you for visiting my project.
    
    I was graduated from Bandung Institute of Technology majoring Metallurgical Engineering.
    I had worked in a mining company for almost 7 years. I made database and made simulation for production planning and product costing using historical data.

    From elementary school, I have loved numbers and joined mathematics olympiad until senior high school.
    I also taught private mathematics when I was in college to junior and high school students.
    I've always loved puzzles and analyze any kind.

    Please reach me for any suggestion for this project or just connect to me.

    Thank you!
    ''')

    contact1, contact2 = st.columns([1,20])
    with contact1:
        st.image('Gmail.png', width=47)
        st.image('LinkedIn.png', width=40)
        st.image('Github.png', width=43)

    with contact2:
        st.subheader('willyrizkiyan@gmail.com')
        st.subheader('https://www.linkedin.com/in/willyrizkiyan')
        st.subheader('https://www.github.com/willyrizkiyan')
