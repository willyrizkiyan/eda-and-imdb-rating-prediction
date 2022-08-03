import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from annotated_text import annotated_text
# import scipy.stats as st
from numerize import numerize


st.set_page_config(page_title="IMDb Dashboard", layout="wide")

'''
# IMDb Dashboard
Made using **streamlit** by Willy Rizkiyan'''
annotated_text(('Hello','everyone,', '#8ef'), ('I Am', 'Willy','#faa'),)
'''
***
'''


with st.sidebar:
    st.write(
    '''
    # IMDb App
    Made using **streamlit** by Willy Rizkiyan
    '''
    )
    year = st.selectbox('Content', ('Dashboard', 'Machine Learning'))
    with st.expander('Disclaimer'):
        st.write(
            '''
            This data is web scrapped from IMDb site filtering by top 10.000 movies.
            '''
        )

st.write('''The Covid-19 pandemic has forced the desire to watch movies in theaters to be "suppressed" because many cinemas are not operating.
However, hobby of watching movies does not have to be "buried", because there are many choices of streaming platforms that can quench the thirst for watching entertainment content!
''')



st.subheader('Movie Genre Distribution')

data = pd.read_csv('imdb processed.csv')
data['star'] = data['star'].str.lower()
data['oscar_wins'] = data['oscar_wins'].astype(int)

time1, time2, time3, time4, time5 = st.columns([1,2,1,2,9])
with time1:
    st.write('From year')

with time2:
    min = st.number_input('', min_value=1913, max_value=2022, value=1913)

with time3:
    st.write('to year')

with time4:
    max = st.number_input('', min_value=1913, max_value=2022, value=2022)

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
table = table.sort_values('Total', ascending=False).reset_index(drop=True).head(10)

table_genre1, table_genre2 = st.columns([1,3])
with table_genre1:
    st.dataframe(table)
with table_genre2:
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.barplot(
        data = table,
        y='Genre',
        x='Total',
        color='cyan',
        ax=ax1
    )
    st.pyplot(fig1)


st.subheader('Budget and Gross Sales')
bud1, bud2 = st.columns([3,2])

data_sort_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
data_sort_budget['profit'] = data_sort_budget['worldwide_gross'] - data_sort_budget['budget']

with bud1:
    data_sort_budget['budget'] = data_sort_budget.apply(lambda x: "{:,}".format(x['budget']), axis=1)
    data_sort_budget['worldwide_gross'] = data_sort_budget.apply(lambda x: "{:,}".format(x['worldwide_gross']), axis=1)
    data_sort_budget['profit'] = data_sort_budget.apply(lambda x: "{:,}".format(x['profit']), axis=1)
    st.dataframe(data_sort_budget[['title', 'budget', 'worldwide_gross', 'profit']])

with bud2:
    data_sort_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
    data_sort_budget['profit'] = data_sort_budget['worldwide_gross'] - data_sort_budget['budget']

    max_budget = 'USD ' + numerize.numerize(data_sort_budget['budget'].max())
    st.write('Movie with highest budget is', data_sort_budget.sort_values('budget', ascending=False)['title'].head(1).values[0], 'with', max_budget)

    max_gross = 'USD ' + numerize.numerize(data_sort_budget['worldwide_gross'].max())
    st.write('Movie with highest gross is', data_sort_budget.sort_values('worldwide_gross', ascending=False)['title'].head(1).values[0], 'with', max_gross)

    max_profit = 'USD '  + numerize.numerize(data_sort_budget['profit'].max())
    st.write('Movie with highest profit is', data_sort_budget.sort_values('profit', ascending=False)['title'].head(1).values[0], 'with', max_profit)





st.subheader('Hypothesis Testing using ANOVA')

# tes1, tes2 = st.columns(2)
# with tes1:
#     alpha = st.slider('Alpha', min_value=0.01, max_value=0.1, value=0.05, step=0.01)
#     type = st.radio('Parental Guide', ('Nudity', 'Violence', 'Profanity', 'Alcohol', 'Frightening'))
#     type = type.str.lower()
#     none = data[data[type]=='None']
#     mild = data[data[type]=='Mild']
#     moderate = data[data[type]=='Moderate']
#     severe = data[data[type]=='Severe']

#     anova_result=st.f_oneway(none['rate'],mild['rate'],moderate['rate'],severe['rate'])
#     p_value=anova_result.pvalue
#     p_value
#     if p_value > alpha:
#         st.write(type + 'has no effect on Rate')
#     else:
#         st.write(type +'has effect on Rate')



st.subheader('Movie Recommendation by Genre')
met1, met2 = st.columns(2)
with met1:
    genre = st.selectbox('Select genre: ', ('Action', 'Adult', 'Adventure', 'Animation',
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
