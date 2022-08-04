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
st.markdown('**Which one is the most successful movie**')
success = st.radio('', ('Pick One :','Avengers: Infinity War', 'Avengers: Endgame', 'Jurassic Park'), horizontal=True)
img1, img2, img3, img4 = st.columns(4)
with img1:
    st.image('https://m.media-amazon.com/images/M/MV5BMjMxNjY2MDU1OV5BMl5BanBnXkFtZTgwNzY1MTUwNTM@._V1_.jpg', width=200)
with img2:
    st.image('https://m.media-amazon.com/images/M/MV5BMTc5MDE2ODcwNV5BMl5BanBnXkFtZTgwMzI2NzQ2NzM@._V1_.jpg', width=200)
with img3:
    st.image('https://m.media-amazon.com/images/M/MV5BMjM2MDgxMDg0Nl5BMl5BanBnXkFtZTgwNTM2OTM5NDE@._V1_.jpg', width=200)
if success == 'Avengers: Endgame':
    st.subheader('You Are Right')
    bud1, bud2 = st.columns([3,2])

    data_sort_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
    data_sort_budget['profit'] = data_sort_budget['worldwide_gross'] - data_sort_budget['budget']

    with bud1:
        data_sort_budget['budget'] = data_sort_budget.apply(lambda x: "{:,}".format(x['budget']), axis=1)
        data_sort_budget['worldwide_gross'] = data_sort_budget.apply(lambda x: "{:,}".format(x['worldwide_gross']), axis=1)
        data_sort_budget['profit'] = data_sort_budget.apply(lambda x: "{:,}".format(x['profit']), axis=1)
        st.dataframe(data_sort_budget[['title', 'budget', 'worldwide_gross', 'profit']])

    with bud2:
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        
        data_sort_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
        data_sort_budget['profit'] = data_sort_budget['worldwide_gross'] - data_sort_budget['budget']

        max_budget = 'USD ' + numerize.numerize(data_sort_budget['budget'].max())
        st.write('Movie with highest budget is', data_sort_budget.sort_values('budget', ascending=False)['title'].head(1).values[0], 'with', max_budget)

        max_gross = 'USD ' + numerize.numerize(data_sort_budget['worldwide_gross'].max())
        st.write('Movie with highest gross is', data_sort_budget.sort_values('worldwide_gross', ascending=False)['title'].head(1).values[0], 'with', max_gross)

        max_profit = 'USD '  + numerize.numerize(data_sort_budget['profit'].max())
        st.write('Movie with highest profit is', data_sort_budget.sort_values('profit', ascending=False)['title'].head(1).values[0], 'with', max_profit)
        st.write('\n')
        st.subheader('Avengers : Endgame is the most successful movie until now')
elif success == 'Avengers: Infinity War':
    st.subheader('Sorry')
    bud1, bud2 = st.columns([3,2])

    data_sort_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
    data_sort_budget['profit'] = data_sort_budget['worldwide_gross'] - data_sort_budget['budget']

    with bud1:
        data_sort_budget['budget'] = data_sort_budget.apply(lambda x: "{:,}".format(x['budget']), axis=1)
        data_sort_budget['worldwide_gross'] = data_sort_budget.apply(lambda x: "{:,}".format(x['worldwide_gross']), axis=1)
        data_sort_budget['profit'] = data_sort_budget.apply(lambda x: "{:,}".format(x['profit']), axis=1)
        st.dataframe(data_sort_budget[['title', 'budget', 'worldwide_gross', 'profit']])

    with bud2:
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        
        data_sort_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
        data_sort_budget['profit'] = data_sort_budget['worldwide_gross'] - data_sort_budget['budget']

        max_budget = 'USD ' + numerize.numerize(data_sort_budget['budget'].max())
        st.write('Movie with highest budget is', data_sort_budget.sort_values('budget', ascending=False)['title'].head(1).values[0], 'with', max_budget)

        max_gross = 'USD ' + numerize.numerize(data_sort_budget['worldwide_gross'].max())
        st.write('Movie with highest gross is', data_sort_budget.sort_values('worldwide_gross', ascending=False)['title'].head(1).values[0], 'with', max_gross)

        max_profit = 'USD '  + numerize.numerize(data_sort_budget['profit'].max())
        st.write('Movie with highest profit is', data_sort_budget.sort_values('profit', ascending=False)['title'].head(1).values[0], 'with', max_profit)
        st.write('\n')
        st.subheader('Avengers : Endgame is the most successful movie until now')

elif success == 'Jurassic Park':
    st.subheader('Sorry')
    bud1, bud2 = st.columns([3,2])

    data_sort_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
    data_sort_budget['profit'] = data_sort_budget['worldwide_gross'] - data_sort_budget['budget']

    with bud1:
        data_sort_budget['budget'] = data_sort_budget.apply(lambda x: "{:,}".format(x['budget']), axis=1)
        data_sort_budget['worldwide_gross'] = data_sort_budget.apply(lambda x: "{:,}".format(x['worldwide_gross']), axis=1)
        data_sort_budget['profit'] = data_sort_budget.apply(lambda x: "{:,}".format(x['profit']), axis=1)
        st.dataframe(data_sort_budget[['title', 'budget', 'worldwide_gross', 'profit']])

    with bud2:
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        
        data_sort_budget = data.sort_values('budget', ascending=False).head(10).reset_index(drop=True)
        data_sort_budget['profit'] = data_sort_budget['worldwide_gross'] - data_sort_budget['budget']

        max_budget = 'USD ' + numerize.numerize(data_sort_budget['budget'].max())
        st.write('Movie with highest budget is', data_sort_budget.sort_values('budget', ascending=False)['title'].head(1).values[0], 'with', max_budget)

        max_gross = 'USD ' + numerize.numerize(data_sort_budget['worldwide_gross'].max())
        st.write('Movie with highest gross is', data_sort_budget.sort_values('worldwide_gross', ascending=False)['title'].head(1).values[0], 'with', max_gross)

        max_profit = 'USD '  + numerize.numerize(data_sort_budget['profit'].max())
        st.write('Movie with highest profit is', data_sort_budget.sort_values('profit', ascending=False)['title'].head(1).values[0], 'with', max_profit)
        st.write('\n')
        st.subheader('Avengers : Endgame is the most successful movie until now')

else:
    st.write('Please pick one first')


st.subheader('Hypothesis Testing using ANOVA')

tes1, tes2 = st.columns(2)
with tes1:
    alpha = st.slider('Alpha', min_value=0.01, max_value=0.1, value=0.05, step=0.01)
    type = st.radio('Parental Guide', ('Nudity', 'Violence', 'Profanity', 'Alcohol', 'Frightening'))
    type = type.lower()
    none = data[data[type]=='None']
    mild = data[data[type]=='Mild']
    moderate = data[data[type]=='Moderate']
    severe = data[data[type]=='Severe']

with tes2:
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    anova_result=stat.f_oneway(none['rate'],mild['rate'],moderate['rate'],severe['rate'])
    p_value = anova_result.pvalue
    st.subheader('p_value : '+ str(p_value))
    if p_value > alpha:
        st.subheader(type + ' has no effect on movie rate')
    else:
        st.subheader(type +' has effect on movie rate')



st.subheader('Movie Recommendation by Genre')
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


st.subheader('Machine Learning')

data.drop('url_id', inplace=True, axis=1)
data.drop('title', inplace=True, axis=1)
data.drop('genre', inplace=True, axis=1)
data.drop('certificate', inplace=True, axis=1)
data.drop('director', inplace=True, axis=1)
data.drop('star', inplace=True, axis=1)
data.drop('aspect_ratio', inplace=True, axis=1)
data.drop('url_image', inplace=True, axis=1)
data.drop('color', inplace=True, axis=1)
data.drop('plot', inplace=True, axis=1)
data.drop('opening_weekend_gross_us_canada', inplace=True, axis=1)
data.drop('gross_us_canada', inplace=True, axis=1)
data.drop('total_nominations', inplace=True, axis=1)

data = data.dropna().reset_index(drop=True)

cols = ['nudity', 'violence', 'profanity', 'alcohol', 'frightening']
data[cols] = data[cols].replace({'None':0, 'Mild':1, 'Moderate':2, 'Severe':3, 'No Rate':0})

rate=data

feature = rate.drop(columns='rate')
target = rate['rate']

feature_rate_pretrain, feature_rate_test, target_rate_pretrain, target_rate_test = train_test_split(feature, target, test_size=0.20, random_state=42)

feature_rate_train, feature_rate_validation, target_rate_train, target_rate_validation = train_test_split(feature_rate_pretrain, target_rate_pretrain, test_size=0.20, random_state=42)



X_rate_train = feature_rate_train.to_numpy()
y_rate_train = target_rate_train.to_numpy()
y_rate_train = y_rate_train.reshape(len(y_rate_train),)

alpha = st.radio('Alpha', (0.01, 0.1, 1, 10, 100), horizontal=True)

ridge_reg = Ridge(alpha=alpha, random_state=42)
ridge_reg.fit(X_rate_train, y_rate_train)

lasso_reg = Lasso(alpha=alpha, random_state=42)
lasso_reg.fit(X_rate_train, y_rate_train)

X_rate_validation = feature_rate_validation.to_numpy()
y_rate_validation = target_rate_validation.to_numpy()
y_rate_validation = y_rate_validation.reshape(len(y_rate_validation),)
mac1, mac2 = st.columns([1,1])
with mac1:
    y_predict_validation_ridge = ridge_reg.predict(X_rate_validation)
    rmse_ridge = np.sqrt(mean_squared_error(y_rate_validation,y_predict_validation_ridge))
    st.write(f'RMSE of Ridge regression model with alpha = {alpha} is {rmse_ridge}')

    y_predict_validation_lasso = lasso_reg.predict(X_rate_validation)
    rmse_lasso = np.sqrt(mean_squared_error(y_rate_validation,y_predict_validation_lasso))
    st.write(f'RMSE of Lasso regression model with alpha = {alpha} is {rmse_lasso}')

    y_predict_train_ridge = ridge_reg.predict(X_rate_train)
    y_predict_train_lasso = lasso_reg.predict(X_rate_train)

with mac2:
    answer_ridge = st.selectbox('Which one best alpha for Ridge Regression:', ('',0.01,0.1,1,10,100), index=0)
    if answer_ridge == 0.01:
        'You Are Right. It has lowest RMSE'
    elif answer_ridge == '':
        ''
    else:
        'Sorry'

    answer_lasso = st.selectbox('Which one best alpha for Lasso Regression:', ('',0.01,0.1,1,10,100), index=0)
    if answer_lasso == 0.01:
        'You Are Right. It has lowest RMSE'
        if alpha == 0.01:
            st.subheader('Result for the best alpha')
            st.write('R-squared for Ridge Regression is {}'.format(r2_score(y_rate_train, y_predict_train_ridge)))
            st.write('R-squared for Lasso Regression is {}'.format(r2_score(y_rate_train, y_predict_train_lasso)))
        else:
            st.subheader('Please change alpha to the best option')
    elif answer_lasso == '':
        ''
    else:
        'Sorry'
        if alpha == 0.01:
            st.subheader('Result for the best alpha')
            st.write('R-squared for Ridge Regression is {}'.format(r2_score(y_rate_train, y_predict_train_ridge)))
            st.write('R-squared for Lasso Regression is {}'.format(r2_score(y_rate_train, y_predict_train_lasso)))
        else:
            st.subheader('Please change alpha to the best option')

st.subheader('These are result for another regressor :')
st.write('You can see by yourself')
def rsqr_score(test, pred):  
    r2_ = r2_score(test, pred)
    return r2_

# RMSE
def rmse_score(test, pred):
    
    rmse_ = np.sqrt(mean_squared_error(test, pred))
    return rmse_

# Print the scores
def print_score(test, pred):
    
    st.write(f"- Regressor: {regr.__class__.__name__}")
    st.write(f"R²: {rsqr_score(test, pred)}")
    st.write(f"RMSE: {rmse_score(test, pred)}\n")

X = rate.drop('rate', axis=1)
y = rate['rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,shuffle=True)

elastic = ElasticNet(alpha=0.01)
svr = SVR()
rdf = RandomForestRegressor()
xgboost = XGBRegressor()
lgbm = LGBMRegressor()

# Train models on X_train and y_train
for regr in [elastic, svr, rdf, xgboost, lgbm]:
    # fit the corresponding model
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    # Print the defined metrics above for each classifier
    print_score(y_test, y_pred)

