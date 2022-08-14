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
    st.success('You Are Right')
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
    st.error('Sorry')
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
    st.error('Sorry')
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
    st.info('Please pick one first')


st.subheader('Hypothesis Testing using ANOVA')

tes1, tes2 = st.columns(2)
with tes1:
    alpha = st.slider('Alpha', min_value=0.01, max_value=0.1, value=0.05, step=0.01)
    type = st.radio('Parental Guide', ('Violence', 'Nudity', 'Profanity', 'Alcohol', 'Frightening'))
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
        st.success('You Are Right. It has lowest RMSE')
    elif answer_ridge == '':
        ''
    else:
        st.error('Sorry')

    answer_lasso = st.selectbox('Which one best alpha for Lasso Regression:', ('',0.01,0.1,1,10,100), index=0)
    if answer_lasso == 0.01:
        st.success('You Are Right. It has lowest RMSE')
        if alpha == 0.01:
            st.subheader('Result for the best alpha')
            st.write('R-squared for Ridge Regression is {}'.format(r2_score(y_rate_train, y_predict_train_ridge)))
            st.write('R-squared for Lasso Regression is {}'.format(r2_score(y_rate_train, y_predict_train_lasso)))
        else:
            st.subheader('Please change alpha to the best option')
    elif answer_lasso == '':
        ''
    else:
        st.error('Sorry')
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

model1, model2, model3, model4, model5 = st.columns(5)
with model1:
    st.write("- Regressor : Elastic Net")
    elastic.fit(X_train, y_train)
    y_pred = elastic.predict(X_test)
    print_score(y_test, y_pred)
with model2:
    st.write("- Regressor : SVR")
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    print_score(y_test, y_pred)
with model3:
    st.write("- Regressor : Random Forest")
    rdf.fit(X_train, y_train)
    y_pred = rdf.predict(X_test)
    print_score(y_test, y_pred)
with model4:
    st.write("- Regressor : XGBoost")
    xgboost.fit(X_train, y_train)
    y_pred = xgboost.predict(X_test)
    print_score(y_test, y_pred)
with model5:
    st.write("- Regressor : Light GBM")
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    print_score(y_test, y_pred)



st.subheader('Prediction')
st.write('\n')
st.write('\n')
st.write('Please specify the variable :')
data_modelling = pd.read_csv('data_for_modelling.txt', sep='\t')
data_actor = pd.read_csv('actor rank.csv', sep='\t')

data_actor['actor'] = data_actor['actor'].str.lower()

cols = ['nudity', 'violence', 'profanity', 'alcohol', 'frightening']
data_modelling[cols] = data_modelling[cols].replace({'None':0, 'Mild':1, 'Moderate':2, 'Severe':3, 'No Rate':0})

data_modelling = data_modelling.dropna()

# Modelling
feature = data_modelling.drop(columns='worldwide_gross')
target = data_modelling['worldwide_gross']

from sklearn.model_selection import train_test_split

X = data_modelling.drop('worldwide_gross', axis=1)
y = data_modelling['worldwide_gross']

scaling = StandardScaler()
scaling.fit(X)
X = scaling.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,shuffle=True)

ridge = Ridge(alpha=0.001)
lasso = Lasso(alpha=0.001)
elastic = ElasticNet(alpha=0.001)
svr = SVR()
rdf = RandomForestRegressor()
xgboost = XGBRegressor()
lgbm = LGBMRegressor()

@st.cache(suppress_st_warning = True)
def rsqr_score(test, pred):
    r2_ = r2_score(test, pred)
    return r2_


@st.cache(suppress_st_warning = True)
def rmse_score(test, pred):
    
    rmse_ = np.sqrt(mean_squared_error(test, pred))
    return rmse_

@st.cache(suppress_st_warning = True)
def print_score(test, pred):
    
    print(f"- Regressor: {lgbm.__class__.__name__}")
    print(f"R²: {rsqr_score(test, pred)}")
    print(f"RMSE: {rmse_score(test, pred)}\n")

# Train models on X_train and y_train
for regr in [ridge, lasso, elastic, svr, rdf, xgboost, lgbm]:
    # fit the corresponding model
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    # Print the defined metrics above for each classifier
    print_score(y_test, y_pred)

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
if a1_value:
    a1_value = a1_value[0]
else:
    a1_value = 0
a2_value = data_actor['rank_actor'].loc[data_actor['actor'].str.contains(a2)].to_list()
if a2_value:
    a2_value = a2_value[0]
else:
    a2_value = 0
a3_value = data_actor['rank_actor'].loc[data_actor['actor'].str.contains(a3)].to_list()
if a3_value:
    a3_value = a3_value[0]
else:
    a3_value = 0
a4_value = data_actor['rank_actor'].loc[data_actor['actor'].str.contains(a4)].to_list()
if a4_value:
    a4_value = a4_value[0]
else:
    a4_value = 0
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
                'is_Adult':0, 'is_Adventure':0, 'is_Animation':0, 'is_Biography':0,
                'is_Comedy':1, 'is_Crime':1, 'is_Drama':0, 'is_Family':0,
                'is_Fantasy':0, 'is_Film-Noir':0, 'is_Horror':0, 'is_History':0,
                'is_Music':0, 'is_Mystery':0, 'is_Romance':0, 'is_Sci-Fi':0,
                'is_Sport':0, 'is_Thriller':0, 'is_War':0, 'is_Western':0,
                'total_star_score':total_value}

data_predict = pd.DataFrame(data_predict, index=[0])

st.write('\n')
st.write('\n')
st.write('\n')

res1, res2, res3 = st.columns(3)
with res1:
    result1 = rdf.predict(data_predict)
    hasil1 = int(result1[0])
    hasil1 = '$ ' + numerize.numerize(hasil1)
    st.write('**Prediksi menggunakan Random Forest :**')
    st.info(hasil1)

with res2:
    result2 = xgboost.predict(data_predict)
    hasil2 = int(result2[0])
    hasil2 = '$ ' + numerize.numerize(hasil2)
    st.write('**Prediksi menggunakan XGBoost :**')
    st.info(hasil2)

with res3:
    result3 = lgbm.predict(data_predict)
    hasil3 = int(result3[0])
    hasil3 = '$ ' + numerize.numerize(hasil3)
    st.write('**Prediksi menggunakan Light GBM :**')
    st.info(hasil3)
