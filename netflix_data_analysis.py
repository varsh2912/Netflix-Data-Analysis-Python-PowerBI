import numpy as np
import pandas as pd

#Reading data from csv file  
data=pd.read_csv("netflix_movies_detailed_up_to_2025.csv")
print(data)

#converting data into a dataframe
df=pd.DataFrame(data)
print(df)

#basic info of the dataset
print(df.head())
print(df.tail(10))
print(df.describe())
print(df.info())
print(df.shape)
print(df.dtypes)
#print(df.size)
print(df[['title','cast','country','cast','release_year']])

print(df.columns)
df.drop('duration',axis=1,inplace=True)
print(df)
print(df.columns)
print(df.shape)


#handling missing values
print(df.isnull().sum())

#filling missing values with unkonwn 
columns = ['director', 'cast', 'country', 'genres', 'description']
df[columns] = df[columns].fillna('Unknown')
print(df)
print(df.isnull().sum())

#check for duplicates values
duplicate_rows=df.duplicated().sum()
print(f"Duplicated rows are {duplicate_rows}")

print(df['rating'].dtype)


#removing space from categorical data
categorical_columns = [
    'type',
    'director',
    'cast',
    'country',
    'genres',
    'language'
]

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()

print(df.head(5))


#changing column name
df.rename(columns={'date_added': 'Date_time'} ,inplace=True)
print(df)
print(df.columns)

#check minimum  year and  maximum year 
print(df['release_year'].min())
print(df['release_year'].max())


#creating new_columns
df['profit'] = df['revenue'] - df['budget']
print(df.columns)


#EDA Analysis

#1.How many Movies are there?
content_counts=df['type'].value_counts()
print(f"The no of movies is:{content_counts}")

# Which languages have the most content?
most_content=df['language'].value_counts().head()
print(f"Language have the most contents are:,{most_content}")

#Which rating category has the most content?
rated_category=df['rating'].value_counts()
print(f"the rating category have the most contents are :{rated_category}")


#Content type by language
con_lang=pd.crosstab(df['type'], df['language'])
print(f"Content type by language{con_lang}")

#Revenue by type
rev=df.groupby('type')['revenue'].sum()
print(f"Revenue by type is: {rev}")

 #Which genre has highest average popularity
print(df.groupby('genres')['popularity'].mean().sort_values(ascending=False).head(10))

#Which language has the highest average rating?
avg_rating=df.groupby('language')['rating'].mean().head(1)
print(f"Language has the average rating:{avg_rating}")


#Which genres has the highest average popularity?
avg_popularity=df.groupby('genres')['popularity'].mean().sort_values(ascending=False).head(10)
print(f"Genre having highest average popularity:{avg_popularity}")

print(df.columns)

#What are the highest-rated titles?
High_rate_title=df.groupby('title')['rating'].max().head(1)
print(High_rate_title)


#unique languages name
print(df['language'].unique())

# no of unique languages
print(" No of Unique languages :")
print(df['language'].nunique())


#Which genres are most common?
freq_genres=df['genres'].value_counts().head(10)
print(f" The most commonly used genre:{freq_genres}")

#Which genres have the highest revenue?
genre_revenue=df.groupby('genres')['revenue'].sum().sort_values(ascending=False).head(5)
print(f" Highest revenue by the genre:,{genre_revenue}")


print(df[['Date_time']])



#Which country produces the most content?
print("The country producing the most content:")
print(df['country'].value_counts().sort_values(ascending=False).head(2))

#Financial Analysis

#Which titles generated the highest profit?
High_profit=df.groupby('title')['profit'].sum().sort_values(ascending=False).head(10)
print(f"Highest profit genrated by the titles is:{High_profit}")

#is there reltionship beetween budget and revenue?
correlation=df['budget'].corr(df['revenue'])
print(f"Relationship between the budget and revenue is:{correlation}")
#yes  weak positive correlation as budget is incresing revenue is slowly incresing




