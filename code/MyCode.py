# start for check email spaming

import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer , WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# وارد کردن مجموعه داده 
email_data_set = pd.read_csv("../data/spam.csv")

# تغییر تیتر ستون ها به  حالت کوچک کلمات :  ( lower case )
email_data_set.columns = [a.lower() for a in email_data_set.columns]


# تحلیل های آماری دیتا
print("all stats for data : ",email_data_set.describe() , "\n")
print("all information for data : ",email_data_set.info(verbose=False) , '\n')   
print("calculate data with differente category : " , email_data_set.value_counts() , '\n') 
category_state = email_data_set['category'].value_counts() 
# plt.figure(figsize=(5 , 4))
# plt.bar(category_state.index , category_state)
# plt.show()
print("find all spam data type : " , email_data_set.query('category == "spam"') , '\n') 


# تقسیم ستون های به متغیر های هدف 
x = email_data_set.drop(columns='category') #  (main column)
y = email_data_set.drop(columns='message') # (target column)


# هندل کردن دیتا های نادیده گرفته شده 
chckNAN = email_data_set.isna().sum()
print("get count null : ", chckNAN , '\n')
data_imputing = SimpleImputer(strategy='most_frequent' , missing_values='str')
data_imputing.fit_transform(x) 


#  به عددی categorical تبدیل داده های 
target_encode = OrdinalEncoder(categories='auto')
y = target_encode.fit_transform(y)


# توکنایز کردن و تکه تکه کردن پیام ها
tokenize_x = [word_tokenize(a.lower()) for a in x['message']]

# stop words حذف حروف 
stop_words = stopwords.words('english')  
clean_x = [a for a in tokenize_x if a not in stop_words]
print("clean Words : " , clean_x[0:1] , '\n')
print("dirt Words:" , tokenize_x[0:1], '\n')

# برگرداندن توکن ها به ریشه کلمه ای خود
word_stemmer = PorterStemmer()
word_lemmatizer = WordNetLemmatizer()
io_stem = [word_stemmer.stem(str(word)) for word in clean_x] 
print("Stemming Words:" , io_stem[0:1], '\n')
io_lemma = [word_lemmatizer.lemmatize(str(word)) for word in io_stem] 
print("Lemmatizing Words:" , io_lemma[0:1], '\n')
x['message'] = io_lemma


# کلمات map محاسبه تکرار کلمات و 
cv_model = CountVectorizer()
x = cv_model.fit_transform(x['message'])
print("\n Count Vectorize Words : ", x , '\n')

# tf_idf_model = TfidfVectorizer()
# x = tf_idf_model.fit_transform(x['message'])
# print("\n Tf-idf Vectorize Words : ", x , '\n')

# تغییر مقدار هدف
y = email_data_set['category']
# تقسیم دیتا به دیتاهای آموزشی و تستی
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=.2)


# پیاده سازی الگوریتم یادگیری ماشین  - نظارت شده 
email_model = LogisticRegression()
email_model.fit(x_train ,y_train)
email_model.predict(x_test)

# بررسی خطا و متریک های کیفیت الگوریتم
accuracy_score_lr = email_model.score(x_test, y_test)
print("\n Score Logistic Regression Model : ", accuracy_score_lr , '\n') 
