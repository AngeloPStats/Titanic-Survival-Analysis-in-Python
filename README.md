# Titanic-Survival-Analysis-in-Python
In this Python-PyCharm analysis, I utilize Titanic survival data, showcasing pie charts, training and testing data, alongside a linear model prediction

# These packages will be used in our example
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm

#Dataset is available here  https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html

titanic = pd.read_csv('titanic.csv')

# Survived column moved to last position

titanic = titanic.assign(survived=titanic['Survived'])
titanic.drop(titanic.columns[0], axis=1, inplace=True)

# Our dataset is set for analysis. Let's examine its context and stats

print(titanic.info())
print(titanic.describe())

Sex = titanic["Sex"]

# Pie Chart Analysis of Male and Female Passengers

male = titanic[titanic["Sex"] == 'male']
female = titanic[titanic["Sex"] == 'female']

sizes = [len(male), len(female)]
labels = 'Males', 'Females'
plt.pie(sizes, labels=labels, autopct='%1.1f%%')

plt.title('Men and women aboard the Titanic')
plt.axis('equal')

plt.show()

# Pie chart for Survivors: Men and Women

s_male = titanic[(titanic["Sex"] == 'male') & (titanic["survived"] == 1)]

s_female = titanic[(titanic["Sex"] == 'female') & (titanic["survived"] == 1)]

sizes = [len(s_male), len(s_female)]
labels = 'Males', 'Females'
plt.pie(sizes, labels=labels, autopct='%1.1f%%')

plt.title('Survivors: The Men and Women')
plt.axis('equal')

plt.show()

# Dividing our dataset in 80% training & 20% testing

x = titanic.iloc[:, [0, 1, 2, 3, 4, 5, 6]]
y = titanic.iloc[:, [7]]

print(x.columns.tolist())
print(y.columns.tolist())

x_train, x_test, y_train, y_test = train_test_split(x, y,
train_size=0.8,
test_size=0.2,
random_state=2024)


# Predicting: Introducing Two Modifications
# Initially, we will discard the "Name" column.
# Subsequently, we will retrieve values for the "Sex" column.

x_train = x_train.drop("Name", axis=1)
x_test = x_test.drop("Name", axis=1)

# Sex

mf = {"male": 0, "female": 1}
titanic = [x_train, x_test]

for df in titanic:
    df['Sex'] = df['Sex'].map(mf)


model=(lm().fit(x_train, y_train))
predictions=model.predict(x_test)

plt.scatter(y_test, predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()
