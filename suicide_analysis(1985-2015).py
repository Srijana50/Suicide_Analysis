import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
import seaborn as sb
from sklearn import metrics
from linearmodels.panel import PooledOLS
from linearmodels.panel import compare

original= pd.read_csv("D:\PythonProject\sucide\master.csv")
masterdata= pd.read_csv("D:\PythonProject\sucide\master.csv")

# to knoww the shape of master dataset
print(np.shape(masterdata))
print(masterdata.head(5)) #printing 5 rows of dataset
#code to get data type of columns in masterdata
print(masterdata.dtypes)

# to check whether there is null value
masterdata.isnull().any()
#HDI for year has empty data. Lets see how many data is null
masterdata.isnull().sum()
#19456 rows are empty in HDI for year so i am droping this
masterdata = masterdata.drop(['HDI_for_year'], axis=1)

#removing countries which doesnot have data for all the years (6 age * 2 age i.e 12 rows for each year)
groupeddata=masterdata.groupby('country_year')
value = groupeddata.size()

#print each row and column of value series
for n, s in value.iteritems():
    print(n,s)

# for more than 50 countries there is no data for year 2016. So we will drop data for 2016
# for some countries there were data for less than 10 years. We will drop the country for which there is less than 5 years of data

#storing data expect for year ==2016
masterdata = masterdata[masterdata.year !=2016]
np.shape(masterdata)

# method to get unique value of the data in data set
def uniquevalue(datalist):
    x = np.array(datalist)
    data_list= np.unique(x)
    return data_list

year=masterdata.year
country = masterdata.country
# calling uniquevalue method to get unique country
UniqueCountry=uniquevalue(country)

#Dropping country for less than 10 years of data
def getCountryToDelete(mainData, country):
    countryToDrop=[]
    groupbycountry= mainData.groupby(country)
    for countries, count in groupbycountry.size().iteritems():
        if count/12 <10:
            countryToDrop.append(countries)
    return countryToDrop

droppedcountry=getCountryToDelete(masterdata, country)
droppedcountry
# drop the countries from the masterdata
for c in droppedcountry:
    masterdata = masterdata[masterdata.country != c]

# to check whether the countries has been deleted or not
np.shape(masterdata)

#assigning unique country value
UniqueCountry = uniquevalue(masterdata.country)
np.shape(UniqueCountry)

#renaming columns
masterdata=masterdata.rename(columns={'suicides/100k pop':'suicideper100k','gdp_for_year ($) ':'gdp_for_year','gdp_per_capita ($)':'gdp_per_capita'})

#histogram for each numeric data in masterdata set
masterdata.hist(bins=50, figsize=(20, 15))
plt.show()

#creating array for each column
population=np.array(masterdata.population)
sucicidesper100K=masterdata['suicideper100k']
#removing commas
masterdata['gdp_for_year'] = masterdata['gdp_for_year'].str.replace(',', '')
gdpForYear=masterdata['gdp_for_year']
gdpForYear=gdpForYear.astype(float)

gdpPerCapita=np.array(masterdata.gdp_per_capita)
generation=np.array(masterdata.generation)
uniqueyear= np.array(uniquevalue(masterdata.year))

## defining different methods
# define method to get sucide rate
def getfilteredData(mainData, country,gender,age):
    filetereddataset=[]
    filetereddataset =mainData[(mainData.country==country)  & (mainData.sex==gender) & (mainData.age==age)]
    return filetereddataset


#method to set value = 0 if male and value =1 if female
def getgender(gen):
    sex=[]
    for i in gen:
        if(i=="male"):
            i=0
            sex.append(i)
        if(i=="female"):
            i=1
            sex.append(i)
    return sex
# for n variable we need n-1 dummy variable


# looking into general statistics
print(masterdata.describe())

## to see the row of data with min and max sucicidesper100K
masterdata[masterdata.suicideper100k==224.97]
masterdata[masterdata.suicideper100k==12.94]


#checking the number of data for each gender
sb.countplot(masterdata.sex)
plt.title("Data according to Gender")
plt.savefig('GenderCount.png')
plt.show()

#checking the number of data for each generation
plt.figure(figsize=(10,10))
sb.countplot(masterdata.generation)
plt.title('Data according to Generation')
plt.xticks(rotation=45)
plt.savefig('GenerationCount.png')
plt.show()

#checking the number of data countrywise
plt.figure(figsize=(20,50))
sb.set_context("paper", 2.5, {"lines.linewidth": 4})
sb.countplot(y=masterdata['country'],label='count')
plt.title('Data according to Country')
plt.savefig('DataofCountry.png')
plt.show()


#checking the number of data in each year
plt.figure(figsize=(30,10))
sb.set_context("paper", 2.0, {"lines.linewidth": 4})
sb.countplot(masterdata['year'],label='count')
plt.title("YearCount")
plt.savefig('Year.png')
plt.show()


#suicideper100k by gender
plt.figure(figsize=(30,10))
sb.set_context("paper", 2.0, {"lines.linewidth": 4})
sb.barplot(data=masterdata[masterdata['year']>1999],x='year',y='suicideper100k',hue='sex')
plt.title('suicideper100k by gender')
plt.savefig('suicideper100k by gender.png')
plt.legend(loc = 'upper right')
plt.show()


#suicideper100k by ageGroup for 15 years
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[masterdata['year']>1999],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k by agegroup from 1999 to 2015')
plt.legend(loc = 'upper right')
plt.savefig('suicideper100k by agegroup.png')
plt.show()

#suicideper100k by generation for 15 years
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[masterdata['year']>1999],x='year',y='suicideper100k',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.title('suicideper100k by Generation from 1999 to 2015')
plt.legend(loc = 'upper right')
plt.savefig('suicideper100k by generation.png')
plt.show()


#summing population on the basis of country
country_population=[]

for country in UniqueCountry:
    country_population.append(sum(masterdata[(masterdata.country==country)].population))

country_population=pd.DataFrame(country_population,columns=['population'])
country=pd.DataFrame(UniqueCountry,columns=['country'])
countryPopulation=pd.concat([country_population,country],axis=1)
countryPopulation=countryPopulation.sort_values(by='population',ascending=False)

#plotting bar graph of top 15 countries with mazimum population
sb.barplot(y=countryPopulation.country[:15],x=countryPopulation.population[:15])
plt.title('15 Countries with maximum population after summing population from 1985 to 2015')
plt.savefig('15 Countries with maximum population after summing population from 1985 to 2015.png')
plt.show()


#minimum population in country
countryPopulation=countryPopulation.sort_values(by='population',ascending=True)
#plotting bar graph of top 15 countries with minimum population
sb.barplot(y=countryPopulation.country[:15],x=countryPopulation.population[:15])
plt.title('15 Countries with minimum population after summing population from 1985 to 2015')
plt.savefig('15 Countries with minimum population after summing population from 1985 to 2015.png')
plt.show()


#summing suicides number by country and studying minimum and maximum sicide
suicidesNo=[]
for country in UniqueCountry:
    suicidesNo.append(sum(masterdata[masterdata.country==country].suicides_no))

suicidesNo=pd.DataFrame(suicidesNo,columns=['suicides_no'])
country=pd.DataFrame(UniqueCountry,columns=['country'])
data_suicide_country=pd.concat([suicidesNo,country],axis=1)
data_suicide_country=data_suicide_country.sort_values(by='suicides_no',ascending=False)

#plotting bar graph of top 15 countries with mazimum suicide number
sb.barplot(y=data_suicide_country.country[:15],x=data_suicide_country.suicides_no[:15])
plt.title('15 Countries with maximum suicide number after summing suicide number from 1985 to 2015')
plt.savefig('15 Countries with maximum suicide number after summing suicide number from 1985 to 2015.png')
plt.show()


#minimum suicide number
data_suicide_country=data_suicide_country.sort_values(by='suicides_no',ascending=True)
#plotting bar graph of top 15 countries with minimum suicide number
sb.barplot(y=data_suicide_country.country[:15],x=data_suicide_country.suicides_no[:15])
plt.title('15 Countries with minimum suicide number after summing suicide number from 1985 to 2015')
plt.savefig('15 Countries with minimum suicide number after summing suicide number from 1985 to 2015.png')
plt.show()


#suicideper100k by USA for 15 years for male
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='United States')& (masterdata.year>1999) & (masterdata.sex=='male')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of male for U.S. from 2000 to 2015')
plt.legend(loc = 'upper right')
plt.savefig('suicideper100k by male for U.S. from 2000 to 2015.png')
plt.show()

#suicideper100k by USA for 15 years for female
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='United States')& (masterdata.year>1999) & (masterdata.sex=='female')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of female for U.S from 2000 to 2015')
plt.savefig('suicideper100k by female for U.S. from 2000 to 2015.png')
plt.legend(loc = 'upper right')
plt.show()

#studying about gdp per year and gdp_per capita it is same for all age group of USA
plt.figure(figsize = (50,10))
plt.plot(uniqueyear, getfilteredData(masterdata,'United States','male','5-14 years').gdp_for_year, color = "blue")
plt.title('GDP for year in USA between 1985 to 2015')
plt.ylabel('GDP per year')
plt.xlabel('Year')
plt.savefig('GDP for year in USA.png')
plt.show()


#studying about gdp per capita of USA
plt.figure(figsize = (50,10))
plt.plot(uniqueyear, getfilteredData(masterdata,'United States','male','5-14 years').gdp_per_capita , color = "blue")
plt.title('GDP per capita in USA between 1985 to 2015')
plt.ylabel('GDP per capita')
plt.xlabel('Year')
plt.savefig('GDP per capita in USA between 1985 to 2015 .png')
plt.show()


# study about UK
#suicideper100k of UK for 15 years for male
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='United Kingdom')& (masterdata.year>1999) & (masterdata.sex=='male')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of male in U.K from 2000 to 2015')
plt.legend(loc = 'upper right')
plt.savefig('suicideper100k of male in U.K from 2000 to 2015.png')
plt.show()

#suicideper100k by UK for 15 years for female
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='United Kingdom')& (masterdata.year>1999) & (masterdata.sex=='female')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of female in U.K from 2000 to 2015')
plt.legend(loc = 'upper right')
plt.savefig('suicideper100k of female in U.K from 1999 to 2015.png')
plt.show()

# study about gdp for year of uk
plt.figure(figsize = (50,10))
plt.plot(uniqueyear, getfilteredData(masterdata,'United Kingdom','male','5-14 years').gdp_for_year, color = "blue")
plt.title('gdp_for_year in UK between 1985 to 2015')
plt.ylabel('gdp_for_year')
plt.xlabel('Year')
plt.savefig('gdp_per_year in UK between 1985 to 2015.png')
plt.show()

# study about gdpPerCapita
plt.figure(figsize = (50,10))
plt.plot(uniqueyear, getfilteredData(masterdata,'United Kingdom','male','5-14 years').gdp_per_capita, color = "blue")
plt.title('gdp_per_Capita in UK between 1985 to 2015')
plt.ylabel('GDP_per_Capita')
plt.xlabel('Year')
plt.savefig('gdp_per_capita in UK between 1985 to 2015.png')
plt.show()

#study about Republic of Korea
#suicideper100k by Republic of Korea for 15 years for male
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='Republic of Korea')& (masterdata.year>1999) & (masterdata.sex=='male')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of male in Republic of Korea from 2000 to 2015')
plt.legend(loc = 'upper right')
plt.savefig('suicideper100k of male in Republic of Korea from 2000 to 2015.png')
plt.show()

#suicideper100k by Republic of Korea for 15 years for female
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='Republic of Korea')& (masterdata.year>1999) & (masterdata.sex=='female')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of female in Republic of Korea from 2000 to 2015')
plt.legend(loc = 'upper right')
plt.savefig('suicideper100k of female in Republic of Korea from 2000 to 2015.png')
plt.show()


#study about gdpperyear Republic of Korea
plt.figure(figsize = (50,10))
plt.plot(uniqueyear, getfilteredData(masterdata,'Republic of Korea','male','5-14 years').gdp_for_year, color = "blue")
plt.title('gdp_for_year in Republic of Korea between 1985 to 2015')
plt.ylabel('gdp_for_year')
plt.xlabel('Year')
plt.savefig('gdp_for_year in Republic of Korea between 1985 to 2015.png')
plt.show()


#study about gdpPerCapita Republic of Korea
plt.figure(figsize = (50,10))
plt.plot(uniqueyear, getfilteredData(masterdata,'Republic of Korea','male','5-14 years').gdp_per_capita, color = "blue")
plt.title('gdp_per_capita in Republic of Korea between 1985 to 2015')
plt.ylabel('gdp_per_capita')
plt.xlabel('Year')
plt.savefig('gdp_per_capita in Republic of Korea between 1985 to 2015.png')
plt.show()


#study about singapore
#suicideper100k by singapore for 15 years for male
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='Singapore')& (masterdata.year>1999) & (masterdata.sex=='male')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of male in Singapore from 2000 to 2015')
plt.legend(loc = 'upper right')
plt.savefig('suicideper100k of male in Singapore from 2000 to 2015.png')
plt.show()

#suicideper100k by singapore for 15 years for female
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='Singapore')& (masterdata.year>1999) & (masterdata.sex=='female')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of female in Singapore from 2000 to 2015')
plt.savefig('suicideper100k of female in Singapore from 2000 to 2015.png')
plt.legend(loc = 'upper right')
plt.show()


#study about gdp_per_year singapore
# Gdpfor year is sam for all gender and age group
plt.figure(figsize = (50,10))
plt.plot(uniqueyear, getfilteredData(masterdata,'Singapore','male','15-24 years').gdp_for_year, color = "blue")
plt.title('gdp_for_year in Singapore between 1985 to 2015')
plt.ylabel('gdp_for_year')
plt.xlabel('Year')
plt.savefig('gdp_for_year in Singapore between 1985 to 2015.png')
plt.show()

#study about gdp_per_capita singapore
plt.figure(figsize = (50,10))
plt.plot(uniqueyear, getfilteredData(masterdata,'Singapore','male','15-24 years').gdp_per_capita, color = "blue")
plt.title('gdp_per_capita in Singapore between 1985 to 2015')
plt.ylabel('gdp_per_capita')
plt.xlabel('Year')
plt.savefig('gdp_per_capita in Singapore between 1985 to 2015.png')
plt.show()


# Norway doesnot have data for 1985 do we will create new array for year
yearForNorway=uniqueyear[uniqueyear!=1985]
np.shape(uniqueyear)

#study about Norway
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='Norway')& (masterdata.year>1999) & (masterdata.sex=='male')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of male in Norway from 2000 to 2015')
plt.legend(loc = 'upper right')
plt.savefig('suicideper100k of male in Norway from 2000 to 2015.png')
plt.show()

#suicideper100k by singapore for 15 years for female
plt.figure(figsize=(20,10))
sb.barplot(data=masterdata[(masterdata.country=='Norway')& (masterdata.year>1999) & (masterdata.sex=='female')],x='year',y='suicideper100k',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.title('suicideper100k of female in Norway from 2000 to 2015')
plt.savefig('suicideper100k of female in Norway from 2000 to 2015.png')
plt.legend(loc = 'upper right')
plt.show()


#study about gdp_per_year Singapore
# Gdpfor year is same for all gender and age group
plt.figure(figsize = (50,10))
plt.plot(yearForNorway, getfilteredData(masterdata,'Norway','male','15-24 years').gdp_for_year, color = "blue")
plt.title('gdp_for_year in Norway between 1986 to 2015')
plt.ylabel('gdp_for_year')
plt.xlabel('Year')
plt.savefig('gdp_for_year in Norway between 1986 to 2015.png')
plt.show()


#study about gdp_per_year Norway
plt.figure(figsize = (50,10))
plt.plot(yearForNorway, getfilteredData(masterdata,'Norway','male','5-14 years').gdp_for_year, color = "blue")
plt.title('gdp_for_year in Norway between 1986 to 2015')
plt.ylabel('gdp_for_year')
plt.xlabel('Year')
plt.savefig('gdp_for_year in Norway between 1986 to 2015.png')
plt.show()

#study about gdp_per_capita for Norway
plt.figure(figsize = (50,10))
plt.plot(yearForNorway, getfilteredData(masterdata,'Norway','male','5-14 years').gdp_per_capita, color = "blue")
plt.title('gdp_per_capita in Norway between 1986 to 2015')
plt.ylabel('gdp_per_capita')
plt.xlabel('Year')
plt.savefig('gdp_per_capita in Norway between 1986 to 2015.png')
plt.show()


#printing correlation between the numerical column in dataset
print(masterdata.corr())

#gdp_for_year in masterdata set has comma so it is not shown in correlation matrix
#dataframe data with numerical value column of master data set and gdpperyear after changing datatype
data=np.array([masterdata.year,masterdata.suicides_no,masterdata.population,masterdata.suicideper100k,masterdata.gdp_per_capita,gdpForYear])
data=data.T
my_df = pd.DataFrame(data=data, columns=['year','suicide_no','population','suicideper100k','gdpPercapita','gdpPeryear'])


#correlation rounded up to 2 decimal places
print(sb.heatmap(round(my_df.corr(),2), annot=True, cmap='YlGnBu'))

np.shape(masterdata)
#seeing the distribution of suicide per 100k
sb.distplot(masterdata.suicideper100k)

#creating array gend for gender where 0 means male and 1 means female
gend=getgender(masterdata.sex)

#creating dummy variable for age
agegroupDummy=pd.get_dummies(masterdata['age'])
#getdummies method created 6 dummies but 5 dumies only need for linear regression(to avoid dummy variable trap)
agegroupDummy=agegroupDummy.drop(['75+ years'], axis='columns')
masterdata=pd.concat([masterdata,agegroupDummy],axis=1)
masterdata.head(5)

#creating dummy variable for generation
generationDummy=pd.get_dummies(masterdata['generation'])
generationDummy= generationDummy.drop(['Silent'], axis='columns')
masterdata=pd.concat([masterdata,generationDummy],axis=1)
masterdata.head(5)


#continent
#reading contient.csv file
continent= pd.read_csv("D:\PythonProject\sucide\continent.csv")
Asia=continent['Asia'].str.strip()
Africa=continent['Africa'].str.strip()
NAmerica=continent['N.America'].str.strip()
SAmerica=continent['S.America'].str.strip()
Oceania=continent['Oceania'].str.strip()
Europe=continent['Europe'].str.strip()

#creating world dictionary
world={'Asia':Asia,'Africa':Africa,'NAmerica':NAmerica,'SAmerica':SAmerica,'Oceania':Oceania,'Europe':Europe}

#creating columnfor continent and copying country value
masterdata['continent']=masterdata.country
con=pd.Series(masterdata['continent'].str.strip()) #.str.strip() method to remove whitespace
#con.str.lower()

#method to replace country with contient
for contient in world:
    for i in  world.get(contient):
        for index,coun in enumerate(con):
            if(i==coun):
                con[index]=contient

#setting continent value
masterdata['continent']=con

#create dummies for continent
contient_dummy= pd.get_dummies(masterdata['continent'])
contient_dummy=contient_dummy.drop(['Asia'],axis='columns')
#concate dummyvariable with masterdata
masterdata=pd.concat([masterdata,contient_dummy],axis=1)
masterdata.head(5)

# defining dummy variable
#age dummy
age5=masterdata['5-14 years']
age15=masterdata['15-24 years']
age25=masterdata['25-34 years']
age35=masterdata['35-54 years']
age55=masterdata['55-74 years']
#generation dummy
boomers=masterdata['Boomers']
GIgen=masterdata['G.I. Generation']
genX=masterdata['Generation X']
genZ=masterdata['Generation Z']
Millenials=masterdata['Millenials']

#agegenderdummy
age_gender5=age5*gend
age_gender15=age15*gend
age_gender25=age25*gend
age_gender35=age35*gend
age_gender55=age55*gend

#contientdummy
africa=masterdata['Africa']
europe=masterdata['Europe']
namerica=masterdata['NAmerica']
samerica=masterdata['SAmerica']
oceania=masterdata['Oceania']


##assigning latest value to variable from masterdataset
year= masterdata.year
country=masterdata.country
y=masterdata.suicideper100k
gdpForYear=(masterdata.gdp_for_year.astype(float))/100000    ## divide by 100k because of large data it was giving memory error while running regression
suicidenum=masterdata.suicides_no

#creating variable array with all independent variable that i will be using  in different model
variable=np.array([y,country,year,year,gend,age5,age15,age25,age35,age55,age_gender5,age_gender15,age_gender25,age_gender35,age_gender55,boomers,GIgen,genX,genZ,Millenials,population,gdpPerCapita,gdpForYear,suicidenum,africa,europe,namerica,samerica,oceania])
#variable.T is use for transpose. so that number of rows of variable y and independent variable is same.
variable=variable.T
np.shape(variable)

#creating dataframe of varible array
finaldata=pd.DataFrame(variable,columns=['y','country','year','years','gend','age5','age15','age25','age35','age55','age_gender5','age_gender15','age_gender25','age_gender35','age_gender55','boomers','GIgen','genX','genZ','Millenials','population','gdppercapita','gdpForYear','suicidenum','africa','europe','namerica','samerica','oceania'])
#setting index
finaldata.set_index(['country','year'], inplace=True)
#printing first 5 element
print(finaldata.head(5))


#model1 with out interaction term using population and gdpForYear
exog_vars=['years','gend','age5','age15','age25','age35','age55','boomers','GIgen','genX','genZ','Millenials','population','gdpForYear','africa','europe','namerica','samerica','oceania']
exog = sm.add_constant(finaldata[exog_vars])
mod1 = PooledOLS(finaldata.y, exog).fit()
print(mod1)
predictions=mod1.predict(exog)

#one empty row was in heading
finaldata.y.drop(finaldata.y.index[[1]])
residuals=predictions.predictions-finaldata.y
sb.set_style('darkgrid')
sb.distplot(residuals.tolist())

##measure error
print('Mean Absolute Error:', metrics.mean_absolute_error(finaldata.y, predictions.predictions))
print('Mean Squared Error:', metrics.mean_squared_error(finaldata.y, predictions.predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(finaldata.y, predictions.predictions)))


#model without interactionterm using gdpPerCapita instead of gdpforyear and population
exog_vars=['years','gend','age5','age15','age25','age35','age55','boomers','GIgen','genX','genZ','Millenials','gdppercapita','africa','europe','namerica','samerica','oceania']
exog = sm.add_constant(finaldata[exog_vars])
mod2 = PooledOLS(finaldata.y, exog).fit()
print(mod2)
predictions=mod2.predict(exog)

#one empty row was in heading
finaldata.y.drop(finaldata.y.index[[1]])
residuals=predictions.predictions-finaldata.y
sb.set_style('darkgrid')
sb.distplot(residuals.tolist())

##measure error
print('Mean Absolute Error:', metrics.mean_absolute_error(finaldata.y, predictions.predictions))
print('Mean Squared Error:', metrics.mean_squared_error(finaldata.y, predictions.predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(finaldata.y, predictions.predictions)))


##interaction term in model 1
exog_vars=['years','gend','age5','age15','age25','age35','age55','age_gender5','age_gender15','age_gender25','age_gender35','age_gender55','boomers','GIgen','genX','genZ','Millenials','population','gdpForYear','africa','europe','namerica','samerica','oceania']
exog = sm.add_constant(finaldata[exog_vars])
mod3 = PooledOLS(finaldata.y, exog).fit()
print(mod3)
predictions=mod3.predict(exog)

#one empty row was in heading
finaldata.y.drop(finaldata.y.index[[1]])
residuals=predictions.predictions-finaldata.y
sb.set_style('darkgrid')
sb.distplot(residuals.tolist())

##measure error
print('Mean Absolute Error:', metrics.mean_absolute_error(finaldata.y, predictions.predictions))
print('Mean Squared Error:', metrics.mean_squared_error(finaldata.y, predictions.predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(finaldata.y, predictions.predictions)))


##interaction in model2
exog_vars=['years','gend','age5','age15','age25','age35','age55','age_gender5','age_gender15','age_gender25','age_gender35','age_gender55','boomers','GIgen','genX','genZ','Millenials','gdppercapita','africa','europe','namerica','samerica','oceania']
exog = sm.add_constant(finaldata[exog_vars])
mod4 = PooledOLS(finaldata.y, exog).fit()
print(mod4)
predictions=mod4.predict(exog)

#one empty row was in heading
finaldata.y.drop(finaldata.y.index[[1]])
residuals=predictions.predictions-finaldata.y
sb.set_style('darkgrid')
sb.distplot(residuals.tolist())

##measure error
print('Mean Absolute Error:', metrics.mean_absolute_error(finaldata.y, predictions.predictions))
print('Mean Squared Error:', metrics.mean_squared_error(finaldata.y, predictions.predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(finaldata.y, predictions.predictions)))


###using suicide number , gdpForYear and population
exog_vars=['years','gend','age5','age15','age25','age35','age55','age_gender5','age_gender15','age_gender25','age_gender35','age_gender55','boomers','GIgen','genX','genZ','Millenials','suicidenum','gdpForYear','population','africa','europe','namerica','samerica','oceania']
exog = sm.add_constant(finaldata[exog_vars])
mod5 = PooledOLS(finaldata.y, exog).fit()
print(mod5)
predictions=mod5.predict(exog)

#one empty row was in heading
finaldata.y.drop(finaldata.y.index[[1]])
residuals=predictions.predictions-finaldata.y
sb.set_style('darkgrid')
sb.distplot(residuals.tolist())

##measure error
print('Mean Absolute Error:', metrics.mean_absolute_error(finaldata.y, predictions.predictions))
print('Mean Squared Error:', metrics.mean_squared_error(finaldata.y, predictions.predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(finaldata.y, predictions.predictions)))


#printing estimates and t stats of all the model.
#compare method is used for panelmodel only
print(compare({'Md1':mod1,'Md2':mod2,'Md3':mod3,'Md4':mod4,'Md5':mod5}))
