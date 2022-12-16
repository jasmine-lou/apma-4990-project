import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def lb():
  print('\n')
  print('--------')
  print('\n')


def read_data_first_time():
  files = glob.glob("./furman_data/*.xlsx")
  df = pd.DataFrame()
  for file in files:
    xls = pd.ExcelFile(file)
    df_t = pd.read_excel(xls,
                         str(file)[14:16] + ' ' + str(file)[16:18] + ' Data')
    df = df.append(df_t)
  df.to_csv('./out.csv')


def read_data():
  df = pd.read_csv('out.csv')
  airbnb = pd.read_csv('AB_NYC_2019.csv')

  return airbnb, df


def compare_neighborhoods(airbnb, furman, br):
  br_map = {
    'QN': 'Queens',
    'MN': 'Manhattan',
    'BX': 'Bronx',
    'BK': 'Brooklyn',
    'SI': 'Staten Island'
  }
  f_n = list(
    furman[furman['Community District'].str.contains(br)]['Name'].unique())
  a_n = list(airbnb[airbnb['neighbourhood_group'].str.contains(
    br_map[br])]['neighbourhood'].unique())

  lb()

  print('Matches')
  print([(x, y) for x in a_n for y in f_n if x in y])

  print('No Matches')
  print()
  for i in a_n:
    not_found = True
    for j in f_n:
      if i in j:
        not_found = False
    if not_found:
      print(i)

  lb()

  print('All Furman Neighborhoods')
  print()
  for i in f_n:
    print(i)

  return


def map_neighborhoods(airbnb, furman):

  br_map = {
    'QN': 'Queens',
    'MN': 'Manhattan',
    'BX': 'Bronx',
    'BK': 'Brooklyn',
    'SI': 'Staten Island'
  }

  matched = {}
  for br in list(br_map.keys()):
    f_n = list(
      furman[furman['Community District'].str.contains(br)]['Name'].unique())
    a_n = list(airbnb[airbnb['neighbourhood_group'].str.contains(
      br_map[br])]['neighbourhood'].unique())

    pairs = [(x, y) for x in a_n for y in f_n if x in y]
    for i in pairs:
      matched[i[0]] = i[1]

  mn = {
    'Murray Hill': 'Stuyvesant Town/Turtle Bay',
    "Hell's Kitchen": 'Clinton/Chelsea',
    'West Village': 'Greenwich Village/Soho',
    'East Village': 'Lower East Side/Chinatown',
    'Kips Bay': 'Stuyvesant Town/Turtle Bay',
    'SoHo': 'Greenwich Village/Soho',
    'NoHo': 'Greenwich Village/Soho',
    'Flatiron District': 'Midtown',
    'Roosevelt Island': None,
    'Little Italy': 'Greenwich Village/Soho',
    'Two Bridges': 'Lower East Side/Chinatown',
    'Nolita': 'Greenwich Village/Soho',
    'Gramercy': 'Stuyvesant Town/Turtle Bay',
    'Theater District': 'Midtown',
    'Tribeca': 'Financial District',
    'Battery Park City': 'Financial District',
    'Civic Center': 'Lower East Side/Chinatown',
    'Marble Hill': 'Riverdale/Fieldston'
  }

  bk = {
    'Kensington': 'Borough Park',
    'Clinton Hill': 'Fort Greene/Brooklyn Heights',
    'Bedford-Stuyvesant': 'Bedford Stuyvesant',
    'South Slope': 'Sunset Park',
    'Windsor Terrace': 'Sunset Park',
    'Prospect-Lefferts Gardens': 'South Crown Heights/Lefferts Gardens',
    'Gowanus': 'Park Slope/Carroll Gardens',
    'Cobble Hill': 'Park Slope/Carroll Gardens',
    'Boerum Hill': 'Fort Greene/Brooklyn Heights',
    'DUMBO': 'Fort Greene/Brooklyn Heights',
    'Gravesend': 'Coney Island',
    'Fort Hamilton': 'Bay Ridge/Dyker Heights',
    'Brighton Beach': 'Coney Island',
    'Cypress Hills': 'Kew Gardens/Woodhaven',
    'Columbia St': 'Park Slope/Carroll Gardens',
    'Vinegar Hill': 'Fort Greene/Brooklyn Heights',
    'Downtown Brooklyn': 'Fort Greene/Brooklyn Heights',
    'Red Hook': 'Park Slope/Carroll Gardens',
    'Sea Gate': 'Coney Island',
    'Navy Yard': 'Fort Greene/Brooklyn Heights',
    'Manhattan Beach': 'Sheepshead Bay',
    'Bergen Beach': 'Flatlands/Canarsie',
    'Bath Beach': 'Bensonhurst',
    'Mill Basin': 'Flatlands/Canarsie'
  }

  bx = {
    'Clason Point': 'Parkchester/Soundview',
    'Eastchester': 'Williamsbridge/Baychester',
    'Woodlawn': 'Williamsbridge/Baychester',
    'Allerton': 'Morris Park/Bronxdale',
    'Concourse Village': 'Highbridge/Concourse',
    'Wakefield': 'Williamsbridge/Baychester',
    'Spuyten Duyvil': 'Riverdale/Fieldston',
    'Morris Heights': 'Fordham/University Heights',
    'Port Morris': 'Mott Haven/Melrose',
    'Mount Eden': 'Highbridge/Concourse',
    'City Island': 'Throgs Neck/Co-op City',
    'North Riverdale': 'Riverdale/Fieldston',
    'Norwood': 'Kingsbridge Heights/Bedford',
    'Claremont Village': 'Morrisania/Crotona',
    'Mount Hope': 'Fordham/University Heights',
    'Van Nest': 'Morris Park/Bronxdale',
    'East Morrisania': 'Morrisania/Crotona',
    'Pelham Bay': 'Throgs Neck/Co-op City',
    'West Farms': 'Belmont/East Tremont',
    'Pelham Gardens': 'Morris Park/Bronxdale',
    'Schuylerville': 'Throgs Neck/Co-op City',
    'Castle Hill': 'Parkchester/Soundview',
    'Olinville': 'Williamsbridge/Baychester',
    'Edenwald': 'Williamsbridge/Baychester',
    'Westchester Square': 'Throgs Neck/Co-op City',
    'Unionport': 'Throgs Neck/Co-op City'
  }

  qn = {
    'Long Island City': 'Woodside/Sunnyside',
    'Middle Village': 'Ridgewood/Maspeth',
    'Ditmars Steinway': 'Astoria',
    'Rockaway Beach': 'Rockaway/Broad Channel',
    'St. Albans': 'Jamaica/Hollis',
    'Briarwood': 'Hillcrest/Fresh Meadows',
    'East Elmhurst': 'Jackson Heights',
    'Arverne': 'Rockaway/Broad Channel',
    'Cambria Heights': 'Queens Village',
    'College Point': 'Flushing/Whitestone',
    'Glendale': 'Ridgewood/Maspeth',
    'Richmond Hill': 'Kew Gardens/Woodhaven',
    'Bellerose': 'Queens Village',
    'Kew Gardens Hills': 'Hillcrest/Fresh Meadows',
    'Bay Terrace': 'Flushing/Whitestone',
    'Bayswater': 'Rockaway/Broad Channel',
    'Springfield Gardens': 'Jamaica/Hollis',
    'Belle Harbor': 'Rockaway/Broad Channel',
    'Far Rockaway': 'Rockaway/Broad Channel',
    'Neponsit': 'Rockaway/Broad Channel',
    'Laurelton': 'Queens Village',
    'Holliswood': 'Hillcrest/Fresh Meadows',
    'Rosedale': 'Queens Village',
    'Edgemere': 'Rockaway/Broad Channel',
    'Jamaica Hills': 'Hillcrest/Fresh Meadows',
    'Douglaston': 'Bayside/Little Neck',
    'Breezy Point': 'Rockaway/Broad Channel'
  }

  si = {
    'Tompkinsville': 'St. George/Stapleton',
    'Emerson Hill': 'South Beach/Willowbrook',
    'Shore Acres': 'St. George/Stapleton',
    'Arrochar': 'South Beach/Willowbrook',
    'Clifton': 'St. George/Stapleton',
    'Graniteville': 'St. George/Stapleton',
    'New Springville': 'South Beach/Willowbrook',
    'Mariners Harbor': 'St. George/Stapleton',
    'Concord': 'South Beach/Willowbrook',
    'Port Richmond': 'St. George/Stapleton',
    'Woodrow': 'Tottenville/Great Kills',
    'Eltingville': 'Tottenville/Great Kills',
    'Lighthouse Hill': 'South Beach/Willowbrook',
    'West Brighton': 'St. George/Stapleton',
    'Dongan Hills': 'South Beach/Willowbrook',
    'Castleton Corners': 'St. George/Stapleton',
    'Randall Manor': 'St. George/Stapleton',
    'Todt Hill': 'South Beach/Willowbrook',
    'Silver Lake': 'St. George/Stapleton',
    'Grymes Hill': 'St. George/Stapleton',
    'New Brighton': 'St. George/Stapleton',
    'Midland Beach': 'South Beach/Willowbrook',
    'Richmondtown': 'Tottenville/Great Kills',
    'Howland Hook': 'St. George/Stapleton',
    'New Dorp Beach': 'South Beach/Willowbrook',
    "Prince's Bay": 'Tottenville/Great Kills',
    'Oakwood': 'Tottenville/Great Kills',
    'Huguenot': 'Tottenville/Great Kills',
    'Grant City': 'South Beach/Willowbrook',
    'Westerleigh': 'St. George/Stapleton',
    'Bay Terrace, Staten Island': 'Tottenville/Great Kills',
    'Fort Wadsworth': None,
    'Rosebank': 'St. George/Stapleton',
    'Arden Heights': 'Tottenville/Great Kills',
    "Bull's Head": 'St. George/Stapleton',
    'New Dorp': 'South Beach/Willowbrook',
    'Rossville': 'Tottenville/Great Kills'
  }

  combined_dict = {**mn, **bk, **bx, **qn, **si, **matched}
  airbnb['neighborhood'] = airbnb['neighbourhood'].map(combined_dict, 'ignore')

  airbnb = airbnb.dropna(axis=0, subset=['neighborhood'])
  return airbnb


def join_dfs(airbnb, furman):

  furman.drop(columns=[
    'Unnamed: 0', '2000', '2006', '2010', '2020', '2021', 'Indicator Category',
    ' Indicator Description', 'Community District'
  ],
              inplace=True)
  furman['Indicator'] = furman['Indicator'].str.replace('"', '')
  furman = pd.pivot(data=furman,
                    index='Name',
                    columns='Indicator',
                    values='2019').reset_index()

  furman = furman[[
    'Rental vacancy rate', 'Severe crowding rate (% of renter households)',
    'Index of housing price appreciation, all property types',
    'Units authorized by new residential building permits',
    'Median rent, all (2021$)',
    'Rental units affordable at  80% AMI (% of recently available units)',
    'Moderately rent-burdened households', 'Name'
  ]]
  combined = airbnb.merge(furman,
                          left_on='neighborhood',
                          right_on='Name',
                          how='left')

  return combined


def keep_necessary_cols_airbnb_only(dff):

  df = dff[[
    'room_type', 'price', 'minimum_nights', 'number_of_reviews',
    'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
    'Name'
  ]]
  df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

  m = pd.get_dummies(df["room_type"], prefix='roomtype', drop_first=True)
  df = df.join(m).drop(columns=['room_type', 'roomtype_Private room'])

  return df


def keep_necessary_cols_furman_only(dff):

  df = dff[[
    'Rental vacancy rate', 'Severe crowding rate (% of renter households)',
    'Index of housing price appreciation, all property types',
    'Units authorized by new residential building permits',
    'Median rent, all (2021$)',
    'Rental units affordable at  80% AMI (% of recently available units)',
    'Moderately rent-burdened households', 'price', 'Name'
  ]]

  df['Rental vacancy rate'] = df['Rental vacancy rate'].str.strip('%').astype(
    float) / 100
  df['Severe crowding rate (% of renter households)'] = df[
    'Severe crowding rate (% of renter households)'].str.strip('%').astype(
      float) / 100
  df['Median rent, all (2021$)'] = df['Median rent, all (2021$)'].str.strip(
    '$').str.replace(',', '').astype(float)
  df['Rental units affordable at  80% AMI (% of recently available units)'] = df[
    'Rental units affordable at  80% AMI (% of recently available units)'].str.strip(
      '%').astype(float) / 100
  df['Moderately rent-burdened households'] = df[
    'Moderately rent-burdened households'].str.strip('%').astype(float) / 100

  df['Index of housing price appreciation, all property types'] = df[
    'Index of housing price appreciation, all property types'].astype(float)
  df['Units authorized by new residential building permits'] = df[
    'Units authorized by new residential building permits'].str.replace(
      ',', '').astype(float)

  return df


def keep_necessary_cols(dff):

  df = dff[[
    'Rental vacancy rate', 'Severe crowding rate (% of renter households)',
    'Index of housing price appreciation, all property types',
    'Units authorized by new residential building permits',
    'Median rent, all (2021$)',
    'Rental units affordable at  80% AMI (% of recently available units)',
    'Moderately rent-burdened households', 'room_type', 'price',
    'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365', 'Name'
  ]]

  df['Rental vacancy rate'] = df['Rental vacancy rate'].str.strip('%').astype(
    float) / 100
  df['Severe crowding rate (% of renter households)'] = df[
    'Severe crowding rate (% of renter households)'].str.strip('%').astype(
      float) / 100
  df['Median rent, all (2021$)'] = df['Median rent, all (2021$)'].str.strip(
    '$').str.replace(',', '').astype(float)
  df['Rental units affordable at  80% AMI (% of recently available units)'] = df[
    'Rental units affordable at  80% AMI (% of recently available units)'].str.strip(
      '%').astype(float) / 100
  df['Moderately rent-burdened households'] = df[
    'Moderately rent-burdened households'].str.strip('%').astype(float) / 100

  df['Index of housing price appreciation, all property types'] = df[
    'Index of housing price appreciation, all property types'].astype(float)
  df['Units authorized by new residential building permits'] = df[
    'Units authorized by new residential building permits'].str.replace(
      ',', '').astype(float)

  df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

  m = pd.get_dummies(df["room_type"], prefix='roomtype', drop_first=True)
  df = df.join(m).drop(columns=['room_type', 'roomtype_Private room'])

  return df


if __name__ == "__main__":

  # data cleaning

  #read_data_first_time()
  airbnb, furman = read_data()
  airbnb = map_neighborhoods(airbnb, furman)

  df = join_dfs(airbnb, furman)

  # keep only furman metrics, airbnb metrics, or all metrics

  #df = keep_necessary_cols_furman_only(df)
  #df = keep_necessary_cols_airbnb_only(df)
  df = keep_necessary_cols(df)

  lb()
  print(df)
  print(df.columns)
  print(df.head())
  print(df.dtypes)
  lb()
  lb()
  lb()
  
  c = list(df.columns)
  print('LOOK HERE')
  print(len(df))
  print(c)
  new = df.describe(percentiles=[.25,.5,.75])

  for i in new.columns:
    print(new[i])
  '''
  i = list(df['Name'].unique())

  vars = pd.DataFrame(columns=list(set(c)), index=list(set(i)))
  v = pd.DataFrame(columns=list(set(c)), index=list(set(i)))
  
  for col in set(c):
    for nh in set(i):
      temp = df[df['Name']==nh][[col]]
      #print(temp)
      std = np.std(np.array(temp))
      mn = np.mean(np.array(temp))
      mi = np.min(np.array(temp))
      ma = np.max(np.array(temp))

      vars.at[nh,col]=std/mn
      #v.at[nh,col]=np.var(np.array(temp))

  vars.to_csv('spread_data.csv')
  #v.to_csv('var_data.csv')

  
  #plotting feature variance
  # for col in v.columns:
  #   plt.bar(v[col].index, v[col])

  # comment out this line if you want your df to have every row being an airbnb listing
  # keep this line in if you want your df to have every row being a neighborhood

  #df = df.groupby('Name').mean()

  # comment out these lines if you did not comment out the line above
  
  #df = df.drop(columns = ['Name'])
  #df = df[df['price'] < 2000]
'''
  # PCA
'''
  X = df.drop(columns=['price']).fillna(0)
  X = X - X.mean()

  
  # pca = PCA()
  # pca.fit(X)

  # two_comp = pca.components_[0:2].T
  # print(two_comp)
  # print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))
  # print(X.columns)
  
  
  y = df['price']
  X['intercept'] = 1
  
  coeffs_test = []
  coeffs_train = []

  #don't use for loop if running with groupby line above commented
  count = 0
  count1= 0
  for i in range(0,1000):
    # OLS
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3) 
  
    #scale the training and testing data
    #X_reduced_train = pca.fit_transform(scale(X_train))
    #X_reduced_test = pca.transform(scale(X_test))[:,:2]
    X_reduced_train = X_train
    X_reduced_test = X_test
  
    #train PCR model on training data 
    #regr = LinearRegression()
    #regr.fit(X_reduced_train[:,:2], y_train)
    #regr.fit(X_reduced_train, y_train)

    #lasso
    regr = LassoCV(cv=5, random_state=0, max_iter=10000)
    regr.fit(X_reduced_train, y_train)
    regr = Lasso(alpha=regr.alpha_)
    regr.fit(X_reduced_train, y_train)

    
    #bagging
    # from sklearn.model_selection import cross_val_score
    # from sklearn.model_selection import RepeatedKFold
    # from sklearn.ensemble import BaggingRegressor
    # regr = BaggingRegressor()
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # n_scores = cross_val_score(regr, X, y, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
    # print(n_scores)
  
    #y_train_pred = list(regr.predict(X_reduced_train[:,:2]))
    y_train_pred = list(regr.predict(X_reduced_train))
    y_test_pred = list(regr.predict(X_reduced_test))
  
    #print('average abs. error (test): ' + str(np.mean(np.abs(np.array(y_test_pred) - np.array(y_test)))))

    if r2_score(y_test, y_test_pred) > 0:
      coeffs_test.append(r2_score(y_test, y_test_pred))
    else:
      count = count + 1
    
    if r2_score(y_train, y_train_pred) > 0:
      coeffs_train.append(r2_score(y_train, y_train_pred))
    else:
      count1 = count1 + 1

    # print(coeffs_test)
    # print(coeffs_train)
    #print(list(zip(regr.coef_, X)))
    print(i)


  print("Testing R^2: " + str(sum(coeffs_test) / len(coeffs_test)))
  print("Training R^2: " + str(sum(coeffs_train) / len(coeffs_train)))
  print(count)
  print(count1)
  #print(regr.alpha_)


'''
