{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **ML1 PROJECT 3** <br/>\n",
    "Kati Schuerger, Will Sherman, Randy Kim"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **INTRODUCTION** <br/>\n",
    "*Assingment: This section is not part of the rubric; meaning, there is no formal assignment for this \"introduction\" section. This can be used to provide some additional information regarding what the reader can find in the following project.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-block alert-info\">\n",
    "<b><h2>Rubric Components</h2>\n",
    "    </b>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='TOP'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "• [Business Understanding](#BUSINESS_UNDERSTANDING)\n",
    "\n",
    "• [Data Understanding 1](#DATA_UNDERSTANDING_1)\n",
    "\n",
    "• [Data Understanding 2](#DATA_UNDERSTANDING_2)\n",
    "\n",
    "• [Modeling and Evaluation 1](#ME1)\n",
    "\n",
    "• [Modeling and Evaluation 2](#ME2)\n",
    "\n",
    "• [Modeling and Evaluation 3](#ME3)\n",
    "\n",
    "• [Modeling and Evaluation 4](#ME4)\n",
    "\n",
    "• [Deployment](#DEPLOYMENT)\n",
    "\n",
    "• [Exceptional Work](#EXCEPTIONAL)\n",
    "\n",
    "• [Appendix](#APPENDIX)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PROJECT OBJECTIVES"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Prediction task\n",
    "\n",
    "We will be building our models to solve a classification problem (classification task) = predict `quart_binary` (0/1). This variable is equal to 1 when the player's score is in the 4th quantile."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data volume <br/>\n",
    "*Try to use as much testing data as possible in a realistic manner.*\n",
    "\n",
    "For hyperparameter tuning, we used a subset of our training data, because we ran into very lengthy run-time when attempting to perform tuning on the full training set. To help decrease the run-time, a subsample of 10% of the train data was made. This will still represent about 270,000 records, which is sufficient for our analysis and modeling. \n",
    "\n",
    "As long as the sample size is greater than 30, we can leverage the CLT, which allows us to make inferences about our population by using a sampling of the data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='BUSINESS_UNDERSTANDING'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **BUSINESS UNDERSTANDING**\n",
    "\n",
    "Jump to [Top](#TOP)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Assignment: Describe the purpose of the data set you selected (i.e., why was this data collected in the first place?). How will you measure the effectiveness of a good algorithm? Why does your chosen validation method make sense for this specific\n",
    "dataset and the stakeholders needs?*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xF52Ysn0tH2w"
   },
   "source": [
    "The **PlayerUnknown’s Battleground** (also known as PUBG) **Finish Placement Prediction** Kaggle competition was posted to \"create a model which predicts players' finishing placement based on their final stats, on a scale from 1 (first place) to 0 (last place).\" This competition utilized publicly available official game data from PUBG through the PUBG Developer API. The questions of interest (QOI) were about the best strategy to win in PUBG, the best skillset, and to be able to model ranking in-game. The dataset provided was a breakdown of post-game metrics and percentile winning placement: with 4446966 records and 29 total features.\n",
    "\n",
    "https://www.kaggle.com/c/pubg-finish-placement-prediction/data\n",
    "\n",
    "Utilizing the `winPlacePerc` (continuous variable), our approach was to create a binary prediction evaluation: `quart_binary` (categorical variable). `winPlacePerc` is a ranking of a \"players' finishing placement based on their final stats, on a scale from 1 (first place) to 0 (last place).\" `quart_binary` will be utilized for evaluating whether clustering improves prediction capability.\n",
    "\n",
    "For this clustering and classification task, we will measure accuracy in combination with 10-fold cross-validation.\n",
    "\n",
    "Possible uses:\n",
    "\n",
    "\"Creating forecasting algorithms for online sports is not new. Therefore, the combination of the two measures could be related to optimizing a sports betting algorithms. You could also use these models for building and managing an effective team for tournaments. For example, the PUBG Global Championship 2021 Grand Finals had combined prize value of over 2 million USD.\" (Kim, Schuerger, Sherman. Lab 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **DATA UNDERSTANDING 1**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='DATA_UNDERSTANDING_1'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Assignment: Describe the meaning and type of data (scale, values, etc.) for each attribute in the data file. Verify data quality: Are there missing values? Duplicate data? Outliers? Are those mistakes? How do you deal with these problems?*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## DATA PRE-PROCESSING 1\n",
    "\n",
    "Jump to [Top](#TOP)\n",
    "\n",
    "### Set up environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# load libraries\n",
    "import csv\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.express as px\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA\n",
    "from sklearn import model_selection\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV\n",
    "from sklearn.linear_model import LogisticRegression, Lasso\n",
    "from sklearn import metrics as mt\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.decomposition import PCA\n",
    "import statistics\n",
    "import time\n",
    "\n",
    "# import libraries to clean up CopyWarning\n",
    "import warnings\n",
    "from pandas.core.common import SettingWithCopyWarning\n",
    "warnings.simplefilter(action=\"ignore\", category=SettingWithCopyWarning)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load in data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# load in the data\n",
    "pubg_raw = pd.read_csv(\"train_V2.csv\")\n",
    "\n",
    "# for kati local run\n",
    "# pubg_raw = pd.read_csv(r\"C:\\Users\\kschue200\\OneDrive - Comcast\\Documents\\00 SMU\\00 machine learning1\\mini lab\\train_V2.csv\")\n",
    "\n",
    "# for will local run\n",
    "# pubg_raw = pd.read_csv(r\"C:\\Users\\sherm\\OneDrive\\Documents\\Grad School - Classes\\MSDS - 7331 - Machine Learning I\\Lab 1\\pubg-finish-placement-prediction\\train_V2.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Reduce memory usage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# this logic iterates through the x_tune to reduce memory usage\n",
    "# code adapted from https://www.kaggle.com/yansun1996/gbr-ipynb\n",
    "# additional reference: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n",
    "\n",
    "def reduce_mem_usage(df):\n",
    "    '''\n",
    "    iterate through all the columns of a dataframe and modify the data type\n",
    "    to reduce memory usage.        \n",
    "    '''\n",
    "    start_mem = df.memory_usage().sum() / 1024**2\n",
    "    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))\n",
    "    for col in df.columns:\n",
    "        col_type = df[col].dtype\n",
    "        if col_type != object:\n",
    "            c_min = df[col].min()\n",
    "            c_max = df[col].max()\n",
    "            if str(col_type)[:3] == 'int':\n",
    "                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n",
    "                    df[col] = df[col].astype(np.int8)\n",
    "                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n",
    "                    df[col] = df[col].astype(np.int16)\n",
    "                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n",
    "                    df[col] = df[col].astype(np.int32)\n",
    "                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n",
    "                    df[col] = df[col].astype(np.int64)  \n",
    "            else:\n",
    "                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n",
    "                    df[col] = df[col].astype(np.float16)\n",
    "                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n",
    "                    df[col] = df[col].astype(np.float32)\n",
    "                else:\n",
    "                    df[col] = df[col].astype(np.float64)\n",
    "    end_mem = df.memory_usage().sum() / 1024**2\n",
    "    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))\n",
    "    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Memory usage of dataframe is 983.90 MB\n",
      "Memory usage after optimization is: 288.39 MB\n",
      "Decreased by 70.7%\n"
     ]
    }
   ],
   "source": [
    "pubg_raw = reduce_mem_usage(pubg_raw)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We have utilized the above to help reduce memory usage & improve run-times."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "random.seed(10) #set seed"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## DATA PRE-PROCESSING 2\n",
    "\n",
    "Jump to [Top](#TOP)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>groupId</th>\n",
       "      <th>matchId</th>\n",
       "      <th>assists</th>\n",
       "      <th>boosts</th>\n",
       "      <th>damageDealt</th>\n",
       "      <th>DBNOs</th>\n",
       "      <th>headshotKills</th>\n",
       "      <th>heals</th>\n",
       "      <th>killPlace</th>\n",
       "      <th>...</th>\n",
       "      <th>revives</th>\n",
       "      <th>rideDistance</th>\n",
       "      <th>roadKills</th>\n",
       "      <th>swimDistance</th>\n",
       "      <th>teamKills</th>\n",
       "      <th>vehicleDestroys</th>\n",
       "      <th>walkDistance</th>\n",
       "      <th>weaponsAcquired</th>\n",
       "      <th>winPoints</th>\n",
       "      <th>winPlacePerc</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7f96b2f878858a</td>\n",
       "      <td>4d4b580de459be</td>\n",
       "      <td>a10357fd1a4a91</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>60</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>244.75</td>\n",
       "      <td>1</td>\n",
       "      <td>1466</td>\n",
       "      <td>0.444336</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>eef90569b9d03c</td>\n",
       "      <td>684d5656442f9e</td>\n",
       "      <td>aeb375fc57110c</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>91.50000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>57</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.004501</td>\n",
       "      <td>0</td>\n",
       "      <td>11.039062</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1434.00</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0.640137</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1eaf90ac73de72</td>\n",
       "      <td>6a4a42c3245a74</td>\n",
       "      <td>110163d8bb94ae</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>68.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>47</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>161.75</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0.775391</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4616d365dd2853</td>\n",
       "      <td>a930a9c79cd721</td>\n",
       "      <td>f1f1f4ef412d7e</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>32.90625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>75</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>202.75</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0.166748</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>315c96c26c9aac</td>\n",
       "      <td>de04010b3458dd</td>\n",
       "      <td>6dc8ff871e21e6</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>100.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>45</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>49.75</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0.187500</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 29 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "               Id         groupId         matchId  assists  boosts  \\\n",
       "0  7f96b2f878858a  4d4b580de459be  a10357fd1a4a91        0       0   \n",
       "1  eef90569b9d03c  684d5656442f9e  aeb375fc57110c        0       0   \n",
       "2  1eaf90ac73de72  6a4a42c3245a74  110163d8bb94ae        1       0   \n",
       "3  4616d365dd2853  a930a9c79cd721  f1f1f4ef412d7e        0       0   \n",
       "4  315c96c26c9aac  de04010b3458dd  6dc8ff871e21e6        0       0   \n",
       "\n",
       "   damageDealt  DBNOs  headshotKills  heals  killPlace  ...  revives  \\\n",
       "0      0.00000      0              0      0         60  ...        0   \n",
       "1     91.50000      0              0      0         57  ...        0   \n",
       "2     68.00000      0              0      0         47  ...        0   \n",
       "3     32.90625      0              0      0         75  ...        0   \n",
       "4    100.00000      0              0      0         45  ...        0   \n",
       "\n",
       "   rideDistance  roadKills  swimDistance  teamKills vehicleDestroys  \\\n",
       "0      0.000000          0      0.000000          0               0   \n",
       "1      0.004501          0     11.039062          0               0   \n",
       "2      0.000000          0      0.000000          0               0   \n",
       "3      0.000000          0      0.000000          0               0   \n",
       "4      0.000000          0      0.000000          0               0   \n",
       "\n",
       "   walkDistance  weaponsAcquired  winPoints  winPlacePerc  \n",
       "0        244.75                1       1466      0.444336  \n",
       "1       1434.00                5          0      0.640137  \n",
       "2        161.75                2          0      0.775391  \n",
       "3        202.75                3          0      0.166748  \n",
       "4         49.75                2          0      0.187500  \n",
       "\n",
       "[5 rows x 29 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pubg_raw.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Check for missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Column 'Id' has 0 NAs\n",
      "Column 'groupId' has 0 NAs\n",
      "Column 'matchId' has 0 NAs\n",
      "Column 'assists' has 0 NAs\n",
      "Column 'boosts' has 0 NAs\n",
      "Column 'damageDealt' has 0 NAs\n",
      "Column 'DBNOs' has 0 NAs\n",
      "Column 'headshotKills' has 0 NAs\n",
      "Column 'heals' has 0 NAs\n",
      "Column 'killPlace' has 0 NAs\n",
      "Column 'killPoints' has 0 NAs\n",
      "Column 'kills' has 0 NAs\n",
      "Column 'killStreaks' has 0 NAs\n",
      "Column 'longestKill' has 0 NAs\n",
      "Column 'matchDuration' has 0 NAs\n",
      "Column 'matchType' has 0 NAs\n",
      "Column 'maxPlace' has 0 NAs\n",
      "Column 'numGroups' has 0 NAs\n",
      "Column 'rankPoints' has 0 NAs\n",
      "Column 'revives' has 0 NAs\n",
      "Column 'rideDistance' has 0 NAs\n",
      "Column 'roadKills' has 0 NAs\n",
      "Column 'swimDistance' has 0 NAs\n",
      "Column 'teamKills' has 0 NAs\n",
      "Column 'vehicleDestroys' has 0 NAs\n",
      "Column 'walkDistance' has 0 NAs\n",
      "Column 'weaponsAcquired' has 0 NAs\n",
      "Column 'winPoints' has 0 NAs\n",
      "Column 'winPlacePerc' has 1 NAs\n"
     ]
    }
   ],
   "source": [
    "# check for misssing values\n",
    "colname = list(pubg_raw.columns)\n",
    "\n",
    "for i in range(len(pubg_raw.columns)):\n",
    "  count = pubg_raw[pubg_raw.columns[i]].isna().sum()\n",
    "  print(\"Column '{col}' has {ct} NAs\".format(col = colname[i], ct = count))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>groupId</th>\n",
       "      <th>matchId</th>\n",
       "      <th>assists</th>\n",
       "      <th>boosts</th>\n",
       "      <th>damageDealt</th>\n",
       "      <th>DBNOs</th>\n",
       "      <th>headshotKills</th>\n",
       "      <th>heals</th>\n",
       "      <th>killPlace</th>\n",
       "      <th>...</th>\n",
       "      <th>revives</th>\n",
       "      <th>rideDistance</th>\n",
       "      <th>roadKills</th>\n",
       "      <th>swimDistance</th>\n",
       "      <th>teamKills</th>\n",
       "      <th>vehicleDestroys</th>\n",
       "      <th>walkDistance</th>\n",
       "      <th>weaponsAcquired</th>\n",
       "      <th>winPoints</th>\n",
       "      <th>winPlacePerc</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2744604</th>\n",
       "      <td>f70c74418bb064</td>\n",
       "      <td>12dfbede33f92b</td>\n",
       "      <td>224a123c53e008</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 29 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                     Id         groupId         matchId  assists  boosts  \\\n",
       "2744604  f70c74418bb064  12dfbede33f92b  224a123c53e008        0       0   \n",
       "\n",
       "         damageDealt  DBNOs  headshotKills  heals  killPlace  ...  revives  \\\n",
       "2744604          0.0      0              0      0          1  ...        0   \n",
       "\n",
       "         rideDistance  roadKills  swimDistance  teamKills vehicleDestroys  \\\n",
       "2744604           0.0          0           0.0          0               0   \n",
       "\n",
       "         walkDistance  weaponsAcquired  winPoints  winPlacePerc  \n",
       "2744604           0.0                0          0           NaN  \n",
       "\n",
       "[1 rows x 29 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# look at records with null values\n",
    "pubg_raw[pubg_raw.isna().any(axis = 1)]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There was one missing value in `winPlacePerc` that was due to a single instance of a player who had a match with no enemy combatants. Therefore, a `winPlacePerc` was not evaluated."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# CREATE PUBG_DF_STG TO INCLUDE OUR UPDATES\n",
    "# drop records with missing values (there is only one)\n",
    "pubg_df_stg = pubg_raw.dropna()\n",
    "pubg_df_stg.isnull().values.any()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Based on our analysis, we dropped the single missing value for a match that had only 1 individual (above).\n",
    "\n",
    "We also identified **unranked** matches as creating significant issues in prediction capabilities (below). There wer matches with `rankPoints` with values of  '-1'. We resolved to remove these from our model. Therefore, our model should only be applied to **ranked** matches. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1701810\n",
      "0\n",
      "0\n",
      "2745155\n"
     ]
    }
   ],
   "source": [
    "# we have some records with -1 values = these take the place of NULL\n",
    "\n",
    "x = pubg_df_stg['rankPoints']\n",
    "y = pubg_df_stg['winPoints']\n",
    "z = pubg_df_stg['killPoints']\n",
    "\n",
    "print(x[x==-1].count())\n",
    "print(y[y==-1].count())\n",
    "print(z[z==-1].count())\n",
    "print(pubg_df_stg.shape[0]-x[x==-1].count())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Subset data - Ranked matches only"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>groupId</th>\n",
       "      <th>matchId</th>\n",
       "      <th>assists</th>\n",
       "      <th>boosts</th>\n",
       "      <th>damageDealt</th>\n",
       "      <th>DBNOs</th>\n",
       "      <th>headshotKills</th>\n",
       "      <th>heals</th>\n",
       "      <th>killPlace</th>\n",
       "      <th>...</th>\n",
       "      <th>revives</th>\n",
       "      <th>rideDistance</th>\n",
       "      <th>roadKills</th>\n",
       "      <th>swimDistance</th>\n",
       "      <th>teamKills</th>\n",
       "      <th>vehicleDestroys</th>\n",
       "      <th>walkDistance</th>\n",
       "      <th>weaponsAcquired</th>\n",
       "      <th>winPoints</th>\n",
       "      <th>winPlacePerc</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>eef90569b9d03c</td>\n",
       "      <td>684d5656442f9e</td>\n",
       "      <td>aeb375fc57110c</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>91.50000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>57</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.004501</td>\n",
       "      <td>0</td>\n",
       "      <td>11.039062</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1434.0000</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0.640137</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1eaf90ac73de72</td>\n",
       "      <td>6a4a42c3245a74</td>\n",
       "      <td>110163d8bb94ae</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>68.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>47</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>161.7500</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0.775391</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4616d365dd2853</td>\n",
       "      <td>a930a9c79cd721</td>\n",
       "      <td>f1f1f4ef412d7e</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>32.90625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>75</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>202.7500</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0.166748</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>315c96c26c9aac</td>\n",
       "      <td>de04010b3458dd</td>\n",
       "      <td>6dc8ff871e21e6</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>100.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>45</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>49.7500</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0.187500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>ff79c12f326506</td>\n",
       "      <td>289a6836a88d27</td>\n",
       "      <td>bac52627a12114</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>100.00000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>44</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>34.6875</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.036987</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 29 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "               Id         groupId         matchId  assists  boosts  \\\n",
       "1  eef90569b9d03c  684d5656442f9e  aeb375fc57110c        0       0   \n",
       "2  1eaf90ac73de72  6a4a42c3245a74  110163d8bb94ae        1       0   \n",
       "3  4616d365dd2853  a930a9c79cd721  f1f1f4ef412d7e        0       0   \n",
       "4  315c96c26c9aac  de04010b3458dd  6dc8ff871e21e6        0       0   \n",
       "5  ff79c12f326506  289a6836a88d27  bac52627a12114        0       0   \n",
       "\n",
       "   damageDealt  DBNOs  headshotKills  heals  killPlace  ...  revives  \\\n",
       "1     91.50000      0              0      0         57  ...        0   \n",
       "2     68.00000      0              0      0         47  ...        0   \n",
       "3     32.90625      0              0      0         75  ...        0   \n",
       "4    100.00000      0              0      0         45  ...        0   \n",
       "5    100.00000      1              1      0         44  ...        0   \n",
       "\n",
       "   rideDistance  roadKills  swimDistance  teamKills vehicleDestroys  \\\n",
       "1      0.004501          0     11.039062          0               0   \n",
       "2      0.000000          0      0.000000          0               0   \n",
       "3      0.000000          0      0.000000          0               0   \n",
       "4      0.000000          0      0.000000          0               0   \n",
       "5      0.000000          0      0.000000          0               0   \n",
       "\n",
       "   walkDistance  weaponsAcquired  winPoints  winPlacePerc  \n",
       "1     1434.0000                5          0      0.640137  \n",
       "2      161.7500                2          0      0.775391  \n",
       "3      202.7500                3          0      0.166748  \n",
       "4       49.7500                2          0      0.187500  \n",
       "5       34.6875                1          0      0.036987  \n",
       "\n",
       "[5 rows x 29 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# SUBSET OUR DATA TO ONLY RETAIN RANKED MATCHES\n",
    "# we have some records with -1 values = take the place of NULL\n",
    "## dropping these to subset our dataset & because they represent non-ranked matches\n",
    "\n",
    "pubg_df = pubg_df_stg[(pubg_df_stg.iloc[:,:] != -1).all(axis=1)]\n",
    "pubg_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Check for duplicate values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check for duplicate records\n",
    "pubg_df.duplicated().any()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There do not appear to be any duplicate records."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create new features = quartiles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Quartile 1 threshold: 0.20\n",
      "Quartile 2 threshold: 0.46\n",
      "Quartile 3 threshold: 0.74\n"
     ]
    }
   ],
   "source": [
    "# GATHER THRESHOLDS FOR US TO SET OUR QARTILES \n",
    "# pull out the winPlacePerc column\n",
    "winPlace = pubg_df.loc[:,'winPlacePerc']\n",
    "\n",
    "# get quartile thresholds for equal allocation |\n",
    "print('Quartile 1 threshold: {:.2f}'.format(winPlace.quantile(0.25)))\n",
    "print('Quartile 2 threshold: {:.2f}'.format(winPlace.quantile(0.5)))\n",
    "print('Quartile 3 threshold: {:.2f}'.format(winPlace.quantile(0.75)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>groupId</th>\n",
       "      <th>matchId</th>\n",
       "      <th>assists</th>\n",
       "      <th>boosts</th>\n",
       "      <th>damageDealt</th>\n",
       "      <th>DBNOs</th>\n",
       "      <th>headshotKills</th>\n",
       "      <th>heals</th>\n",
       "      <th>killPlace</th>\n",
       "      <th>...</th>\n",
       "      <th>swimDistance</th>\n",
       "      <th>teamKills</th>\n",
       "      <th>vehicleDestroys</th>\n",
       "      <th>walkDistance</th>\n",
       "      <th>weaponsAcquired</th>\n",
       "      <th>winPoints</th>\n",
       "      <th>winPlacePerc</th>\n",
       "      <th>quartile</th>\n",
       "      <th>quart_int</th>\n",
       "      <th>quart_binary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>eef90569b9d03c</td>\n",
       "      <td>684d5656442f9e</td>\n",
       "      <td>aeb375fc57110c</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>91.50000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>57</td>\n",
       "      <td>...</td>\n",
       "      <td>11.039062</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1434.0000</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0.640137</td>\n",
       "      <td>q3</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1eaf90ac73de72</td>\n",
       "      <td>6a4a42c3245a74</td>\n",
       "      <td>110163d8bb94ae</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>68.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>47</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>161.7500</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0.775391</td>\n",
       "      <td>q4</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4616d365dd2853</td>\n",
       "      <td>a930a9c79cd721</td>\n",
       "      <td>f1f1f4ef412d7e</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>32.90625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>75</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>202.7500</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0.166748</td>\n",
       "      <td>q1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>315c96c26c9aac</td>\n",
       "      <td>de04010b3458dd</td>\n",
       "      <td>6dc8ff871e21e6</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>100.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>45</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>49.7500</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0.187500</td>\n",
       "      <td>q1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>ff79c12f326506</td>\n",
       "      <td>289a6836a88d27</td>\n",
       "      <td>bac52627a12114</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>100.00000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>44</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>34.6875</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.036987</td>\n",
       "      <td>q1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 32 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "               Id         groupId         matchId  assists  boosts  \\\n",
       "1  eef90569b9d03c  684d5656442f9e  aeb375fc57110c        0       0   \n",
       "2  1eaf90ac73de72  6a4a42c3245a74  110163d8bb94ae        1       0   \n",
       "3  4616d365dd2853  a930a9c79cd721  f1f1f4ef412d7e        0       0   \n",
       "4  315c96c26c9aac  de04010b3458dd  6dc8ff871e21e6        0       0   \n",
       "5  ff79c12f326506  289a6836a88d27  bac52627a12114        0       0   \n",
       "\n",
       "   damageDealt  DBNOs  headshotKills  heals  killPlace  ...  swimDistance  \\\n",
       "1     91.50000      0              0      0         57  ...     11.039062   \n",
       "2     68.00000      0              0      0         47  ...      0.000000   \n",
       "3     32.90625      0              0      0         75  ...      0.000000   \n",
       "4    100.00000      0              0      0         45  ...      0.000000   \n",
       "5    100.00000      1              1      0         44  ...      0.000000   \n",
       "\n",
       "   teamKills  vehicleDestroys  walkDistance  weaponsAcquired winPoints  \\\n",
       "1          0                0     1434.0000                5         0   \n",
       "2          0                0      161.7500                2         0   \n",
       "3          0                0      202.7500                3         0   \n",
       "4          0                0       49.7500                2         0   \n",
       "5          0                0       34.6875                1         0   \n",
       "\n",
       "   winPlacePerc  quartile  quart_int  quart_binary  \n",
       "1      0.640137        q3          3             0  \n",
       "2      0.775391        q4          4             1  \n",
       "3      0.166748        q1          1             0  \n",
       "4      0.187500        q1          1             0  \n",
       "5      0.036987        q1          1             0  \n",
       "\n",
       "[5 rows x 32 columns]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# create new variable fields to use for our predictions (quartile, quart_int, quart_binary)\n",
    "# we will use quart_binary for our classification task in this mini-lab\n",
    "\n",
    "pubg_df['quartile'] = np.where(pubg_df.winPlacePerc < 0.20, 'q1', \n",
    "                       np.where(pubg_df.winPlacePerc < 0.46, 'q2',\n",
    "                                np.where(pubg_df.winPlacePerc < 0.74, 'q3',\n",
    "                                         np.where(pubg_df.winPlacePerc >= 0.74, 'q4',\n",
    "                                                  'other'))))\n",
    "pubg_df['quart_int'] = np.where(pubg_df.winPlacePerc < 0.20, '1', \n",
    "                       np.where(pubg_df.winPlacePerc < 0.46, '2',\n",
    "                                np.where(pubg_df.winPlacePerc < 0.74, '3',\n",
    "                                         np.where(pubg_df.winPlacePerc >= 0.74, '4',\n",
    "                                                  'other'))))\n",
    "pubg_df['quart_binary'] = np.where(pubg_df.winPlacePerc < 0.74, '0',\n",
    "                                         np.where(pubg_df.winPlacePerc >= 0.74, '1',\n",
    "                                                  'other'))\n",
    "pubg_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Percentage of players in Quartile 1: 24.86%\n",
      "Percentage of players in Quartile 2: 25.19%\n",
      "Percentage of players in Quartile 3: 24.55%\n",
      "Percentage of players in Quartile 4: 25.41%\n",
      "Miscategorized quartile values:   0\n"
     ]
    }
   ],
   "source": [
    "# check the spread of our quartiles\n",
    "# make sure they are balanced\n",
    "\n",
    "quartiles = np.append(sorted(pubg_df.quartile.unique()), 'other')\n",
    "e = 1\n",
    "\n",
    "for i in quartiles:\n",
    "    if e < len(quartiles):\n",
    "        count = len(pubg_df[pubg_df.quartile == i])\n",
    "        print('Percentage of players in Quartile {}: {:.2f}%'.format(e, \n",
    "                                                                 (count / (len(pubg_df.quartile))*100)))\n",
    "        e += 1\n",
    "    else:\n",
    "        count = len(pubg_df[pubg_df.quartile == i])\n",
    "        print('Miscategorized quartile values:  ', count)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Remove unuseful variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove some unuseful object-datatype variables \n",
    "del pubg_df['Id']\n",
    "del pubg_df['groupId']\n",
    "del pubg_df['matchId']\n",
    "# del pubg_df['matchType'] # <-- we will leave in for now"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The above features are specific to individuals & will not be useful for the QOI: predicting players' finishing placement based on their stats."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Evaluate outliers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We will evaluate outliers by looking at the **data shape** for each of our attributes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x720 with 25 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "in_plot = pubg_df.hist(bins=30, figsize = (15, 10))[2]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Much of the data appears to be right-skewed, which also means we are likely to have many \"outliers\". This may impact our ability to run analytics unless we normalize and/or standardize our data.\n",
    "\n",
    "There are a few notable exceptions: `matchDuration`, `maxPlace`, and `numGroups`. These appear to be either bimodal or trimodal with `matchDuration` being fairly centered on the mean but `maxPlace` and `numGroups` again exhibiting right-skew.\n",
    "\n",
    "Therefore, our approach will be to evaluate clustering by first utilizing *principal component analysis* (PCA)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### New features <br/>\n",
    "Let's create some new features that we could use in our modeling later on."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "pubg_df['totalDistance'] = pubg_df['rideDistance'] + pubg_df['swimDistance'] + pubg_df['walkDistance']\n",
    "pubg_df['killsAssist'] = pubg_df['kills'] + pubg_df['assists']\n",
    "pubg_df['totalItems'] = pubg_df['heals'] + pubg_df['boosts'] + pubg_df['weaponsAcquired']\n",
    "pubg_df['healItems'] = pubg_df['heals'] + pubg_df['boosts']\n",
    "# pubg_df['MMR'] = pubg_df['killPoints'] + pubg_df['winPoints'] # + pubg_df['rankPoints'] <- may take out rank points as it depends on killPoints & winPoints"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**PLEASE NOTE: Data Scale and Descriptions provided in \"Appendix\" at the end of this notebook.** <br/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create object with feature column names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# FEATURE COLUMN NAMES (create cols_df)\n",
    "# make a list of the columns in our df (to be used for models below)\n",
    "\n",
    "cols_df = pubg_df.columns.values.tolist()\n",
    "cols_df.remove('winPlacePerc') # remove target variable 1 (continuous)\n",
    "cols_df.remove('quartile') # remove target variable 2 (categorical)\n",
    "cols_df.remove('quart_int') # remove target variable 2 (numeric representation)\n",
    "cols_df.remove('quart_binary')\n",
    "# print(cols_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove additional variables that are causing extra noise in our model\n",
    "# (specifically, these variables are highly correlated with other variables \n",
    "# which impacts our model's convergence perfomance)\n",
    "\n",
    "cols_df.remove('winPoints') # highly correlated with rankPoints and killPoints\n",
    "cols_df.remove('killPoints') # highly correlated with rabnkPoints and winPoints\n",
    "cols_df.remove('maxPlace') # perfectly correlated with numGroups\n",
    "# print(cols_df)\n",
    "\n",
    "# We maintain the rankPoints variable since we subset on this earlier to only work with 'ranked' matches"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove matchType = causes issues when we scale data for PCA\n",
    "\n",
    "del pubg_df['matchType']\n",
    "# del finalDF['matchType']\n",
    "cols_df.remove('matchType')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'kills', 'killStreaks', 'longestKill', 'matchDuration', 'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'totalDistance', 'killsAssist', 'totalItems', 'healItems']\n"
     ]
    }
   ],
   "source": [
    "# print cols_df (feature columns to be used in models)\n",
    "print(cols_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## FINAL DATAFRAME\n",
    "\n",
    "Jump to [Top](#TOP)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The below df.info() confirms that we have at least 30k records and 15 columns in data. We have 2,745,155 records and 21 feature columns—2 response vectors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 2745155 entries, 1 to 4446965\n",
      "Data columns (total 27 columns):\n",
      " #   Column           Dtype  \n",
      "---  ------           -----  \n",
      " 0   assists          int8   \n",
      " 1   boosts           int8   \n",
      " 2   damageDealt      float16\n",
      " 3   DBNOs            int8   \n",
      " 4   headshotKills    int8   \n",
      " 5   heals            int8   \n",
      " 6   killPlace        int8   \n",
      " 7   kills            int8   \n",
      " 8   killStreaks      int8   \n",
      " 9   longestKill      float16\n",
      " 10  matchDuration    int16  \n",
      " 11  numGroups        int8   \n",
      " 12  rankPoints       int16  \n",
      " 13  revives          int8   \n",
      " 14  rideDistance     float16\n",
      " 15  roadKills        int8   \n",
      " 16  swimDistance     float16\n",
      " 17  teamKills        int8   \n",
      " 18  vehicleDestroys  int8   \n",
      " 19  walkDistance     float16\n",
      " 20  weaponsAcquired  int16  \n",
      " 21  totalDistance    float16\n",
      " 22  killsAssist      int8   \n",
      " 23  totalItems       int16  \n",
      " 24  healItems        int8   \n",
      " 25  winPlacePerc     float16\n",
      " 26  quart_binary     object \n",
      "dtypes: float16(7), int16(4), int8(15), object(1)\n",
      "memory usage: 138.8+ MB\n"
     ]
    }
   ],
   "source": [
    "temp = pubg_df[cols_df]\n",
    "temp['winPlacePerc'] = pubg_df['winPlacePerc']\n",
    "temp['quart_binary'] = pubg_df['quart_binary']\n",
    "finalDF = temp\n",
    "finalDF.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='DATA_UNDERSTANDING_2'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **DATA UNDERSTANDING 2**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Assignment: Visualize the any important attributes appropriately. Important: Provide an interpretation for any charts or graphs.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CpvljC1is5mw"
   },
   "source": [
    "## SIMPLE STATISTICS\n",
    "\n",
    "Jump to [Top](#TOP)\n",
    "\n",
    "Because most of our data is continuous (either integers or floats), a large amount of our data can be evaluated for simple statistical measures."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 364
    },
    "id": "QSSeWFsoeppH",
    "outputId": "59d6956c-b735-4680-c923-6653ce0cf2f6"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>assists</th>\n",
       "      <th>boosts</th>\n",
       "      <th>damageDealt</th>\n",
       "      <th>DBNOs</th>\n",
       "      <th>headshotKills</th>\n",
       "      <th>heals</th>\n",
       "      <th>killPlace</th>\n",
       "      <th>kills</th>\n",
       "      <th>killStreaks</th>\n",
       "      <th>longestKill</th>\n",
       "      <th>...</th>\n",
       "      <th>swimDistance</th>\n",
       "      <th>teamKills</th>\n",
       "      <th>vehicleDestroys</th>\n",
       "      <th>walkDistance</th>\n",
       "      <th>weaponsAcquired</th>\n",
       "      <th>totalDistance</th>\n",
       "      <th>killsAssist</th>\n",
       "      <th>totalItems</th>\n",
       "      <th>healItems</th>\n",
       "      <th>winPlacePerc</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2745155.00</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>...</td>\n",
       "      <td>2745155.0</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2745155.000</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2745155.00</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "      <td>2.745155e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2.237921e-01</td>\n",
       "      <td>1.123926e+00</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.490603e-01</td>\n",
       "      <td>2.259672e-01</td>\n",
       "      <td>1.398382e+00</td>\n",
       "      <td>4.751563e+01</td>\n",
       "      <td>9.219162e-01</td>\n",
       "      <td>5.412292e-01</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.903151e-02</td>\n",
       "      <td>8.756154e-03</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3.676111e+00</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.145708e+00</td>\n",
       "      <td>6.198419e+00</td>\n",
       "      <td>2.522308e+00</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>5.763510e-01</td>\n",
       "      <td>1.735702e+00</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.134096e+00</td>\n",
       "      <td>5.980902e-01</td>\n",
       "      <td>2.725667e+00</td>\n",
       "      <td>2.741419e+01</td>\n",
       "      <td>1.551348e+00</td>\n",
       "      <td>7.063531e-01</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.815228e-01</td>\n",
       "      <td>9.752691e-02</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.453115e+00</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.819689e+00</td>\n",
       "      <td>5.401557e+00</td>\n",
       "      <td>3.939304e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>2.400000e+01</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>157.875</td>\n",
       "      <td>2.000000e+00</td>\n",
       "      <td>161.25</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>2.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1.999512e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>82.25</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>4.700000e+01</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>695.500</td>\n",
       "      <td>3.000000e+00</td>\n",
       "      <td>807.50</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>5.000000e+00</td>\n",
       "      <td>1.000000e+00</td>\n",
       "      <td>4.582520e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>2.000000e+00</td>\n",
       "      <td>184.00</td>\n",
       "      <td>1.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>2.000000e+00</td>\n",
       "      <td>7.100000e+01</td>\n",
       "      <td>1.000000e+00</td>\n",
       "      <td>1.000000e+00</td>\n",
       "      <td>2.167188e+01</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1980.000</td>\n",
       "      <td>5.000000e+00</td>\n",
       "      <td>2792.00</td>\n",
       "      <td>2.000000e+00</td>\n",
       "      <td>9.000000e+00</td>\n",
       "      <td>4.000000e+00</td>\n",
       "      <td>7.407227e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2.200000e+01</td>\n",
       "      <td>2.800000e+01</td>\n",
       "      <td>6616.00</td>\n",
       "      <td>4.000000e+01</td>\n",
       "      <td>6.400000e+01</td>\n",
       "      <td>7.300000e+01</td>\n",
       "      <td>1.010000e+02</td>\n",
       "      <td>7.200000e+01</td>\n",
       "      <td>1.100000e+01</td>\n",
       "      <td>1.094000e+03</td>\n",
       "      <td>...</td>\n",
       "      <td>3824.0</td>\n",
       "      <td>8.000000e+00</td>\n",
       "      <td>5.000000e+00</td>\n",
       "      <td>25776.000</td>\n",
       "      <td>1.530000e+02</td>\n",
       "      <td>41280.00</td>\n",
       "      <td>8.500000e+01</td>\n",
       "      <td>1.530000e+02</td>\n",
       "      <td>7.700000e+01</td>\n",
       "      <td>1.000000e+00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "            assists        boosts  damageDealt         DBNOs  headshotKills  \\\n",
       "count  2.745155e+06  2.745155e+06   2745155.00  2.745155e+06   2.745155e+06   \n",
       "mean   2.237921e-01  1.123926e+00          NaN  6.490603e-01   2.259672e-01   \n",
       "std    5.763510e-01  1.735702e+00          NaN  1.134096e+00   5.980902e-01   \n",
       "min    0.000000e+00  0.000000e+00         0.00  0.000000e+00   0.000000e+00   \n",
       "25%    0.000000e+00  0.000000e+00         0.00  0.000000e+00   0.000000e+00   \n",
       "50%    0.000000e+00  0.000000e+00        82.25  0.000000e+00   0.000000e+00   \n",
       "75%    0.000000e+00  2.000000e+00       184.00  1.000000e+00   0.000000e+00   \n",
       "max    2.200000e+01  2.800000e+01      6616.00  4.000000e+01   6.400000e+01   \n",
       "\n",
       "              heals     killPlace         kills   killStreaks   longestKill  \\\n",
       "count  2.745155e+06  2.745155e+06  2.745155e+06  2.745155e+06  2.745155e+06   \n",
       "mean   1.398382e+00  4.751563e+01  9.219162e-01  5.412292e-01           NaN   \n",
       "std    2.725667e+00  2.741419e+01  1.551348e+00  7.063531e-01           NaN   \n",
       "min    0.000000e+00  1.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   \n",
       "25%    0.000000e+00  2.400000e+01  0.000000e+00  0.000000e+00  0.000000e+00   \n",
       "50%    0.000000e+00  4.700000e+01  0.000000e+00  0.000000e+00  0.000000e+00   \n",
       "75%    2.000000e+00  7.100000e+01  1.000000e+00  1.000000e+00  2.167188e+01   \n",
       "max    7.300000e+01  1.010000e+02  7.200000e+01  1.100000e+01  1.094000e+03   \n",
       "\n",
       "       ...  swimDistance     teamKills  vehicleDestroys  walkDistance  \\\n",
       "count  ...     2745155.0  2.745155e+06     2.745155e+06   2745155.000   \n",
       "mean   ...           NaN  2.903151e-02     8.756154e-03           NaN   \n",
       "std    ...           NaN  1.815228e-01     9.752691e-02           NaN   \n",
       "min    ...           0.0  0.000000e+00     0.000000e+00         0.000   \n",
       "25%    ...           0.0  0.000000e+00     0.000000e+00       157.875   \n",
       "50%    ...           0.0  0.000000e+00     0.000000e+00       695.500   \n",
       "75%    ...           0.0  0.000000e+00     0.000000e+00      1980.000   \n",
       "max    ...        3824.0  8.000000e+00     5.000000e+00     25776.000   \n",
       "\n",
       "       weaponsAcquired  totalDistance   killsAssist    totalItems  \\\n",
       "count     2.745155e+06     2745155.00  2.745155e+06  2.745155e+06   \n",
       "mean      3.676111e+00            NaN  1.145708e+00  6.198419e+00   \n",
       "std       2.453115e+00            NaN  1.819689e+00  5.401557e+00   \n",
       "min       0.000000e+00           0.00  0.000000e+00  0.000000e+00   \n",
       "25%       2.000000e+00         161.25  0.000000e+00  2.000000e+00   \n",
       "50%       3.000000e+00         807.50  0.000000e+00  5.000000e+00   \n",
       "75%       5.000000e+00        2792.00  2.000000e+00  9.000000e+00   \n",
       "max       1.530000e+02       41280.00  8.500000e+01  1.530000e+02   \n",
       "\n",
       "          healItems  winPlacePerc  \n",
       "count  2.745155e+06  2.745155e+06  \n",
       "mean   2.522308e+00           NaN  \n",
       "std    3.939304e+00  0.000000e+00  \n",
       "min    0.000000e+00  0.000000e+00  \n",
       "25%    0.000000e+00  1.999512e-01  \n",
       "50%    1.000000e+00  4.582520e-01  \n",
       "75%    4.000000e+00  7.407227e-01  \n",
       "max    7.700000e+01  1.000000e+00  \n",
       "\n",
       "[8 rows x 26 columns]"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stats = finalDF.describe()\n",
    "stats"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "phJTXvzaU9xK"
   },
   "source": [
    "The simple statistics that we pulled are meaningful because they tell us the shape of our data by attribute. One of the most useful measures is standard deviation, which tells us the spread of our data. The largest standard deviation is around 1,498, which tells us that we have some variables in our data with funky outliers or that operate on a scale much larger than other attributes. Performing PCA will be beneficial to standardize these variables. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## FEATURE VISUALIZATIONS\n",
    "\n",
    "Jump to [Top](#TOP)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "UWrh6P6H6DqQ"
   },
   "source": [
    "### Correlation of attributes\n",
    "\n",
    "We will look at relationship between `winPlacePerc` and other variables. Our **target** variable, `quart_binary`, is derived from the `winPlacePerc` value. For this reason, the next visualizations provide insight on relevant features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 568
    },
    "id": "4oKKoAYtYdvR",
    "outputId": "19700cac-30a0-4e95-a2ef-392694d411cd"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "cmap = sns.diverging_palette(220, 10, as_cmap=True)\n",
    "fig, ax = plt.subplots(figsize = (15, 10))\n",
    "corrMatrix = finalDF.corr()\n",
    "sns.heatmap(corrMatrix, ax=ax, annot=True, fmt=\".1f\", cmap=cmap,\n",
    "            mask=np.tril(np.ones_like(corrMatrix, dtype=bool)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1UPZoHfJDLOS"
   },
   "source": [
    "**Correlation with winPlacePerc:** <br/>\n",
    "\n",
    "1. There is strong correlation between `winPlacePerc` and `weaponsAcquired`, `winPoints`, and `boosts`. A positive correlation means that an increase in each of these attributes is associated with an increase in `winPlacePerc` (the relationship is not 1 to 1, each has a different strength of the relationship). <br/>\n",
    "\n",
    "2. A strong negative correlation exists between `winPlacePerc` and `killPlace`. This tells us that a low `killPlace` (lower is better) is associated with a high `winPlacePerc` (higher is better), which makes sense, a player with a better `killPlace` would rank better. <br/>\n",
    "\n",
    "The strong correlation between these attributes and `winPlacePerc` (driver of our target variable = `quart_binary`) tells us that these are likely the ones that have the most influence on the `winPlacePerc` attribute, and should be included in our final model. \n",
    "\n",
    "**Additional observations** (related to the attributes strongly correlated with `winPlacePerc`: `weaponsAcquired`, `winPoints`, `boosts`, and `killPlace`):\n",
    "1. The `weaponsAcquired` feature does not appear to have a high correlation with any other attributes.\n",
    "2. There's a strong correlation between `winPoints` and `killPoints`.\n",
    "3. A strong correlation exists between `boosts` and `walkDistance` and a strong negative correlation between `boosts` and `killPlace`.\n",
    "4. Another strong negative correlation can be found between `killPoints` and `kills`, `killStreaks`, and `walkDistance`.\n",
    "\n",
    "**Additional observations** (misc):\n",
    "1. We find high negative correlation between `rankPoints` and `killPoints`.\n",
    "2. There's high negative correlation between `rankPoints` and `winPoints`. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### walkDistance, headshotKills, and winPlacePerc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 499
    },
    "id": "swGfQoq8ogUT",
    "outputId": "46c399f0-c6bf-4696-f7fe-94bc973b3694"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3UAAAJNCAYAAACWUFxUAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAADpj0lEQVR4nOz9eZhd1XXn/7/XuWPNk0pjaRaDEAiBJRuwjacEHAdj4thpHKdjhyQOMYk7Q+fbyS9xp9v9y+8hnfSQxPk1pttx7LRjMjpyCGBsjAl4AjGZQYAEmmeVaq66dYezvn+cU/OVVCXVraorfV6PearOPufss++5xfOwvPdey9wdERERERERqU7BfA9AREREREREzp2COhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqWnO8BTMeiRYt8zZo18z0MERERERGRefH000+fdPf2cueqIqhbs2YNO3bsmO9hiIiIiIiIzAsz23e6c1p+KSIiIiIiUsUU1ImIiIiIiFQxBXUiIiIiIiJVrCr21ImIiIiIyIWhUChw8OBBcrncfA9lQcpms3R0dJBKpaZ9j4I6ERERERGZMwcPHqShoYE1a9ZgZvM9nAXF3ens7OTgwYOsXbt22vdp+aWIiIiIiMyZXC5HW1ubAroyzIy2trYZz2IqqBMRERERkTmlgO70zuXdKKgTEREREZEF633vex/d3d1nvObjH/84a9euZcuWLVx77bV873vfG23/+7//+1kdTyKRYMuWLVx55ZV8+MMfZnBwcFb7PxcK6kREREREZMF64IEHaG5uPut1f/RHf8Rzzz3H3XffzS/90i9VbDw1NTU899xzvPjii6TTae65555p3VcsFis2JgV1IiIiIiIyb/7rf/2v/Omf/ikAv/7rv8673/1uAB555BF+5md+hjVr1nDy5En27t3Lxo0b+cVf/EU2bdrETTfdxNDQ0JT+brzxRnbv3j2l/TOf+Qzbtm3jyiuv5BOf+ATuDsDu3bv5kR/5Ea6++mquvfZaXn/9dSAKErdt28bmzZv5/d///bJjf/vb387u3bsZGBjgjjvuYNu2bVxzzTVs374dgL/8y7/kwx/+MO9///u56aab6O/v5+d+7ue46qqr2Lx5M//wD/9w/i8QBXUiIiIiIjKPbrzxRh5//HEAduzYQX9/P4VCgSeeeIK3v/3tE67dtWsXd911Fy+99BLNzc1lg6J//ud/5qqrrprS/iu/8is89dRTvPjiiwwNDXH//fcD8NGPfpS77rqL559/nu9+97ssW7aMhx9+mF27dvHkk0/y3HPP8fTTT/Ov//qvE/orFos8+OCDXHXVVfzBH/wB7373u3nqqad49NFH+a3f+i0GBgYA+N73vscXv/hFvvWtb/Ff/st/oampiRdeeIEf/vCHowHs+VJQJyIiIiIi8+ZNb3oTTz/9NH19fWQyGa6//np27NjB448/PiWoG9k3N3Lf3r17R8/91m/9Flu2bOHee+/l85///JTnPProo7zlLW/hqquu4lvf+hYvvfQSfX19HDp0iJ/4iZ8AohpxtbW1PPzwwzz88MNcc801XHvttbzyyivs2rULgKGhIbZs2cLWrVtZtWoVP//zP8/DDz/M3XffzZYtW3jnO99JLpdj//79APzoj/4ora2tAHzzm9/krrvuGh1TS0vLrLxD1akTEREREZF5k0qlWLNmDV/4whe44YYb2Lx5M48++iivv/46GzdunHBtJpMZ/T2RSExYfvlHf/RHfOhDHyr7jFwuxyc/+Ul27NjBypUr+U//6T+Ry+VGl2BO5u78zu/8Ttm9eSN76iZf/w//8A9cdtllE9p/8IMfUFdXN+G6SmT+1EydiIiIiIjMqxtvvJE//uM/5sYbb+Ttb38799xzD1u2bJm1AGik7tuiRYvo7+8fzYjZ2NhIR0cH//RP/wTA8PAwg4OD3HzzzfzFX/wF/f39ABw6dIjjx4+ftv+bb76ZP/uzPxsNEp999tmy191000189rOfHT3u6uo6788GCupERERERGSevf3tb+fIkSNcf/31LFmyhGw2O2Xp5flobm7mF3/xF7nqqqu47bbb2LZt2+i5v/qrv+JP//RP2bx5MzfccANHjx7lpptu4qd/+qe5/vrrueqqq/jQhz5EX1/fafv/9Kc/TaFQYPPmzVx55ZV8+tOfLnvd7/3e79HV1cWVV17J1VdfzaOPPjorn89ON+W4kGzdutV37Ngx38MQEREREZHztHPnzinLKmWicu/IzJ52963lrtdMnYiIiIiISBVTUCciIiIiIlLFFNSJiIiIiIhUMQV1C4i74148bWpVERERERGRyVSnboFwH8D9ME4XRhuwDLPa+R6WiIiIiIgscArqFgD3YUJ/GRiIjhnEvYuAzZil53dwIiIiIiKyoGn55QLgDDES0I3pj9tFRERERGS+PPTQQ1x22WVs2LCBu+++e8p5d+dTn/oUGzZsYPPmzTzzzDNzPkYFdQuAneZrOF27iIiIiIhUXqlU4q677uLBBx/k5Zdf5itf+Qovv/zyhGsefPBBdu3axa5du7j33nv55V/+5Tkfp6KGBaEGWDSprT1uFxERERGR+fDkk0+yYcMG1q1bRzqd5vbbb2f79u0Trtm+fTs/+7M/i5lx3XXX0d3dzZEjR+Z0nArqFgCzFIFtwLgcYznGRgJbj5m2PIqIiIiITMeXv/wV1qxZTxCkWbNmPV/+8lfOu89Dhw6xcuXK0eOOjg4OHTo042sqTVHDAmGWxWwpsHS+hyIiIiIiUlW+/OWv8IlP3Mng4CAA+/bt5xOfuBOAj370I+fcb7lSY2Y242sqTTN1IiIiIiJS1X73d39vNKAbMTg4yO/+7u+dV78dHR0cOHBg9PjgwYMsX758xtdUmoI6ERERERGpavv3H5hR+3Rt27aNXbt2sWfPHvL5PPfddx+33nrrhGtuvfVWvvSlL+HufP/736epqYlly5ad13NnSssvRURERESkqq1atZJ9+/aXbT8fyWSSz372s9x8882USiXuuOMONm3axD333APAnXfeyfve9z4eeOABNmzYQG1tLV/4whfO65nnwsqtAV1otm7d6jt27JjvYYiIiIiIyHnauXMnGzdunNU+J++pA6itreXee+85rz1186XcOzKzp919a7nrtfxSRERERESq2kc/+hHuvfceVq9ehZmxevWqqg3ozoWWX4qIiIiISNX76Ec/ctEEcZNppk5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5EREREROQ0HnroIS677DI2bNjA3XffPeX8l7/8ZTZv3szmzZu54YYbeP755+d8jMp+KSIiIiIiUkapVOKuu+7iG9/4Bh0dHWzbto1bb72VK664YvSatWvX8thjj9HS0sKDDz7IJz7xCX7wgx/M6Tg1UyciIiIiIlLGk08+yYYNG1i3bh3pdJrbb7+d7du3T7jmhhtuoKWlBYDrrruOgwcPzvk4FdSJiIiIiEjVe/jvdvCTV32Gt7f8Oj951Wd4+O92nHefhw4dYuXKlaPHHR0dHDp06LTXf/7zn+fHfuzHzvu5M6XllyIiIiIiUtUe/rsd/OGn/pbhoQIAxw508Yef+lsAbvrw1nPu192ntJlZ2WsfffRRPv/5z/PEE0+c8/POlWbqRERERESkqn3uMw+MBnQjhocKfO4zD5xXvx0dHRw4cGD0+ODBgyxfvnzKdT/84Q/5hV/4BbZv305bW9t5PfNcKKgTEREREZGqdvxg14zap2vbtm3s2rWLPXv2kM/nue+++7j11lsnXLN//34++MEP8ld/9Vdceuml5/W8c6XllyIiIiIiUtUWd7Rw7MDUAG5xR8t59ZtMJvnsZz/LzTffTKlU4o477mDTpk3cc889ANx555185jOfobOzk09+8pOj9+zYcf77+WbCyq0TXWi2bt3qc/1iRERERERk9u3cuZONGzfOap+T99QBZGpS/Ic//anz2lM3X8q9IzN72t3LfhjN1ImIiIiISFUbCdw+95kHOH6wi8UdLfzSf3xfVQZ050JBnYiIiIiIVL2bPrz1ogniJlOiFBERERERkSqmoE5ERERERKSKVTSoM7NmM/t7M3vFzHaa2fVm1mpm3zCzXfHP80tJIyIiIiIichGr9EzdnwAPufvlwNXATuC3gUfc/RLgkfhYREREREREzkHFgjozawRuBD4P4O55d+8GPgB8Mb7si8BtlRqDiIiIiIjI+XjooYe47LLL2LBhA3ffffdpr3vqqadIJBL8/d///RyOLlLJmbp1wAngC2b2rJn9HzOrA5a4+xGA+OfiCo5BRERERETknJRKJe666y4efPBBXn75Zb7yla/w8ssvl73uP/yH/8DNN988D6OsbFCXBK4F/pe7XwMMMIOllmb2CTPbYWY7Tpw4UakxioiIiIiIlPXkk0+yYcMG1q1bRzqd5vbbb2f79u1TrvuzP/szfvInf5LFi+dnvqqSQd1B4KC7/yA+/nuiIO+YmS0DiH8eL3ezu9/r7lvdfWt7e3sFhykiIiIiItVu5/0vcO97/oT/tukz3PueP2Hn/S+cd5+HDh1i5cqVo8cdHR0cOnRoyjVf/epXufPOO8/7eeeqYkGdux8FDpjZZXHTe4CXga8BH4vbPgZMDXVFRERERESmaef9L/Dwf7yfviM94NB3pIeH/+P95x3YufuUNjObcPxrv/Zr/OEf/iGJROK8nnU+khXu/1eBL5tZGngD+DmiQPJvzezngf3Ahys8BhERERERuYA9/j++RTFXmNBWzBV4/H98i423XHXO/XZ0dHDgwIHR44MHD7J8+fIJ1+zYsYPbb78dgJMnT/LAAw+QTCa57bbbzvm5M1XRoM7dnwO2ljn1nko+V0RERERELh59R3tm1D5d27ZtY9euXezZs4cVK1Zw33338dd//dcTrtmzZ8/o7x//+Me55ZZb5jSgg8rP1ImIiIiIiFRUw9KmaOllmfbzkUwm+exnP8vNN99MqVTijjvuYNOmTdxzzz0A87qPbjwrt050odm6davv2LFjvochIiIiIiLnaefOnWzcuHF2+4z31I1fgpnMprjpM7ec1/LL+VLuHZnZ0+5ebhWkZupERERERKS6jQRuj/+Pb9F3tIeGpU28/dffXZUB3blQUCciIiIiIlVv4y1XXTRB3GSVrFMnIiIiIiIiFaagTkREREREpIopqBMREREREaliCupERERERESqmII6ERERERGR03jooYe47LLL2LBhA3fffXfZa7797W+zZcsWNm3axDve8Y45HqGyX4qIiIiIiJRVKpW46667+MY3vkFHRwfbtm3j1ltv5Yorrhi9pru7m09+8pM89NBDrFq1iuPHj8/5ODVTJyIiIiIiUsaTTz7Jhg0bWLduHel0mttvv53t27dPuOav//qv+eAHP8iqVasAWLx48ZyPU0GdiIiIiIhUvb5Hn2T/x3+XPT/+y+z/+O/S9+iT593noUOHWLly5ehxR0cHhw4dmnDNa6+9RldXF+985zt505vexJe+9KXzfu5MafmliIiIiIhUtb5Hn6TzT7+MD+cBKB0/ReeffhmAhne9+Zz7dfcpbWY24bhYLPL000/zyCOPMDQ0xPXXX891113HpZdees7PnSnN1ImIiIiISFXr+uL20YBuhA/n6fri9tPcMT0dHR0cOHBg9PjgwYMsX758yjXvfe97qaurY9GiRdx44408//zz5/XcmVJQJyIiIiIiVa104tSM2qdr27Zt7Nq1iz179pDP57nvvvu49dZbJ1zzgQ98gMcff5xiscjg4CA/+MEP2Lhx43k9d6a0/FJERERERKpaor2V0vGpAVyivfW8+k0mk3z2s5/l5ptvplQqcccdd7Bp0ybuueceAO688042btzIe9/7XjZv3kwQBPzCL/wCV1555Xk9d6as3DrRhWbr1q2+Y8eO+R6GiIiIiIicp507d876TNbkPXUAlknT9qmPnteeuvlS7h2Z2dPuvrXc9ZqpExERERGRqjYSuHV9cTulE6dItLfS8rEPVGVAdy4U1ImIiIiISNVreNebL5ogbjIlShEREREREaliCupERERERESqmII6ERERERGRKqagTkREREREpIopqBMRERERETmNhx56iMsuu4wNGzZw9913Tznf09PD+9//fq6++mo2bdrEF77whTkfo4I6ERERERGRMkqlEnfddRcPPvggL7/8Ml/5yld4+eWXJ1zz53/+51xxxRU8//zzfPvb3+Y3f/M3yefzp+mxMhTUiYiIiIiIlPHkk0+yYcMG1q1bRzqd5vbbb2f79u0TrjEz+vr6cHf6+/tpbW0lmZzbynGqUyciIiIiIlXPX38KnrkfBrqgrgWuvQVbv+28+jx06BArV64cPe7o6OAHP/jBhGt+5Vd+hVtvvZXly5fT19fH3/zN3xAEczt3ppk6ERERERGpav76U/Dd+6KADqKf370vaj+fft2ntJnZhOOvf/3rbNmyhcOHD/Pcc8/xK7/yK/T29p7Xc2dKQZ2IiIiIiFS3Z+6HUmFiW6kQtZ+Hjo4ODhw4MHp88OBBli9fPuGaL3zhC3zwgx/EzNiwYQNr167llVdeOa/nzpSCOhERERERqW4jM3TTbZ+mbdu2sWvXLvbs2UM+n+e+++7j1ltvnXDNqlWreOSRRwA4duwYr776KuvWrTuv586U9tSJiIiIiEh1q2spH8DVtZxXt8lkks9+9rPcfPPNlEol7rjjDjZt2sQ999wDwJ133smnP/1pPv7xj3PVVVfh7vzhH/4hixYtOq/nzpSVWye60GzdutV37Ngx38MQEREREZHztHPnTjZu3DirfY7uqRu/BDORghtuP+9kKfOh3Dsys6fdfWu56zVTJyIiIiIiVc3Wb8Nh1rNfVgsFdSIiIiIiUvVs/Ta4SIK4yZQoRUREREREpIopqBMREREREaliCupERERERESqmII6ERERERGRKqagTkRERERE5DTuuOMOFi9ezJVXXln2vLvzqU99ig0bNrB582aeeeaZOR6hgjoREREREZHT+vjHP85DDz102vMPPvggu3btYteuXdx777388i//8hyOLqKgTkRERERE5DRuvPFGWltbT3t++/bt/OzP/ixmxnXXXUd3dzdHjhyZwxGqTp2IiIiIiFwAwvAYzhvAMJDBWEcQLKn4cw8dOsTKlStHjzs6Ojh06BDLli2r+LNHKKgTEREREZGqFgV0rwJh3DKM8yphSMUDO3ef0mZmFX3mZFp+KSIiIiIiVS2aoQsntYZxe2V1dHRw4MCB0eODBw+yfPnyij93PAV1IiIiIiJS5YZn2D57br31Vr70pS/h7nz/+9+nqalpTpdegpZfioiIiIhI1ctQPoDLnHfPH/nIR/j2t7/NyZMn6ejo4D//5/9MoVAA4M477+R973sfDzzwABs2bKC2tpYvfOEL5/3MmVJQJyIiIiIiVc1YN2lPHUCAse68+/7KV75y5meb8ed//ufn/ZzzoaBORERERESqWhAsIQyZl+yXC4GCOhERERERqXpRAHdxBHGTKVGKiIiIiIhIFVNQJyIiIiIic6pcbTeJnMu7UVAnIiIiIiJzJpvN0tnZqcCuDHens7OTbDY7o/u0p05EREREROZMR0cHBw8e5MSJE/M9lAUpm83S0dExo3sU1ImIiIiIyJxJpVKsXbt2vodxQdHySxERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKJSvZuZntBfqAElB0961m1gr8DbAG2Av8lLt3VXIcIiIiIiIiF6q5mKl7l7tvcfet8fFvA4+4+yXAI/GxiIiIiIiInIP5WH75AeCL8e9fBG6bhzGIiIiIiIhcECod1DnwsJk9bWafiNuWuPsRgPjn4gqPQURERERE5IJV0T11wFvd/bCZLQa+YWavTPfGOAj8BMCqVasqNT4REREREZGqVtGZOnc/HP88DnwVeDNwzMyWAcQ/j5/m3nvdfau7b21vb6/kMEVERERERKpWxYI6M6szs4aR34GbgBeBrwEfiy/7GLC9UmMQERERERG50FVy+eUS4KtmNvKcv3b3h8zsKeBvzezngf3Ahys4BhERERERkQtaxYI6d38DuLpMeyfwnko9V0RERERE5GIyHyUNREREREREZJYoqBMREREREaliCupERERERESqmII6ERERERGRKqagTkREREREpIopqBMREREREaliCupERERERESqmII6ERERERGRKqagTkREREREpIopqBMREREREaliCupERERERESqmII6ERERERGRKqagTkREREREpIopqBMREREREaliCupERERERESqmII6ERERERGRKqagTkREREREpIopqBMREREREaliCupERERERESqmII6ERERERGRKqagTkREREREpIopqBMREREREaliCuoqxL2Ee2m+hyEiIiIiIhe45HwP4EITBXJdhL4fcIxVGC2Y6VWLiIiIiMjs00zdrOsh9BeBXqAP95eA7vkdkoiIiIiIXLAU1M2y0I+VaTuEu8/DaERERERE5EKnoG6WWdkVrVp6KSIiIiIilaGgbpaZLWHiazUCW4GZzdeQRERERETkAqYppFnXQGBbcD8FOGatQON8D0pERERERC5QCupmWTQj14iZAjkREREREak8Lb8UERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqpiCOhERERERkSqmoE5ERERERKSKKagTERERERGpYgrqREREREREqljFgzozS5jZs2Z2f3zcambfMLNd8c+WSo9BRERERETkQjUXM3X/Dtg57vi3gUfc/RLgkfhYREREREREzkFFgzoz6wB+HPg/45o/AHwx/v2LwG2VHIOIiIiIiMiFrNIzdf8T+H+AcFzbEnc/AhD/XFzhMYiIiIiIiFywKhbUmdktwHF3f/oc7/+Eme0wsx0nTpyY5dGJiIiIiIhcGCo5U/dW4FYz2wvcB7zbzP4vcMzMlgHEP4+Xu9nd73X3re6+tb29vYLDFBERERERqV4VC+rc/XfcvcPd1wC3A99y958BvgZ8LL7sY8D2So1BRERERETkQjcfderuBn7UzHYBPxofi4iIiIiIyDlIzsVD3P3bwLfj3zuB98zFcy807iFgmNl8D0VERERERBaIOQnq5Py4D+PeiXMEox5YjlnDfA9LREREREQWAAV1C5y7434IZ390TB/uJwi4BrO6eR6diIiIiIjMt/nYUyczksM5OKmtiPvAvIxGREREREQWFgV1C57F/0xq1b46ERERERFBQd2CZ5bFWDOpNQ1o6aWIiIiIiGhPXVUwW4qRJfSTGLWYLcKsdr6HJSIiIiIiC4CCuipglgLaSVj7fA9FREREREQWGC2/FBERERERqWIK6kRERERERKqYgjoREREREZEqpqBORERERESkiimoExERERERqWIK6kRERERERKqYgjoREREREZEqpqBORERERESkiimoExERERERqWIK6kRERERERKqYgjoREREREZEqpqBORERERESkiimoExERERERqWIK6kRERERERKqYgjoREREREZEqpqBORERERESkiimoExERERERqWIK6kRERERERKrYtII6M6szsyD+/VIzu9XMUpUdmoiIiIiIiJzNdGfq/hXImtkK4BHg54C/rNSgREREREREZHqmG9SZuw8CHwT+zN1/AriicsMSERERERGR6Zh2UGdm1wMfBf4lbktWZkgiIiIiIiIyXdMN6n4N+B3gq+7+kpmtAx6t2KhERERERERkWqY12+bujwGPjTt+A/hUpQYlIiIiIiIi03PGoM7M/hnw051391tnfUQiIiIiIiIybWebqfvjORmFiIiIiIiInJMzBnXxsksRERERERFZoM62/PIFzrz8cvOsj0hERERERESm7WzLL2+Zk1GIiIiIiIjIOTnb8st9czUQERERERERmbmzLb98wt3fZmZ9TFyGaYC7e2NFRyciIiIiIiJndLaZurfFPxsmnzOzTKUGVS3cHRjCKWBkOZ9X4j6Mk8NIATWY2ayNU0RERERELlzBmU6a2adP094IfL0iI6oS7iXcjxL6DtyfJfSnce85x756Cf2ZuJ8duB/GvTTLIxYRERERkQvRGYM64O1m9gfjG8xsKfA48GjFRlUVBnFeBcL4OE/oO3EfnlEv7nlCfwUYuS/E2QUMzN5QRURERETkgnW2oO5W4Goz++8A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EKH/EMhHx/TjPkDAJZgl5nlsBULfBXSNG9tJArZglpnXsYmIiIiILCRafnkRcx9gJKAbcxTIzcNoJnJyjAR0Y4ZwH5yP4YiIiIiILFgK6i5iZlauNf5nftlpxlB+zCIiIiIiFy8FdRe1eqB2QovRAWTnZTQTZYGlk9oamTxeEREREZGLnfbUXcTMMgRcifspnH7MWjGaMZv/WN8sScBa3FtwujAaovFZer6HJiIiIiKyoCiou8iZ1WK2MGe/zDKYLQGWzPdQREREREQWrIpNyZjZSjN71Mx2mtlLZvbv4vZWM/uGme2Kf7ZUagwXOvcQLw3gpX7cw/kejoiIiIiIzINKrrMrAr/p7huB64C7zOwK4LeBR9z9EuCR+FhmyMMcPvwqYf+jhP3fxodewsOh+R6WiIiIiIjMsYoFde5+xN2fiX/vA3YCK4APAF+ML/sicFulxnAh8+JJfHg34IDjhb144dh8D0tERERERObYnGTEMLM1wDXAD4Al7n4EosAPWDwXY7jQeOFombaDuPs8jEZEREREROZLxYM6M6sH/gH4NXfvncF9nzCzHWa248SJE5UbYJWyRFOZthbVcRMRERERuchUNKgzsxRRQPdld//HuPmYmS2Lzy8Djpe7193vdfet7r61vb29ksOsSpZaClYzriGDpVfO34BERERERGReVKykgUVTRp8Hdrr7fx936mvAx4C745/bKzWGC5klGgjqboCwD3fHEg1Yom6+hyUiIiIiInOsknXq3gr8W+AFM3subvv/EAVzf2tmPw/sBz5cwTFc0CxRC4latOBSREREROTiVbGgzt2fgNPGG++p1HPninseGMC9tKALeM+El/IweBKKg5Bpgpo2zOYkl86sier1DeCewywD1GJWyf/vQkRERERkfum/ds+B+xChvwL0xMdJAq7GrGF+B3YevDQMh38AR3dEDRbA+lugZcP8DmwG3B3nOO6vxMdgrANWYJaY38GJiIiIiFRIdU3DLBBREs+ecS1FQt+Le2m+hnT+BjvHAjoAD2HvN/DhaScsXQCGcH9tQovzBjA4P8MREREREZkDCurOSbkgoR+o4qCuOFCmbQhKubkfyzlyCkA4td3zcz8YEREREZE5oqDuHJg1Tm2jHUjN/WBmS6aZKVsgs62Qqp4lpUYGSE9qDbDxpR9ERERERC4wCurOSSPGGsaCoFbM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\n",
      "text/plain": [
       "<Figure size 1080x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (15, 10))\n",
    "sns.scatterplot(data = finalDF, x=\"walkDistance\", y=\"headshotKills\",\n",
    "                hue=\"winPlacePerc\", sizes=(20, 200), palette=\"magma\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CSk53OR9wfEV"
   },
   "source": [
    "Distance walked has the highest correlation with `winPlacePerc`. This makes sense, because if a player stays alive longer, they likely also walk a greater distance than players that do not survive for very long. Additionally, these players (that stay alive longer and walk greater distance) also are likely place higher. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **MODELING AND EVALUATION SECTIONS SUMMARY** #\n",
    "*Assignment (from course overview document): Different tasks will require different evaluation methods. Be as thorough as possible when analyzing the data you have chosen and use visualizations of the results to explain the performance and expected outcomes whenever possible.* <br/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Option A: Cluster Analysis\n",
    "* ***Train:** Perform cluster analysis using several clustering methods (adjust parameters).* <br/>\n",
    "* ***Eval:** Use internal and/or external validation measures to describe and compare the clusterings and the clusters— how did you determine a suitable number of clusters for each method?* <br/>\n",
    "* ***Visualize:** Use tables/visualization to discuss the found results. Explain each visualization in detail.* <br/>\n",
    "* ***Summarize:** Describe your results. What findings are the most interesting and why?* <br/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**We will need to organize content to include MORE THAN 1    cluster approaches in multiple sections.** <br/>\n",
    "Need to have all sections for each cluster approach: (1) TRAIN, (2) EVAL, (3) VISUALIZE, (4) SUMMARIZE."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='ME1'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **MODELING AND EVALUATION 1**\n",
    "\n",
    "Jump to [Top](#TOP)\n",
    "\n",
    "*Assignment: Train and adjust parameters.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## TRAIN-TEST SPLIT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Data shape:    (1921608, 25)\n",
      "Validating Data shape:  (549031, 25)\n",
      "Test Data shape:        (274516, 25)\n"
     ]
    }
   ],
   "source": [
    "# TRAIN-TEST-SPLIT-0\n",
    "# reference: https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn\n",
    "\n",
    "train_ratio = 0.70\n",
    "validation_ratio = 0.20\n",
    "test_ratio = 0.10\n",
    "\n",
    "# creation of feature-space and response-vectors\n",
    "dataX = finalDF.loc[:, cols_df]\n",
    "dataY = finalDF.loc[:, ('winPlacePerc','quart_binary')]\n",
    "\n",
    "# train is now 70% of the entire data set\n",
    "x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=(1-train_ratio), random_state=17)\n",
    "\n",
    "# test is now 10% of the initial data set\n",
    "# validation is now 20% of the initial data set \n",
    "x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,\n",
    "                                                test_size=test_ratio/(test_ratio + validation_ratio), random_state=117) \n",
    "\n",
    "print(\"Training Data shape:   \", x_train.shape)\n",
    "print(\"Validating Data shape: \", x_val.shape)\n",
    "print(\"Test Data shape:       \", x_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Evaluation of Index Selection\n",
      "Head:\n",
      " Int64Index([1764352, 83992, 965563, 1672690, 3041784], dtype='int64') \n",
      "Tail:\n",
      " Int64Index([3198970, 1694316, 2366209, 1797295, 1310957], dtype='int64')\n"
     ]
    }
   ],
   "source": [
    "# TRAIN-TEST-SPLIT-1\n",
    "# look at VALIDATE data to confirm that the train-test-split has randomized the data order\n",
    "\n",
    "print(\"Evaluation of Index Selection\\nHead:\\n\",x_val.head().index,\n",
    "      \"\\nTail:\\n\",x_val.tail().index)\n",
    "\n",
    "# confirmed = test_train_split has randomized the data order"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tuning Data shape:  (50000, 25)\n",
      "(697494, 2)\n",
      "(2047661, 2)\n",
      "(12653, 2)\n",
      "(37347, 2)\n"
     ]
    }
   ],
   "source": [
    "# TRAIN-TEST-SPLIT-2\n",
    "# create TUNE data (x_tune, y_tune) = subset of TRAIN \n",
    "indices = x_train.sample(n=50000, replace=False, random_state=17).index\n",
    "x_tune = x_train.loc[indices, :]\n",
    "y_tune = y_train.loc[indices, :]\n",
    "\n",
    "print(\"Tuning Data shape: \", x_tune.shape)\n",
    "print(dataY[dataY.quart_binary=='1'].shape)\n",
    "print(dataY[dataY.quart_binary=='0'].shape)\n",
    "print(y_tune[y_tune.quart_binary=='1'].shape)\n",
    "print(y_tune[y_tune.quart_binary=='0'].shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-block alert-info\">\n",
    "<b><h3>Please Consider the \"Exploratory Analysis\" below as a continuation of Data Understanding 2</h3>\n",
    "    </b>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### KMEANS CLUSTERS: EXPLORATORY ANALYSIS\n",
    "\n",
    "Jump to [Top](#TOP)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Note: We will use Random Forest as our classification model type."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### K-means baseline model (Random Forest classifier)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Build RF Classifier model to use as our baseline (to compare with models we build using cluster-variables)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline average accuracy =  91.916 +- 0.27782008566696637\n"
     ]
    }
   ],
   "source": [
    "# reference: ML Notebook 9: Clustering and Discretization\n",
    "# updated 03-30-22 - to run this using SUBSET OF DATA = JUST USE TUNE\n",
    "\n",
    "from sklearn.model_selection import StratifiedKFold, cross_val_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "X = x_tune.loc[:, cols_df]\n",
    "y = y_tune.loc[:, ('quart_binary')]\n",
    "cv = StratifiedKFold(n_splits=10)\n",
    "\n",
    "clf = RandomForestClassifier(n_estimators=150, random_state=17)\n",
    "\n",
    "acc = cross_val_score(clf, X=X, y=y, cv=cv)\n",
    "\n",
    "print (\"Baseline average accuracy = \", acc.mean()*100, \"+-\", acc.std()*100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### K-means clusters: Ride distance with walk distance\n",
    "\n",
    "#### Plot: rideDistance + walkDistance"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Reference: notebook 09* <br/>\n",
    "Okay, now let's start with a bit of feature engineering. We will start by using kmeans on `rideDistance` and `walkDistance` together."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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i3juueewbds2nDlzpsb3ITs7G0uXLoXD4UBaWhqeffbZ+v6KYXPdk82bN+OOO+7AL37xC/covl+/fpg+fXrE7wkDASIioijGpQEiIqIoxkCAiIgoijEQICIiimIMBIiIiKIYAwEiIqIoxkCAqIG5ePEifv3rXwMAZs+ejU8++cTnmi5dumDkyJEYOXIk7r33XkybNg3nzp0DABQWFuJ///d/A76/Xq/HU089FZnGV8HhcGDatGkwm8318vkuBw8exPjx4wFUbon98ccfa/yeXbp0AQDs3r0bH330UY3fj6iuNP3SeERN1Keffur+782bN2PSpEn4/PPP0apVK7z77rsBX1deXo4TJ07URRN9bN68GWlpaYiJiamXz/dnyZIltfp+w4YNw4QJE3DvvfciMTGxVt+bKBIYCBBFyIgRI7Bq1SrccsstmDlzJrRaLRYsWICjR49i3bp1WLt2LebPn4///Oc/uHLlCrp06YKVK1f6fS+z2YyJEydi+PDheOSRR3yeHzt2LD766CN888036NKlCyZMmIA9e/bgH//4B9577z3I5XK0adMGy5cvx+LFi1FUVISnnnoKb7/9Nt544w1kZWWhvLwcycnJeOONN9CyZUukpaUhPT0dR44cgVwux6pVq9C2bVt89913ePXVVyGKIm666SasWLECMTExeO2113Do0CE4HA488MADeOyxxyRtFEURmZmZ+PjjjwFUznZotVrk5uaisLAQTz31FB588EGYzWbMmTMHJ0+ehCAImDRpEkaNGoVPPvkEO3bsQFlZGYYOHYqioiLExMTgp59+wrVr1/Dcc8/h008/RV5eHu6++27Mnj0bBoMBL730EgoLC1FUVIQ77rjDp+MfP348pk2bhpMnT2L79u0AgIqKCly4cAF79+6FyWTC/PnzUVZWhmbNmmHu3Lm49dZbcfHiRWRkZMBkMuG2226TvOewYcOwceNGTJ8+vbr/fIjqDJcGiCJk8ODB7pO/8vPz8cMPPwAAvvnmGwwZMgRHjx6FUqnE1q1b8X//93/Q6/XukqGebDYbpk2bhvT0dL9BgEunTp3w888/Sx5btWoV1q9fj08++QQ333wzfv75Z8yZMwfJycl4++23ce7cOfz888/YsmULdu3ahV/84hf4+9//DgAoLi7GHXfcgZ07d6Jfv37YuHEjrFYrnn/+eSxbtgz/+Mc/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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# reference: notebook 09\n",
    "from matplotlib import pyplot as plt\n",
    "# changed this from \"ggplot\" to \"seaborn\"\n",
    "plt.style.use(\"seaborn\") \n",
    "\n",
    "# reference: another type of style = \"fivethirtyeight\" \n",
    "# plt.style.use(\"fivethirtyeight\") \n",
    "\n",
    "# %matplotlib inline\n",
    "x_ride_walk = x_tune[['rideDistance','walkDistance']].values\n",
    "\n",
    "plt.scatter(x_ride_walk[:, 1], x_ride_walk[:, 0]+np.random.random(x_ride_walk[:, 1].shape)/2, \n",
    "             s=20)\n",
    "plt.xlabel('walkDistance (normalized)'), plt.ylabel('rideDistance')\n",
    "plt.grid()\n",
    "plt.title('Ride vs Walk')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There doesn't appear to be a strong correlation between `rideDistance` and `walkDistance`. Not sure this would be very useful."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Accuracy: rideDistance + walkDistance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average accuracy (with kmeans for ride/walkDistance)=  91.82400000000001 +- 0.29445542956447535\n"
     ]
    }
   ],
   "source": [
    "# reference: notebook 09\n",
    "\n",
    "from sklearn.cluster import KMeans\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "x_ride_walk = x_tune[['rideDistance','walkDistance']]\n",
    "\n",
    "cls = KMeans(n_clusters=8, init='k-means++',random_state=17)\n",
    "cls.fit(x_ride_walk)\n",
    "newfeature = cls.labels_ # the labels from kmeans clustering\n",
    "\n",
    "cv_ride_walk = StratifiedKFold(n_splits=10)\n",
    "\n",
    "x_ride_walk = x_tune.loc[:, cols_df]\n",
    "y_ride_walk = y_tune.loc[:, ('quart_binary')]\n",
    "x_ride_walk = np.column_stack((x_ride_walk,pd.get_dummies(newfeature)))\n",
    "\n",
    "acc_ride_walk = cross_val_score(clf,x_ride_walk,y=y_ride_walk,cv=cv_ride_walk)\n",
    "\n",
    "print (\"Average accuracy (with kmeans for ride/walkDistance)= \", acc_ride_walk.mean()*100, \"+-\", acc_ride_walk.std()*100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It does not look like this first pass at clusters has any positive impact on our performance. Accuracy did not improve (actually went down slightly), standard deviation did not go down (it went up just a bit). \n",
    "\n",
    "For reference: baseline model has accuracy 91.916 +- 0.27782. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we took a look at the data shapes in the above section (Evaluate outliers), to see if there are any attributes that might be good for clusters. <br/>\n",
    "\n",
    "It looks like `matchDuration` has a couple groupings. It also looks like `numGroups` has 3 clusters. <br/>\n",
    "Let's take a look at them. <br/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### K-means clusters: Match duration with walk distance\n",
    "\n",
    "#### Plot: matchDuration + walkDistance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfwAAAFlCAYAAAAOO1qYAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAABx4UlEQVR4nO3deXwT1fo/8M9kbZqkpC0tiFD2UoRbFgFRFMWtKiKgorRaroILyKKgFfQioKBXZFFBwQte5VpZZZULX9EfXkEREUGWVkoFylKkC6UtWdqs8/sjZJqZTNKkTdq0ed6v1/f7Islkcmbw8sw55znPYViWZUEIIYSQZk3S2A0ghBBCSOhRwCeEEEIiAAV8QgghJAJQwCeEEEIiAAV8QgghJAJQwCeEEEIiAAV8Quro2LFjmDVrls9jDhw4gAcffFD0s6VLl2LgwIEYPnw4hg8fjqFDh2LatGk4e/ZsyNp5/PhxTJkyJajnbwybN2/G888/DwDIzMzEN99843HMjBkzcNttt3H394EHHsCsWbNQWloKACguLsbo0aN9/s6FCxcwefLk4F8AIY2AAj4hdXTq1CkUFxfX6xwPPPAAtm3bhm3btmHHjh0YPHgw/v73v8NgMASplfx2/u1vf8OSJUuCdu5w99RTT/Hub5s2bfDMM8/AbrejVatWWLdunc/v//XXXygoKGig1hISWhTwSUQ6cOAAHn/8cbz00ksYPnw4Ro8eje+//x5PP/007rjjDrzzzjsAAIfDgXnz5mHUqFF44IEHcP/99+PQoUO4dOkSlixZgt9++w2vvfYaAGDjxo0YOnQohg0bhjFjxuDSpUsAAJPJhKlTp2L48OG477778Ntvv3lt14gRI9C5c2ds374dANCtWzdcuXKF+9z1+sCBA3jooYcwevRoDBs2DBaLxa92uo846PV6vPLKK3jwwQcxbNgwvPfee7DZbACcDwZLly7F6NGjceedd2LNmjUebV2/fj3Gjx/PvT59+jRuu+022O12LFmyBMOGDcPDDz+McePGoaSkhPfdEydO4Pbbb+dejxs3DtOnTwcAWCwW3HTTTdDr9di4cSNGjRqFESNGYMiQIaLtcLHZbHjxxRfx8ssvc9fhjmEYjB8/HtXV1di3bx8KCwvRp08fru2jR4/Gww8/jJEjR2L16tWw2+2YOXMmzp8/j3HjxgEAPvnkE4waNQrDhg3D3Xffje+++w6Ac7RmxowZGDduHO677z78/e9/5665oKAAmZmZ3H8bO3fuBOAcYZg4cSIefvhhDBs2DJ988onXayMkGCjgk4h1/PhxPPfcc9i2bRs0Gg1WrFiBf/3rX9i8eTPWrFmD4uJiHD16FCUlJVi/fj127tyJkSNHYuXKlbjuuuswZcoU9OvXD//85z+Rl5eHhQsX4tNPP8X27dtx5513Yvny5QCAoqIirqc5evRoLF261Ge7unXrhvz8/Frb/+eff2LRokXYvn07cnNz/Wqnu3nz5kGn02H79u3YtGkTTp48ic8++wyAM+jGxsZi3bp1WLJkCf75z3/CbDbzvj906FAcOnSIGyLfvHkzHn74YZSUlOA///kPNm3ahM2bN2PQoEE4duwY77vdu3eHTCZDfn4+qqurcebMGfzyyy8AgP379yM1NRUSiQRfffUVVqxYga1bt+L999/HggULRO+F1WrFiy++iPj4eCxcuBAymSyg+/vvf/8bd955JzZv3owVK1bgt99+A8MwmDdvHpKSkvDvf/8bFy9exM8//4zs7Gxs374dU6dO5Y2W/Pbbb/jwww/xzTffQKVScaMH06ZNw3333YcdO3ZgxYoVWLx4MQwGA7KysvDII49g8+bN2LhxI37++WfuYYCQUPD+vwpCmrm2bdvihhtuAAAkJSVBq9VCoVAgLi4OarUalZWV6NOnD1q0aIF169bhwoULOHDgANRqtce59u/fj1tvvRXXXXcdAOdQMuAcSWjXrh169eoFAEhJScGmTZt8tothGERFRdXa/uuuuw7XX389APjdTnd79+7F2rVrwTAMFAoFRo8ejf/85z947rnnAAB33XUXAKBHjx6wWCwwmUxQKpXc9zUaDe655x58/fXXeOqpp7B9+3asXr0arVq1QkpKCkaOHInBgwdj8ODBuPnmmz1+/5577sHevXvRtWtXDBw4ECdPnsSff/6J3bt3495774VarcYnn3yCPXv24OzZs8jLy4PJZBK9lvnz58NoNOK7774DwzA+r5thGKhUKo+2TJ8+HceOHcPNN9+MmTNnQiLh94euv/56vPfee9i+fTvOnTuHo0ePwmg0cp8PGDAAGo0GAHDDDTegsrISFRUVyMvLw6hRowA4/87+3//7fzCZTDh48CAqKyvx4YcfAnCOBOXl5eGBBx7w2X5C6op6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zR7HdOvWDcOHD8fw4cNx//33Y9KkSTh37hwAoLi4GM8++6zX8+v1ekycODE0ja+F3W7HpEmTUFVV1Si/73LgwAFkZmYCcC41PX78eL3P2a1bNwDAt99+iy+//LLe5yOkoTTv8nGENAPbtm3j/rx27VqMGzcOO3fuRKtWrbBy5Uqv36usrMSJEycaooke1q5di1tvvRUqlapRfl/M22+/HdTz3XvvvRgzZgzuv/9+xMfHB/XchIQCBXxC6mnYsGH44IMP0LlzZ7z88svQaDR488038fvvv2P58uVYtmwZ5syZgz///BOXL19Gt27dsHjxYtFzVVVVYezYsXjwwQfxxBNPeHyenp6OL7/8Ej/++CO6deuGMWPG4Pvvv8f27dvx6aefQiqVom3btliwYAHmzZuHkpISTJw4ER9//DHef/997N+/H5WVlUhMTMT777+Pli1b4tZbb0VaWhoOHToEqVSKDz74AO3atcPPP/+Md999FyzLok2bNli0aBFUKhXee+89/Prrr7Db7Xj44Yfx1FNP8drIsiyys7OxceNGAM7RC41Gg9zcXBQXF2PixIl45JFHUFVVhZkzZ+LkyZNgGAbjxo3DiBEjsHnzZmzZsgUVFRUYMmQISkpKoFKp8Mcff+Dq1auYNm0atm3bhry8PNx9992YMWMGDAYDXn/9dRQXF6OkpAQ333yzR4DPzMzEpEmTcPLkSWzatAkAUF1djQsXLmDPnj0wmUyYM2cOKioqEBUVhTfeeAM33HADCgsLkZWVBZPJhF69evHOee+992L16tWYMmVKXf/zIaTB0JA+IfV0++23c7tU5efn4/DhwwCAH3/8EXfccQd+//13yOVyrF+/Ht999x30ej1XJtOd1WrFpEmTkJaWJhrsXbp06YIzZ87w3vvggw/w2WefYfPmzbj++utx5swZzJw5E4mJifj4449x7tw5nDlzBuvWrcOuXbtw3XXX4euvvwYAlJaW4uabb8bWrVvRv39/rF69GhaLBa+88grmz5+P7du3Izk5GVu2bMGGDRsAAFu2bMHGjRuxe/du/Pbbb7y25OXlQavV8jYBKSoqwpo1a7B8+XK89957AIClS5ciNjYW//3vf/Gf//wHS5cuRV5eHgDndMWWLVswbdo0AEBJSQnWr1+P5557Dq+99hrefPNNbN26FRs2bIBer8cPP/yA7t27Y/369di1axcOHjyI3Nxc0fs3ZswYbNu2DVu3bkXXrl0xbdo0JCQkYPr06cjKysKWLVswd+5crmrZ3Llz8fDDD2Pbtm3o27cv71z9+vXD999/7/XvipBwQj18Qurp9ttvx6pVqzBw4EAuGJeVlWHv3r1YsmQJ2rRpA51Oh9WrV+PMmTM4e/YsTCaTx3k+/PBDSCQSfPTRRz5/j2EYREVF8d4bMmQI0tPTcffddyMtLQ3du3dHYWEh93n79u0xffp0fPXVVygoKMCRI0eQlJTEfe7aEbBr16747bffcPLkSbRq1YrbvvPll18GAEyZMgUnTpzAL7/8AgAwmUw4efIkr4b32bNn0bp1a177Bg0aBIZhkJycjIqKCgDAL7/8gnfeeQcAEBcXh7vuugu//vorNBoNbrjhBt6GVYMHDwYAtGnTBl27duWG0HU6HSorK/Hggw/i2LFjWLVqFc6cOYOKigrRe+zuww8/hFwuxzPPPAOj0YicnBy89tpr3Ocmkwnl5eX49ddfsWjRIgDAQw89hJkzZ3LHXH/99VxOBSHhjgI+IfXUp08fzJgxAz///DMGDBiA+Ph4fPPNN7DZbGjTpg12796NJUuWYMyYMXj44YdRXl4OsYrWQ4cOhclkwpIlSzB9+nSvv3fy5Ek8/vjjvPdmzpyJvLw87NmzB1lZWZg0aRJvA5acnBy8/PLLeOqpp5CWlgaJRMJrg1KpBOB8mGBZFnK5nNvSFHAmABqNRtjtdmRlZeHee+8FAFy5cgVqtZrXFoZhPHaXdD+/i/AesCwLu90OAB4PNHK5nPuz2M6V2dnZ2LVrFx577DHccsst3D4Y3nzzzTf43//+h3Xr1gFw7p+hUCh4+RJFRUXQ6XS8tjIMw9v9TSaT8a6JkHBGQ/qE1JNMJkNqaiqys7MxYMAADBw4EJ988gluv/12AMD+/ftx//3345FHHkFMTAwOHDjABTZ33bt3R1ZWFrZv3+412W7NmjVgGAY33XQT957NZsO9996L2NhYPP/88xg+fDhOnDgBmUwGm80GADh48CAGDBiA9PR0dOjQAT/88INoG1w6duyIsrIynDp1CgDw6aefYu3atRg4cCA2bNgAq9UKo9GIjIwMHDlyhPfd9u3b4+LFi7Xet4EDB3Lz/FeuXMHu3bsxYMCAWr8nZt++fXj88cfx0EMPwWw2Iy8vDw6HQ/TYEydOYP78+fjoo4+4pEKtVosOHTpwAX/fvn3ctMott9zCTX98++23MJvN3LkKCwvRvn37OrWZkIZGPXxCguD222/HwYMH0blzZyQkJKCsrAx33HEHAGDUqFF45ZVXsGPHDsjlcvTt25c33O5Op9Ph5ZdfxsyZM7n58uHDhwNw9kLbtWuHlStXevQyp0yZgrFjx0KpVCI+Ph7vvvsuYmJi0KZNG2RmZmLhwoWYNGkShg0bBgDo2bOn1zYAzh75ggUL8Oqrr8JqtSIpKQnvvfceFAoFzp07h5EjR8Jms+Hhhx/mPXwAQEpKCsrLy6HX63nz+EITJ07EnDlzMGzYMNjtdowfPx49evTAyZMna7/hAn//+98xZ84crFixAhqNBn369EFhYSFv2sJlwYIFsNlsePHFF7mHnjfeeAMLFizAnDlz8Omnn0Iul+P9998HwzCYNWsWsrKysH79evTs2ZM3onHgwAHcddddAbeXkMZAu+URQoLuiy++gEQiwZNPPtnYTQmp9PR0fPTRR7QsjzQJNKRPCAm69PR07Nu3r9EL74TSN998g7S0NAr2pMmgHj4hhBASAaiHTwghhEQACviEEEJIBKCATwghhEQACviEEEJIBKCATwghhEQACviEEEJIBPj/UQP35PComDAAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# reference: notebook 09\n",
    "# from matplotlib import pyplot as plt\n",
    "# plt.style.use(\"seaborn\")\n",
    "\n",
    "# %matplotlib inline\n",
    "x_match_walk = x_tune[['matchDuration','walkDistance']].values\n",
    "\n",
    "plt.scatter(x_match_walk[:, 1], x_match_walk[:, 0]+np.random.random(x_match_walk[:, 1].shape)/2, \n",
    "             s=20)\n",
    "plt.xlabel('walkDistance (normalized)'), plt.ylabel('matchDuration')\n",
    "plt.grid()\n",
    "plt.title('matchDuration vs walkDistance')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Boo yah! It looks like we have about 3 'clusters' when we look at `matchDuration` vs. `walkDistance`! <br/>\n",
    "\n",
    "What does it look like when use matchDuration with our new feature = `totalDistance`?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### K-means clusters: Match duration with total distance\n",
    "\n",
    "#### Plot: matchDuration + totalDistance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgQAAAFlCAYAAACUQvD0AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAABs3ElEQVR4nO3deXxTVfo/8M/N2jRJaVooFWmhLKUIskkFB8Uvbh1FxCouIGXYVBiBEQdEkUWgoAgyMzDA13HBrxUFhAF1UNGfC8woIMMmVMpm2ZfuJVuznt8fIbe5NzdL26Tr83695jUkubk5uUTOc895znM4xhgDIYQQQlo0WUM3gBBCCCENjwICQgghhFBAQAghhBAKCAghhBACCggIIYQQAgoICCGEEAIKCAipk19++QXz5s0LeszevXvx4IMPSr62atUqDBw4EMOHD8fw4cMxdOhQvPDCCzhz5kzU2nnkyBFMmzYtouevrXCuX02Oe+mll/Duu+8CAO666y5kZWVh+PDheOihhzBs2DCsXbsWTqcTAPDtt98iNzc36Pl++OEH/O1vfwvjmxDS9CkaugGENGWnTp3C1atX63SOBx54QNDZbdu2DX/4wx+wfft26HS6ujYRgLCdN998M1auXBmR89ZVuNevttd5+fLluPnmmwEAFosFM2bMwGuvvYa5c+fi7rvvxt133x30/UeOHEFlZWWNP5eQpogCAtJi7d27FytWrMANN9yAwsJCaDQaPPPMM8jLy0NhYSHuu+8+zJ49G263G0uWLMHhw4dhNpvBGENubi7atWuHlStXwmg04uWXX8Zrr72GzZs3Y926dZDJZDAYDFi6dCkAT2c0ffp0/Pbbb7DZbMjNzUX//v0l2/Xwww/js88+w+eff46RI0eiW7du2L17NxISEgCAf3zy5EksXrwYsbGxMJvN2LJlC954442Q7Xz44YexaNEi/Otf/4LRaMSCBQtQUFAAjuNwxx134IUXXoBCocDNN9+MZ555Bj/++COKioowceJEjBo1StDWjRs34vvvv8f//u//AgBOnz6NsWPH4ocffsDq1avxzTffQKlUwmAw4LXXXkNSUhL/3suXL/tdv40bNyIvLw8ymQytW7fG3LlzERMTIzhu8eLFkn8ft9xyS9C/79jYWMybNw/33HMPpk+fjq+//ho7duzAW2+9ha+//hpr164Fx3GQy+V48cUXoVKpsGHDBrhcLuj1ejz77LN49dVXcfbsWVRUVECr1WL58uXo1KkTcnJy0KdPHxw4cACXL1/GbbfdhkWLFkEmk+H777/HX//6V7jdbsTGxmLBggXIyMjAgQMHsHz5clitVshkMkyZMgVDhgyp9e+ZkDpjhLRQe/bsYd27d2f5+fmMMcYmTJjAnnjiCWaz2VhpaSnr0aMHu3LlCjtw4ACbOnUqc7lcjDHG3nrrLfbss88yxhjbsmULe+aZZxhjjB07dowNGDCAXbp0iTHG2Lp169jcuXP5zzl06BD//JgxYxhjjK1cuZItWLDAr22vv/46e/XVVxljjKWnp7PS0lL+Ne/jPXv2sIyMDHbhwgXGGAu7nXv27GFDhw5ljDH24osvskWLFjG3281sNhsbP348e+utt/jPycvLY4wxduTIEdazZ09WVVUlaKfRaGT9+/dnRUVFjDHG3njjDbZixQp26dIl1q9fP2az2RhjjL377rvsm2++8fuevu366aef2D333MN/1y1btrD777+fud1uwXHBvuesWbPYO++8wxhjbMiQIeyXX37x+8wBAwaww4cPC8559913s4MHDzLGGPv3v//NVq1a5ff38+WXX7JFixbx55k7dy5buHAhY4yx0aNHs2nTpjGXy8WMRiO7/fbb2e7du1lxcTG75ZZb+N/Yjh072IQJE1hFRQW777772Pnz5xljjF25coUNHjyYXbx40a+9hNQXGiEgLVr79u1x0003AQBSU1Oh1+uhUqmQkJAArVaLyspK9O3bF61atcKGDRtw/vx57N27F1qt1u9cu3fvxu23344bbrgBADB27FgAnpGIlJQU9O7dGwCQkZGBLVu2BG0Xx3GIiYkJ2f4bbrgBN954IwCE3U5fu3btwscffwyO46BSqfDkk0/i//7v//DMM88AAD+k3qNHD9jtdlgsFqjVav79Op0O9957Lz777DOMHTsWn3/+OdavX4+2bdsiIyMD2dnZGDx4MAYPHozbbrstaFv+/e9/44EHHuBHQh555BEsXrwYFy5cEBxXm+/pi+M4aDQawXNDhw7FlClTcOedd2LQoEF4+umn/d73+9//HikpKcjLy8PZs2fx888/o2/fvvzrQ4YMgUwmg06nQ4cOHVBZWYkDBw6ga9eu/G/svvvuw3333YedO3eiuLgYzz33nKBdx48fR7t2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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# reference: notebook 09\n",
    "# from matplotlib import pyplot as plt\n",
    "# plt.style.use(\"seaborn\")\n",
    "\n",
    "# %matplotlib inline\n",
    "x_match_total = x_tune[['matchDuration','totalDistance']].values\n",
    "\n",
    "plt.scatter(x_match_total[:, 1], x_match_total[:, 0]+np.random.random(x_match_total[:, 1].shape)/2, \n",
    "             s=20)\n",
    "plt.xlabel('totalDistance (normalized)'), plt.ylabel('matchDuration')\n",
    "plt.grid()\n",
    "plt.title('matchDuration vs totalDistance')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`totalDistance` also looks to have 3 clusters, not totally sure if it is 'better' than the `walkDistance`. At a glance, `walkDistance` appears to be a bit more tightly clustered. Let's try to run the model for both and see if we have any difference."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### K-means clusters: Match duration with walk distance\n",
    "\n",
    "#### Accuracy: matchDuration + walkDistance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average accuracy (with kmeans for matchDuration/walkDistance)=  91.77799999999999 +- 0.3699135034031609\n"
     ]
    }
   ],
   "source": [
    "# AVERAGE ACCURACY FOR MATCH-DURATION + WALK-DISTANCE\n",
    "# reference: notebook 09\n",
    "\n",
    "from sklearn.cluster import KMeans\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "x_match_walk = x_tune[['matchDuration','walkDistance']]\n",
    "\n",
    "cls_match_walk = KMeans(n_clusters=8, init='k-means++',random_state=17)\n",
    "cls_match_walk.fit(x_match_walk)\n",
    "newfeature_match_walk = cls_match_walk.labels_ # the labels from kmeans clustering\n",
    "\n",
    "cv_match_walk = StratifiedKFold(n_splits=10)\n",
    "\n",
    "x_match_walk = x_tune.loc[:, cols_df]\n",
    "y_match_walk = y_tune.loc[:, ('quart_binary')]\n",
    "x_match_walk = np.column_stack((x_match_walk,pd.get_dummies(newfeature_match_walk)))\n",
    "\n",
    "acc_match_walk = cross_val_score(clf,x_match_walk,y=y_match_walk,cv=cv_match_walk)\n",
    "\n",
    "print (\"Average accuracy (with kmeans for matchDuration/walkDistance)= \", acc_match_walk.mean()*100, \"+-\", acc_match_walk.std()*100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Well, it does not appear that the cluster dummy variables for (`matchDuration` + `walkDistance`) is much help either. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### K-means clusters: Match duration with total distance\n",
    "\n",
    "#### Accuracy: matchDuration + totalDistance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average accuracy (with kmeans for matchDuration/totalDistance)=  91.83800000000002 +- 0.28403520908506996\n"
     ]
    }
   ],
   "source": [
    "# MATCH DURATION + TOTAL DISTANCE \n",
    "# from sklearn.cluster import KMeans\n",
    "# import numpy as np\n",
    "# import pandas as pd\n",
    "\n",
    "x_match_total = x_tune[['matchDuration','totalDistance']]\n",
    "\n",
    "cls_match_total = KMeans(n_clusters=8, init='k-means++',random_state=17)\n",
    "cls_match_total.fit(x_match_total)\n",
    "newfeature_match_total = cls_match_total.labels_ # the labels from kmeans clustering\n",
    "\n",
    "cv_match_total = StratifiedKFold(n_splits=10)\n",
    "\n",
    "x_match_total = x_tune.loc[:, cols_df]\n",
    "y_match_total = y_tune.loc[:, ('quart_binary')]\n",
    "x_match_total = np.column_stack((x_match_total,pd.get_dummies(newfeature_match_total)))\n",
    "\n",
    "acc_match_total = cross_val_score(clf,x_match_total,y=y_match_total,cv=cv_match_total)\n",
    "\n",
    "print (\"Average accuracy (with kmeans for matchDuration/totalDistance)= \", acc_match_total.mean()*100, \"+-\", acc_match_total.std()*100)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Slight improvement with this model.**\n",
    "\n",
    "Looks like this version is still very slightly worse than the baseline model; the accuracy is just under, and the standard deviation has gone up ever-so-slightly. \n",
    "\n",
    "For reference: baseline model has accuracy 91.916 +- 0.27782. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### K-means clusters: Total items with total distance\n",
    "\n",
    "#### Accuracy: totalItems + totalDistance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average accuracy (with kmeans for totalItems/totalDistance)=  91.856 +- 0.33380233672040255\n"
     ]
    }
   ],
   "source": [
    "# TOTAL ITEMS + TOTAL DISTANCE \n",
    "# from sklearn.cluster import KMeans\n",
    "# import numpy as np\n",
    "# import pandas as pd\n",
    "\n",
    "x_items_distance = x_tune[['totalItems','totalDistance']]\n",
    "\n",
    "cls_items_distance = KMeans(n_clusters=8, init='k-means++',random_state=17)\n",
    "cls_items_distance.fit(x_items_distance)\n",
    "newfeature_items_distance = cls_items_distance.labels_ # the labels from kmeans clustering\n",
    "\n",
    "cv_items_distance = StratifiedKFold(n_splits=10)\n",
    "\n",
    "x_items_distance = x_tune.loc[:, cols_df]\n",
    "y_items_distance = y_tune.loc[:, ('quart_binary')]\n",
    "x_items_distance = np.column_stack((x_items_distance,pd.get_dummies(newfeature_items_distance)))\n",
    "\n",
    "acc_items_distance = cross_val_score(clf,x_items_distance,y=y_items_distance,cv=cv_items_distance)\n",
    "\n",
    "print (\"Average accuracy (with kmeans for totalItems/totalDistance)= \", acc_items_distance.mean()*100, \"+-\", acc_items_distance.std()*100)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Looks like this version is not as good as the baseline model when including dummy variables with clusters on `totalItems` and `totalDistance`.\n",
    "\n",
    "For reference: baseline model has accuracy 91.916 +- 0.27782. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### K-means clusters: Heal items with kills assist\n",
    "\n",
    "#### Accuracy: healItems + killsAssist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average accuracy (with kmeans for healItems/killsAssist)=  91.816 +- 0.2637877935007584\n"
     ]
    }
   ],
   "source": [
    "# HEAL ITEMS + KILLS ASSISST  \n",
    "# from sklearn.cluster import KMeans\n",
    "# import numpy as np\n",
    "# import pandas as pd\n",
    "\n",
    "x_heal_kill = x_tune[['healItems','killsAssist']]\n",
    "\n",
    "cls_heal_kill = KMeans(n_clusters=8, init='k-means++',random_state=17)\n",
    "cls_heal_kill.fit(x_heal_kill)\n",
    "newfeature_heal_kill = cls_heal_kill.labels_ # the labels from kmeans clustering\n",
    "\n",
    "cv_heal_kill = StratifiedKFold(n_splits=10)\n",
    "\n",
    "x_heal_kill = x_tune.loc[:, cols_df]\n",
    "y_heal_kill = y_tune.loc[:, ('quart_binary')]\n",
    "x_heal_kill = np.column_stack((x_heal_kill,pd.get_dummies(newfeature_heal_kill)))\n",
    "\n",
    "acc_heal_kill = cross_val_score(clf,x_heal_kill,y=y_heal_kill,cv=cv_heal_kill)\n",
    "\n",
    "print (\"Average accuracy (with kmeans for healItems/killsAssist)= \", acc_heal_kill.mean()*100, \"+-\", acc_heal_kill.std()*100)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Baseline average accuracy =  91.916 +- 0.27782."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The main take-away is that we don't see any particular benefits from clustering for our prediction analysis. We don't particularly improve Accuracy or Standard Deviation.\n",
    "\n",
    "This has been a partially informed analysis from what we believe would make sense to cluster on after looking at the attributes/features in the data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Scale data to prep for PCA & Hierarchical"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Because we haven't been able to improve the prediction capabilities on raw data, we'll attempt PCA in order to make our clustering as well-informed as possible.\n",
    "\n",
    "First, create copy of DF and scale the data. <br/>\n",
    "We will scale the data to have a mean of 0 and standard deviation of 1. <br/>\n",
    "We want to scale the data before doing PCA. Scaling the data tells PCA that all the attributes are of equal importance. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scale data\n",
    "\n",
    "# define scaler\n",
    "scaler = StandardScaler()\n",
    "\n",
    "# create copy of dataframe to scale for PCA\n",
    "scaled_x_tune = x_tune.copy()\n",
    "\n",
    "# create scaled version of data frame \n",
    "scaled_x_tune = pd.DataFrame(scaler.fit_transform(scaled_x_tune), columns = scaled_x_tune.columns)\n",
    "\n",
    "# reference: https://www.statology.org/scree-plot-python/\n",
    "\n",
    "# confirm we have removed matchType (can delete this later)\n",
    "# cols_df_tune = scaled_x_tune.columns.values.tolist()\n",
    "# cols_df_tune"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "import scipy.cluster.hierarchy as sch\n",
    "from sklearn.cluster import AgglomerativeClustering\n",
    "import sklearn.metrics as sm\n",
    "\n",
    "random.seed(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Create new DF for our clustering efforts = sample 2,000 records from x_tune."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_tune_df = pd.DataFrame(y_tune, columns=y_tune.columns)\n",
    "y_tune_df = y_tune_df.reset_index(drop=False)\n",
    "\n",
    "indices_slim = scaled_x_tune.sample(n=2000, replace=False, random_state=17).index\n",
    "x_tune_slim = scaled_x_tune.loc[indices_slim, :]\n",
    "y_tune_slim = y_tune_df.loc[indices_slim, :]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Hierarchical clustering is an algorithm that groups similar objects into groups called clusters. Simply, it forms as many clusters as there are data points then take two nearest data points and make them a cluster. Then take two nearest clusters and make them a cluster and repeat this step until there is only one finaly cluster.\n",
    "\n",
    "Dendrogram shows the hierarchical relationship between the clusters as we show for our x_tune_slim."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfUAAAFLCAYAAADPiBUUAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAA9RUlEQVR4nO3de5QU5Z038G919/RleqZnYAAFEQ8gyIIhKncvRJMouFnyRs2aiGG9JLpmPRpIdoUoYDZi8LIviXE3AX0TohBlNWxWMDmrAioiSERlWAdBDmLEGZGZYW490/eu9w94iuqa6q6qvk139fdzjsdmurrq11XV9Xtu9ZQky7IMIiIiKnuOgQ6AiIiI8oNJnYiIyCaY1ImIiGyCSZ2IiMgmmNSJiIhsgkmdiIjIJlwDHUA2eBceERFVEkmSTC1Xlknd7JcjIiKqJGx+JyIisgkmdSIiIptgUiciIrIJJnUiIiKbYFInIiKyCSZ1IiIim2BSJyIisgkmdSIiIptgUiciIrKJgib1xsZGLFiwAADwwQcfYP78+ViwYAG++93voq2tDQDw3HPP4dprr8X111+PV199tZDhEBER2VrBpol98sknsWnTJvh8PgDAgw8+iGXLluFv/uZvsGHDBjz55JP43ve+h3Xr1mHjxo2IRCKYP38+LrnkErjd7kKFRUREZFsFS+qjRo3C448/jnvuuQcAsGrVKgwbNgwAkEgk4PF4sG/fPlx44YVwu91wu90YNWoUDhw4gMmTJxcqLNv57eYmvNnYPNBhEFEFu+SLZ+HWeZMGOgxCAZvf58yZA5frdJlBJPR3330X69evx80334xgMIja2lplGb/fj2AwWKiQbOnNxma0dYUHOgwiqlBtXWFWLEpIUZ/S9uc//xm//vWv8cQTT2Dw4MGoqalBb2+v8n5vb29KkidzhtR58ZulVw10GERUgb674uWBDoFUijb6/YUXXsD69euxbt06nH322QCAyZMn45133kEkEkFPTw8OHz6M8ePHFyskIiIiWylKTT2RSODBBx/E8OHDcddddwEApk2bhrvvvhsLFizA/PnzIcsyFi1aBI/HU4yQiIiIbKegSX3kyJF47rnnAAB/+ctfdJe5/vrrcf311xcyDCIioorAyWeIiIhsgkmdiIjIJpjUiYiIbIJJnYiIyCaY1ImIiGyCSZ2IiMgmmNSJiIhsgkmdiIjIJpjUiYiIbIJJnYiIyCaY1ImIiGyCSZ2IiMgmmNSJiIhsgkmdiIjIJpjUiYiIbIJJnYiIyCaY1ImIiGyCSZ2IiMgmmNSJiIhsgkmdiIjIJpjUiYiIbIJJnYiIyCaY1ImIiGyCSZ2IiMgmmNSJiIhsgkmdiIjIJpjUiYiIbIJJnYiIyCaY1ImIiGyCSZ2IiMgmmNSJiIhsgkmdiIjIJpjUiYiIbIJJnYiIyCaY1ImIiGyCSZ2IiMgmmNSJiIhsgkmdiIjIJpjUiYiIbIJJnYiIyCYKmtQbGxuxYMECAMBf//pX3HDDDZg/fz7uv/9+JJNJAMBzzz2Ha6+9Ftdffz1effXVQoZDRERkawVL6k8++SSWLl2KSCQCAFi5ciUWLlyIZ555BrIsY+vWrWhtbcW6deuwYcMG/OY3v8GqVasQjUYLFRIREZGtFSypjxo1Co8//rjy76amJkyfPh0AMHv2bOzcuRP79u3DhRdeCLfbjdraWowaNQoHDhwoVEhERES2VrCkPmfOHLhcLuXfsixDkiQAgN/vR09PD4LBIGpra5Vl/H4/gsFgoUIiIiKytaINlHM4Tm+qt7cXgUAANTU16O3tTfm7OskTERGReUVL6hMnTsTu3bsBANu3b8fUqVMxefJkvPPOO4hEIujp6cHhw4cxfvz4YoVERERkKy7jRfJj8eLFWLZsGVatWoUxY8Zgzpw5cDqdWLBgAebPnw9ZlrFo0SJ4PJ5ihURERGQrkizL8kAHQdn77oqXAQC/WXrVAEdCRJWI16DSwslniIiIbIJJnYiIyCaY1ImIiGyCSZ2IiMgmmNSJiIhsgkmdiIjIJpjUiYiIbIJJnYiIyCaY1ImIiGyCSZ2IiMgmmNSJiIhsomgPdCEisqvfbm7Cm43NAx3GgGjrCgM4PQd8Jbrki2fh1nmTBjoMAKypExHl7M3GZiW5VZohdV4MqfMOdBgDpq0rXFIFOtbUiYjyYEidl08qq0Cl1kLBmjoREZFNMKkTERHZBJM6ERGRTTCpExER2QSTOhERkU0wqRMREdkEkzoREZFNMKkTERHZBJM6ERGRTTCpExER2QSTOhERkU0wqRMREdkEkzoREZFNMKkTERHZBJM6ERGRTTCpExER2QSTOhERkU0wqRMREdkEkzoREZFNMKkTERHZhKWkHgwGcejQoULFQkRERDkwTOrPP/88lixZghMnTuBv//Zvcffdd2P16tXFiI2IiIgsMEzqzz77LH74wx/ixRdfxFe+8hVs3rwZL7/8cjFiIyIiIgtMNb8PGzYMr7/+Oi6//HK4XC5EIpFCx0VEREQWGSb1c889F//4j/+ITz/9FLNmzcLChQsxefLkYsRGREREFriMFvjZz36G9957D+PHj4fb7cbXv/51zJ49O6uNxWIxLFmyBM3NzXA4HHjggQfgcrmwZMkSSJKEcePG4f7774fDwUH5REREVhkm9WQyiT179uAPf/gDli1bhv379+PSSy/NamOvv/464vE4NmzYgDfffBO/+MUvEIvFsHDhQsyYMQPLly/H1q1bceWVV2a1fiIioly8snk/9je2mF6+uysMAHhsxRZTy0/84ghcOW9iVrGZYVgl/ulPf4pQKISmpiY4nU588sknuPfee7Pa2OjRo5FIJJBMJhEMBuFyudDU1ITp06cDAGbPno2dO3dmtW4iIqJc7W9sURK1GZfWVePSumpTy3Z3hS0VGLJhWFNvamrCH//4R2zfvh0+nw8PP/ww5s2bl9XGqqur0dzcjKuvvhodHR1YvXo13n77bUiSBADw+/3o6enJat1ERET5EKjz4gdLv5r39ZqtzefCMKlLkoRoNKok3o6ODuW1Vb/73e9w6aWX4kc/+hE+++wz3HTTTYjFYsr7vb29CAQCWa2biIio0hk2v//DP/wDbrnlFrS2tuLBBx/Etddei5tuuimrjQUCAdTW1gIA6urqEI/HMXHiROzevRsAsH37dkydOjWrdRMREVU6w5r6N77xDZx//vnYvXs3EokE1qxZg/POOy+rjd1888249957MX/+fMRiMSxatAjnn38+li1bhlWrVmHMmDGYM2dOVusmIiKqdIZJ/eDBg1i9ejV+/vOf4/Dhw1i+fDkeeOABjBkzxvLG/H4/HnvssX5/X79+veV1ERERUSrD5vdly5bhmmuuAQCMHTsW//RP/4T77ruv4IERERGRNYZJPRQKpUw2c8kllyAUChU0KCIiIrLOMKkPHjwYzz77LHp7e9Hb24vnn38eDQ0NxYiNiIiILDDsU1+5ciX+9V//FY888giqqqowbdo0PPjgg8WIjYiIqKRZmYHOyuxz2c48Z5jUR4wYgTVr1lheMRERkd2JGegCdV7DZc0sA5yeea4gSf2NN97AL37xC3R1dUGWZeXvW7dutbwxIiIiu8n3DHS5zDxnmNRXrFiBJUuWYNy4cVnPJEdERESFZ5jUBw0ahCuuuKIYsRAREVEODJP6lClTsHLlSlx22WXweDzK36dNm1bQwIiIiMgaw6S+b98+AMD+/fuVv0mShKeffrpwUREREZFlhkl93bp1xYiDiGzgyNqn0L5z10CHUXSRmi8BAPbcdscAR1JcDRfPwuhbsnvAFxWGYVLfu3cv1qxZg76+PsiyjGQyiZaWFmzbtq0Y8RFRGWnfuQuRtnZ4hlTWBFULg68PdAhFF2lrR/vOXUzqJcYwqd9777347ne/iz/+8Y9YsGABXn75ZUycaP3eOSKqDJ4hDZj65OqBDoMKrNJaJcqFYVJ3u9247rrr0NzcjEAggEceeQTz5s0rRmxERERkgeHc7x6PB52dnRg9ejQaGxvhdDqRSCSKERsRERFZYJjUb775ZixatAhXXHEFXnjhBXzta1/D+eefX4zYiIiIyALD5veLL74Yc+fOhSRJ2LhxIz7++GPU1tYWIzYiIiKyIG1N/bPPPkNLSwtuvPFGHDt2DC0tLejs7ERtbS1uu+22YsZIREREJqStqf/yl7/E7t27cfz4cdx4442nP+By4fLLLy9GbERERGRB2qS+cuVKAMATTzyB22+/vWgBERERUXYMB8rNmTMHmzZtgizLWL58Oa677jq8//77xYiNiIiILDBM6vfeey+SySS2bt2KI0eO4Mc//jFWrFhRjNiIiIjIAsOkHolE8I1vfAOvvvoq5s2bh6lTpyIajRYjNiIiIrLAMKk7nU689NJLeO2113D55Zdjy5YtcDgMP0ZERERFZpidf/rTn+K1117D8uXLMWzYMPzpT39i8zsREVEJSjv6vbW1FUOHDkVtbS3uuusuAEBLSwv+5V/+pWjBERERkXlpk/rSpUuxZs0afOc734EkSZBlWXlPkiRs3bq1KAESERGROWmT+po1awCAz00nIiIqExnnfj98+DD+8Ic/4KOPPoLH48G5556Lv//7v8fw4cOLFR8RERGZlHag3K5duzB//nyEw2F86UtfwqxZs3DixAlcd911+Mtf/lLMGImIiMiEjHO//+Y3v+n3mNVrr70WDz30EJ555pmCB0dERETmpa2pB4NB3eemT548GaFQqKBBERERkXVpk7rLZfiodSIiIiohaTN3b28v9uzZk3Irm9DX11fQoIiIiMi6tEn9jDPOwGOPPab73rBhwwoWEBEREWUnbVJft25dMeMgIiKiHPHJLERERDbBpE5ERGQTTOpEREQ2YXjfWnNzM9avX4+urq6UkfArV64saGBERERkjWFSX7hwIaZOnYqpU6dCkqScN7hmzRps27YNsVgMN9xwA6ZPn44lS5ZAkiSMGzcO999/PxwONiAQERFZZZjU4/E4Fi9enJeN7d69G++99x6effZZhEIh/Pa3v8XKlSuxcOFCzJgxA8uXL8fWrVtx5ZVX5mV7RERElcSwSjxlyhRs27YN0Wg0543t2LED48ePx5133ok77rgDl19+OZqamjB9+nQAwOzZs7Fz586ct0NERFSJDGvq//M//4P169en/E2SJHzwwQeWN9bR0YGWlhasXr0an376Kb7//e9DlmWlWd/v96Onp8fyeomIiMhEUt+xY0feNlZfX48xY8bA7XZjzJgx8Hg8OHbsmPJ+b28vAoFA3rZHRESnHVn7FNp37srLuiJt7QCAPbfdkZf1AUDDxbMw+pab8ra+SmSY1E+cOIFNmzaht7cXsiwjmUzi008/xSOPPGJ5Y1OmTMHTTz+NW265BcePH0coFMKsWbOwe/duzJgxA9u3b8fMmTOz+iJEZF0+L/IAL/Slrn3nLkTa2uEZ0pDzuvKxDrVIWzvad+7isc6RqdHvw4cPx969e/HVr34Vr732Gr7whS9ktbErrrgCb7/9Nr75zW9ClmUsX74cI0eOxLJly7Bq1SqMGTMGc+bMyWrdRGRdPi/yAC/05cAzpAFTn1w90GH0k8+CYCUzTOrHjx/H008/jYcffhhXXXUVvve97+Gmm7L/gd1zzz39/qbtsyei4inVizzACz2RVYaj3+vq6gAAo0ePxoEDBzBo0KCCB0VERETWGdbUZ86cibvvvhuLFy/GrbfeiqamJni93mLERkRERBYYJvVFixbhk08+wVlnnYVVq1bh7bffxp133lmM2IiIiMgCU/OxNjY24uc//znGjBmD+vp6nHHGGYWOi4iIiCwyTOr/9m//htdffx0vv/wyEokENm7ciIceeqgYsREREVWccF8M3V1hvLJ5v+XPGib1HTt24NFHH4XH40FNTQ3Wrl2L7du3ZxUoERERZeatroKclLG/scXyZw2TunhimpjKNRqN8ilqREREJchwoNzcuXOxcOFCdHV14Xe/+x02bdqEv/u7vytGbERERGSBYVK//fbb8cYbb2DEiBH47LPPcNddd+GKK64oRmxERERkQdqk/vbbbyuvvV4vvvzlL6e8N23atMJGRkRERJakTeq//OUvAQCdnZ04evQoLrzwQjgcDrz33nsYP348NmzYULQgiYiIyFjapL5u3ToAwG233YZ///d/xznnnAMAaG5uxvLly4sTHREREZlmOIy9paVFSegAMGLECLS0WB9mT0RERIVlOFBu0qRJWLx4Ma6++mrIsozNmzdj6tSpxYiNiIiILDBM6itWrMD69euVPvSLL74Y8+fPL3hgREREZE3apN7a2oqhQ4eira0Nc+fOxdy5c5X3jh8/jhEjRhQlQCIiIjInbVJfunQp1qxZg+985zuQJAmyLKf8f+vWrcWMk4iIiAykTepr1qwBAGzbtq1owRAREVH20ib1H//4xxk/uHLlyrwHQ0RENJC6OkIDHUJO0ib16dOnFzMOIiIiylHapH7NNdcAAILBIF544QXceOON+Pzzz7FhwwbcfvvtRQuQiIiIzDGcfOaf//mfcfz4cQCA3+9HMpnEPffcU/DAiIiIyBpTM8otWrQIAFBTU4NFixbhk08+KXhgREREZI1hUpckCQcPHlT+ffjwYbhchnPWEBERUZEZZufFixfj1ltvxRlnnAEA6OjowCOPPFLwwIiIiMgaw6R+8cUX49VXX8WHH34Il8uFMWPGwO12FyM2IiIissAwqae7X533qRMREZUWw6Suvl89Ho9j69atGDNmTEGDIiIiIusMk7q4X1345je/iRtuuKFgARERUXk5svYptO/cldM6Im3tAIA9t92RczwNF8/C6Ftuynk95chw9LvW4cOHlfvWiYiI2nfuUpJytjxDGuAZ0pBzLJG29pwLGOXMsKY+YcIESJIEAJBlGYMHD8YPf/jDggdGRETlwzOkAVOfXD3QYeSlpl/ODJP6gQMHihEHERER5Sht8/szzzyjvD506FDKew8++GDhIiIiIqKspE3qzz//vPJaO9f7nj17ChcRERERZSVt87ssy7qv7Wbd3o146+i7Ax1G1tr7vggAuHPzfQMcSXZmnn0RFlxw3UCHQURkC6ZGv4uBcnb01tF30R7qHOgwsnbWrEacNatxoMPISnuos6wLVEREpSZtTd3OiVyrwVeP/5jHcQLFVq6t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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visulaizations\n",
    "dendrogram = sch.dendrogram(sch.linkage(x_tune_slim, method = 'ward'))\n",
    "plt.ylabel('Euclidean Distances')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Although, we could potentially define the best clusters using the dendrogram, it is a better practice to find the accuracy of mean and standard deviation to determine which is a better distance meterics, linkage, and number of clusters for the model.\n",
    "\n",
    "There are other types of distance metrics could be used but we decided to use Euclidean and Cosine distance and for linkage criteria, we looked at ward, complete, and average. We had to consider that if linkage is ward, only Euclidean is accpected.\n",
    "\n",
    "The best model would be with a high accuracy and a low standard deviation. However, it is difficult identify which model is the best so we calculated a difference for each algorithm to determine the best possible model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's reassess and do some PCA before we try clusters again."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Modeling and Evaluation 1 (cont.)**\n",
    "\n",
    "*Assignment: Train and adjust parameters.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PCA FOR CLUSTERS\n",
    "\n",
    "Jump to [Top](#TOP)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Why do we do PCA before clustering?*\n",
    "\n",
    "In short, we perform PCA for **dimensionality reduction**.\n",
    "\n",
    "Clustering is highly sensitive to dimensionality, it is also hard to visualize outside of 3-Dimensions. We will use PCA to reduce the dimensionality of the data before we try clustering again. \n",
    "\n",
    "*reference: https://www.kaggle.com/code/robroseknows/pubg-clustering-exploration/notebook* \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is necessary to **normalize** data before performing PCA. The PCA calculates a new projection of your data set. And the new axis are based on the standard deviation of your variables. <br/>\n",
    "\n",
    "*reference: https://www.researchgate.net/post/Is-it-necessary-to-normalize-data-before-performing-principle-component-analysis#:~:text=Yes%2C%20it%20is%20necessary%20to,standard%20deviation%20of%20your%20variables*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Perform PCA on scaled_x_tune"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import library\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# define PCA model to use (start with 14 = random number)\n",
    "pca14 = PCA(n_components = 14)\n",
    "\n",
    "# fit PCA model to data (scaled_x_tune)\n",
    "pca14_fit = pca14.fit(scaled_x_tune)\n",
    "P14 = pca14.transform(scaled_x_tune)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.910351412370801\n"
     ]
    }
   ],
   "source": [
    "# how much of the variance is explained by 14 components\n",
    "print(sum(pca14.explained_variance_ratio_))\n",
    "\n",
    "# reference: https://www.kaggle.com/code/robroseknows/pubg-clustering-exploration/notebook"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This tells us that about 91% of the variance in x_tune can be explained using 14 components. <br/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's look at the breakdown of % variance explained by each principal component."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[34.350872  11.092523   5.4613895  4.959305   4.2574177  4.1059637\n",
      "  4.072749   3.9200096  3.6937654  3.5024033  3.2831855  3.1688287\n",
      "  2.8406036  2.3261254]\n"
     ]
    }
   ],
   "source": [
    "print(pca14.explained_variance_ratio_ * 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This tells us that: <br/>\n",
    "* The first principal component explains 34.3% of the total variance in the scaled_x_tune dataset. <br/>\n",
    "* The second principal component explains an additional 11.1% of the total variance in the scaled_x_tune dataset. <br/>\n",
    "* The third principal component explains an additional 5.5% of the total variance in the scaled_x_tune dataset. <br/>\n",
    "* The fourth principal component explains an additional 4.9% of the total variance in the scaled_x_tune dataset. <br/>\n",
    "* And so on."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### PCA Scree plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfUAAAFlCAYAAADyLnFSAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAA9NklEQVR4nO3deXhU9d3+8feZGbKRQFgCRpAYCEGpC6KPooIUELQuCAUNomgFqW31pwiixAUpKqAoBakE9GldaH3AhkVww6IiihuisUYTQAwgiyEgW/Zkzvn9Mc1AyDIBMjOZOffrunJl5kxO5vMhIff5fs9mWJZlISIiIiHPEewCREREpHEo1EVERMKEQl1ERCRMKNRFRETChEJdREQkTCjURUREwoQr2AWISOPJysrimWee4cCBA1iWxSmnnMIDDzxA165dA1rH559/ztixY0lOTsYwDCzLwul0ctddd9G/f3/mzp3L/v37mTx5cr3fZ/To0Tz99NO0bt06QJWLhDaFukiYKC8v54477uDvf/87v/rVrwB4/fXXGTt2LO+99x5OpzOg9XTq1InXX3/d+zw3N5cbb7yR9957r8HfY926df4oTSRsKdRFwkRJSQmHDx+muLjYu2zw4MHExsbidrtxOp1kZmby4osv4nA4aNWqFU8++STbt2/niSeeICYmhqKiIpYsWcLHH39MRkYGFRUVREVF8cADD3DeeecBkJGRwbvvvotpmnTo0IFHH32U9u3b+6zvjDPOICoqip07d1ZbvnnzZqZOncqBAwcwDIPRo0czZMgQ0tPTAbj11lt5/vnnSUxMbMR/LZEwZYlI2Pj73/9unXPOOVb//v2t++67z/rXv/5lFRcXW5ZlWTk5OdZFF11k7dq1y7Isy3rxxRetRx55xPrss8+sM844w9qxY4dlWZaVl5dnXXPNNdYvv/xiWZZlbdq0ybr00kutoqIia9myZda4ceOsiooKy7Isa9GiRdbtt99eo47PPvvMuvrqq6stW7VqlXXJJZdYxcXF1rPPPmv9+c9/tioqKqwBAwZYq1atsizLsn7++WerT58+1ldffWVZlmWlpqZa+/bt88O/lEh40khdJIzcdtttXH/99axfv57169fzwgsv8MILL5CZmcmnn35K7969vSPe3/3ud4Bn/3diYiIdOnQAPFPee/bs8b4OYBgG27dv54MPPuDbb79l2LBhAJimSUlJSa21bN++neuuuw6AyspKTjnlFObNm0d0dLT3a7Zu3UpZWRmDBg0CoH379gwaNIiPPvrIOzMgIg2nUBcJExs2bODrr7/m9ttvp1+/fvTr14/x48dzzTXXsG7dOpxOJ4ZheL++tLTUOxUeExPjXW6aJhdffDGzZ8/2Ltu9ezft2rXDNE1uv/12Ro4cCXj24x88eLDWeo7dp14bt9tdrSYAy7KorKw8rt5FxEOntImEidatW5ORkcGXX37pXVZQUEBhYSGpqalcdNFFfPrpp+zZsweARYsWMXPmzBrf5+KLL2bdunVs2bIFgA8//JDBgwdTWlpK7969yczMpLCwEIA5c+Zw//33n3DNnTt3xuVy8e677wKQn5/PqlWruOSSSwBwOp0KeJHjoJG6SJhITk7mueee4y9/+Qs///wzkZGRxMXFMW3aNDp37gzAxIkTuf322wFISEhg2rRpbN26tdr3SUlJYerUqYwfPx7LsnC5XGRkZNC8eXOuv/568vPzueGGGzAMg8TERGbMmHHCNTdr1ox58+bx+OOPM3fuXNxuN3feeSe9evUC4Morr2TUqFHMnTuX1NTUE34fEbswLEu3XhUREQkHmn4XEREJEwp1ERGRMKFQFxERCRMKdRERkTChUBcREQkTIX9KW0HB4WCX0GhatYph//5i318Yhuzau137Bvv2bte+wb69N3bfCQlxdb6mkXoT4nIF9i5aTYlde7dr32Df3u3aN9i390D2rVAXEREJEwp1ERGRMKFQFxERCRN+O1DONE2mTJnCxo0biYiI4PHHHycpKcn7+qpVq3j++ecxDIO0tDSuv/56AIYMGUJcnOcggI4dOzJ9+nR/lSgiIhJW/Bbqq1evpry8nMWLF5OVlcWMGTPIyMgAPLdbfOaZZ1iyZAkxMTFcddVVDBgwgObNmwOwcOFCf5UlIiIStvw2/b5hwwb69OkDQI8ePcjOzva+5nQ6eeutt4iLi+PAgQMANG/enNzcXEpKShg9ejS33HILWVlZ/ipPREQk7PhtpF5YWEhsbKz3edV9kV0uz1tW3UN56tSp9O3bF5fLRVRUFGPGjOH6669n69atjB07lnfeece7Tm1atYoJq9Mk6jv/MNzZtXe79g327d2ufYN9ew9U334L9djYWIqKirzPTdOsEc6DBg3i8ssvZ9KkSSxfvpxrr72WpKQkDMMgOTmZ+Ph4CgoKSExMrPN9wulCBgkJcWF1MZ3jYdfe7do32Ld3u/YN9u29sfsOysVnevbsydq1awHIysoiNTXV+1phYSE333wz5eXlOBwOoqOjcTgcZGZmMmPGDADy8/MpLCwkISHBXyVWs2yZi759Y0hMjKVv3xiWLQv5i+2JiIjN+C25Bg4cyLp16xgxYgSWZTFt2jRWrlxJcXExaWlpXHvttdx00024XC66devG4MGDcbvdpKenc+ONN2IYBtOmTat36r2xLFvm4o47or3Pc3Kc/31ewtChlX5/fxERkcZgWJZlBbuIk9EYUxp9+8aQk1Nzv3z37m7WrAnc9L5dp6bAvr3btW+wb+927Rvs23tYTL+Hkk2bav9nqGu5iIhIU6TUAlJTzeNaLiIi0hQp1IFx48prXX7PPbUvFxERaYoU6sDQoZUsWFBCYqJnZN6ypcWCBTpITkREQovO2/qvoUMrad/eYsiQGLp0MRXoIiIScjRSP0rXrp6R+ubNDkL7nAAREbEjhfpR2ra1iI+3OHzYYM8eI9jliIiIHBeF+lEM48hoXaeziYhIqFFyHaNrVzfgmYIXEREJJUquYxy9X11ERCSUKLmOoVAXEZFQpeQ6hkJdRERClZLrGJ06WURGWuze7aCwMNjViIiINJxC/RhOJ3TurNG6iIiEHqVWLTQFLyIioUipVQuFuoiIhCKlVi0U6iIiEoqUWrVQqIuISChSatWiSxcTw7DIy3NQURHsakRERBpGoV6LmBg47TSLykqDrVv1TyQiIqFBiVWHlBRNwYuISGhRYtVB+9VFRCTUKLHqoFAXEZFQo8SqQ2qqQl1EREKLEqsOR+9Tt6wgFyMiItIACvU6tG1r0bq1SWGhwc8/G8EuR0RExCeFej10BLyIiIQSpVU9dLCciIiEEqVVPRTqIiISSpRW9VCoi4hIKFFa1UOhLiIioURpVY/TTrOIjLT4+WcHhw8HuxoREZH6KdTr4XR67tgGGq2LiEjTp6TyQVPwIiISKvyWVKZpMnnyZNLS0hg1ahTbtm2r9vqqVasYNmwYw4cP51//+leD1gkGhbqIiIQKl7++8erVqykvL2fx4sVkZWUxY8YMMjIyAHC73TzzzDMsWbKEmJgYrrrqKgYMGMCXX35Z5zrBolAXEZFQ4bdQ37BhA3369AGgR48eZGdne19zOp289dZbuFwu9u3bB0Dz5s3rXSdYFOoiIhIq/BbqhYWFxMbGep87nU4qKytxuTxv6XK5ePfdd5k6dSp9+/bF5XL5XKc2rVrF4HI5/dUGvXqBYcDWrU7i4+No1sxvbwVAQkKcf9+gCbNr73btG+zbu137Bvv2Hqi+/RbqsbGxFBUVeZ+bplkjnAcNGsTll1/OpEmTWL58eYPWOdb+/cWNW3gtTjutOdu3O/jiiyLvLVn9ISEhjoICe547Z9fe7do32Ld3u/YN9u29sfuubwPBb3PKPXv2ZO3atQBkZWWRmprqfa2wsJCbb76Z8vJyHA4H0dHROByOetcJJk3Bi4hIKPDbSH3gwIGsW7eOESNGYFkW06ZNY+XKlRQXF5OWlsa1117LTTfdhMvlolu3bgwePBjDMGqs0xR07Wry3nsKdRERadr8FuoOh4OpU6dWW9alSxfv47S0NNLS0mqsd+w6TYFG6iIiEgqUUg2gUBcRkVCglGqAo0PdsoJcjIiISB0U6g3Qpo1FmzYmRUUGu3cbwS5HRESkVgr1BkpJ0RS8iIg0bUqoBqo6P12hLiIiTZUSqoE0UhcRkaZOCdVAGqmLiEhTp4RqII3URUSkqVNCNdBpp1lERVnk5zs4dCjY1YiIiNSkUG8ghwO6dNFoXUREmi6l03HQfnUREWnKlE7HQfvVRUSkKVM6HQddA15ERJoypdNxOBLqziBXIiIiUpNC/Th07mxiGBZbtxqUlwe7GhERkeoU6schOho6dbJwuw3y8vRPJyIiTYuS6Thpv7qIiDRVSqbjpFAXEZGmSsl0nBTqIiLSVCmZjpNCXUREmiol03Hq2tUNeELdsoJcjIiIyFEU6sepdWto29akuNhg1y4j2OWIiIh4KdRPgC4XKyIiTZFS6QRov7qIiDRFSqUToFAXEZGmSKl0AnQLVhERaYqUSidA+9RFRKQpUiqdgI4dLaKjLfbscXDwYLCrERER8VConwCHA7p00WhdRESaFiXSCdJ+dRERaWqUSCdI+9VFRKSpUSKdoCMjdWeQKxEREfFQqJ8gjdRFRKSpUSKdoM6dTRwOi61bDcrKgl2NiIiIQv2ERUVBp04WpmmQl6d/RhERCT6Xv76xaZpMmTKFjRs3EhERweOPP05SUpL39TfeeIOXX34Zp9NJamoqU6ZMweFwMGTIEOLi4gDo2LEj06dP91eJJy011WTrVgebNzs44wwz2OWIiIjN+S3UV69eTXl5OYsXLyYrK4sZM2aQkZEBQGlpKbNnz2blypVER0czfvx4PvjgA3r37g3AwoUL/VVWo0pJMXn3Xe1XFxGRpsFvabRhwwb69OkDQI8ePcjOzva+FhERwaJFi4iOjgagsrKSyMhIcnNzKSkpYfTo0dxyyy1kZWX5q7xGkZrqBhTqIiLSNPhtpF5YWEhsbKz3udPppLKyEpfLhcPhoG3btoBnVF5cXMyll17Kpk2bGDNmDNdffz1bt25l7NixvPPOO7hcdZfZqlUMLldwTiv7n//xfM7La0ZCQrNG+Z4JCXGN8n1CkV17t2vfYN/e7do32Lf3QPXtt1CPjY2lqKjI+9w0zWrhbJomM2fOJC8vj7lz52IYBsnJySQlJXkfx8fHU1BQQGJiYp3vs39/sb9a8MmzXRJHbq5Ffn4hjpMcsCckxFFQcLgxSgs5du3drn2DfXu3a99g394bu+/6NhD8Nm/cs2dP1q5dC0BWVhapqanVXp88eTJlZWXMmzfPOw2fmZnJjBkzAMjPz6ewsJCEhAR/lXjSWrWCtm1NiosNdu0ygl2OiIjYnN9G6gMHDmTdunWMGDECy7KYNm0aK1eupLi4mLPOOovMzEwuuOACbr31VgBuueUWhg8fTnp6OjfeeCOGYTBt2rR6p96bgq5dTfbu9RwB37GjO9jliIiIjfktMR0OB1OnTq22rEuXLt7Hubm5ta73zDPP+Kskv+ja1eTTTz0Hy/Xrp1AXEZHg0WHbJ6lrV10uVkREmgYl0UlSqIuISFOhJDpJCnUREWkqlEQnqUMHi5gYi4ICBwcOBLsaERGxM4X6SXI4oEsXjdZFRCT4lEKNQFPwIiLSFNR5Stvy5cvrXXHIkCGNXEroOhLqTqAyuMWIiIht1Rnqn3/+OQDbt29n27Zt9O3bF6fTyccff0xKSopC/SipqRqpi4hI8NUZ6lX3MR81ahQrVqygdevWABw8eJA777wzMNWFiJQUhbqIiASfzxTas2cP8fHx3ufR0dEUFBT4s6aQ07mzicNhsW2bQWlpsKsRERG78nmZ2F//+tfcdtttDBo0CMuyePvtt/nNb34TiNpCRmQkJCVZ5OU5yMtzcOaZZrBLEhERG/IZ6unp6axatYovvvgCwzAYPXo0AwYMCERtISU11SQvz3NjF4W6iIgEQ4N2Ardt25aUlBTuv/9+WrRo4e+aQpL2q4uISLD5TKCXX36Z2bNn89JLL1FcXMzkyZP529/+FojaQkpqqucObQp1EREJFp8JtGzZMv72t78RHR1NfHw8mZmZLFmyJBC1hRSN1EVEJNh8JpDD4SAiIsL7PDIyEqfT6deiQlHVBWh++MGBqV3qIiISBD5D/cILL+TJJ5+kpKSE1atX88c//pFevXoForaQEh8PCQkmJSUGO3cawS5HRERsyGeo33///SQlJdGtWzeWL19O3759eeCBBwJRW8jRleVERCSYfJ7S5nA4uOaaa+jbty+WZQGeC9Kceuqpfi8u1KSkmKxb5wn1/v3dwS5HRERsxmeoz58/n+eff574+HgMw8CyLAzD4L333gtEfSFFI3UREQkmn6GemZnJ6tWrvdd+l7rpCHgREQkmn+mTmJhIy5YtA1FLyNNIXUREgsnnSP30009n5MiRXHTRRdVObbvrrrv8WlgoSky0iImx2LvXwf790KpVsCsSERE78TmkbN++PX369KkW6FI7h0NT8CIiEjw+R+oakR+frl1N/vMfJ5s3O7nwQl2FRkREAqfOUB86dCjLli3jjDPOwDCOXEyl6uj3nJycgBQYaqquLKeRuoiIBFqdob5s2TIAcnNzA1ZMOFCoi4hIsPicfv/ll19YsWIFRUVFWJaFaZrs2LGDp556KhD1hRyFuoiIBIvP5Bk3bhw5OTmsWLGCkpISVq1ahcOhwKpLcrKJ02mxfbtBaWmwqxERETvxmc579uzhySefpH///gwaNIh//OMffP/994GoLSRFRkJSkoVpGvz4ozZ+REQkcHymTtWFZ5KTk8nNzaWVTr72KTXVc913TcGLiEgg+dyn3qtXL+6++24eeOABRo8ezXfffUdUVFQgagtZOlddRESCwWeo33vvvWzfvp0OHTowa9Ys1q9fr3PXfdDlYkVEJBjqDPXly5dXe/7VV18BEB8fzyeffMKQIUP8WVdI00hdRESCoc5Q//zzz+td0Veom6bJlClT2LhxIxERETz++OMkJSV5X3/jjTd4+eWXcTqdpKamMmXKFIB61wkVVae1bdniwDQ9l48VERHxtzpDffr06d7HlZWVbNy4EafTSbdu3apdYa4uq1evpry8nMWLF5OVlcWMGTPIyMgAoLS0lNmzZ7Ny5Uqio6MZP348H3zwAW63u851QknLltCuncmePQ527DDo1MkKdkkiImIDPvepf/LJJ9x///20a9cO0zQ5dOgQs2fP5pxzzql3vQ0bNtCnTx8AevToQXZ2tve1iIgIFi1aRHR0NODZaIiMjOSjjz6qc51Qk5rqCfXNmx106uQOdjkiImIDPkN92rRp/O///i9nnHEGAN9++y2PPvooS5curXe9wsJCYmNjvc+dTieVlZW4XC4cDgdt27YFYOHChRQXF3PppZfy9ttv17lOXVq1isHlcvpqI+DOPhs+/hh2744hIaHh6yUkxPmvqCbOrr3btW+wb+927Rvs23ug+vYZ6hEREd5ABzj77LMb9I1jY2MpKiryPjdNs1o4m6bJzJkzycvLY+7cuRiG4XOd2uzfX9ygegLttNOaAVF8/XU5BQVlDVonISGOgoLD/i2sibJr73btG+zbu137Bvv23th917eB4PMQrgsuuICHHnqIb775huzsbJ588kk6dOjA+vXrWb9+fZ3r9ezZk7Vr1wKQlZVFampqtdcnT55MWVkZ8+bN807D+1onlOgIeBERCTSfI/WqW6w+/fTT1ZY/++yzGIbBK6+8Uut6AwcOZN26dYwYMQLLspg2bRorV66kuLiYs846i8zMTC644AJuvfVWAG655ZZa1wlVOlddREQCzbAsq95Ds0tKSrwj6So7d+6kQ4cOfi2soZrqVI5lQefOsRQVGeTkFNKmje8j4O06NQX27d2ufYN9e7dr32Df3pvU9PuQIUPIysryPn/11VdJS0trlMLCmWEcOV/9hx80WhcREf/zOf3+xBNPkJ6eTv/+/fn++++JjIzktddeC0RtIS8lxSQry8nmzQ4uukintYmIiH/5DPULLriAUaNGMXPmTJo3b878+fM59dRTA1FbyNN+dRERCSSfoT5q1CgcDgcrV65k586dTJgwgX79+jFp0qRA1BfSdAS8iIgEks+0GTRoEC+//DIdO3bkoosuYunSpZSVNey8a7vTSF1ERAKpzrT5z3/+A3hG6kdr3rw5PXr08GtR4eL0002cTovt2w1KSoJdjYiIhLs6Q/3RRx/1Pj72aPeXXnrJbwWFk4gISE42sSyDH3/UaF1ERPyrzqQ5+vT1Y6fbfZzaLkfRfnUREQmUOpPm6NurHnur1YbcelU8qs5VV6iLiIi/KWn8TKEuIiKBUucpbbt27SI9Pb3G46rn0jAKdRERCZQ6Q/3o89AvvPDCaq8d+1zqVhXqW7Y4cLvB2fRu/S4iImGizlAfOnRoIOsIWy1aQPv2Jvn5DnbsMEhK0kGGIiLiH5oTDgBdhEZERAJBKRMAOq1NREQCoUEpU1xcTG5uLpZlUVxc7O+awo5G6iIiEgg+U+bTTz/luuuu409/+hN79+6lX79+fPzxx4GoLWxopC4iIoHgM2VmzZrFq6++SosWLUhISOCf//wnTz31VCBqCxsaqYuISCD4TBnTNElISPA+T0lJ8WtB4eiUUyxiYy1++cXBvn26Gp+IiPiHz1A/5ZRT+OCDDzAMg0OHDpGRkcGpp54aiNrChmHoIjQiIuJ/PhNm6tSprFy5kt27dzNw4EBycnKYOnVqIGoLK9qvLiIi/lbnxWeqtGnThttvv51Zs2Zx+PBhsrOzadeuXSBqCytV+9U3bVKoi4iIf/hMmKeffpqnn34agJKSEubNm8fcuXP9Xli4qRqp//CDQl1ERPzDZ8KsWbOGF154AYB27drx4osv8u677/q9sHCjI+BFRMTffCZMZWUlpaWl3ucVFRV+LShcnX66ictl8dNPBiUlwa5GRETCkc996iNGjOC3v/0t/fv3B2Dt2rXcdNNNfi8s3DRrBsnJJps3O9myxcFZZ5nBLklERMKMz1D/3e9+x/nnn8/69etxuVzMnDmT7t27B6K2sJOS4gn1zZsV6iIi0vgaNP2+b98+WrduTYsWLdi0aRPLly8PQGnhR/vVRUTEn3yO1CdMmMCuXbvo0qULhnHkamhDhgzxZ11hSeeqi4iIP/kM9Y0bN/L2229XC3Q5MRqpi4iIP/lMly5dulBQUBCIWsJe1Uh9yxYHbneQixERkbDjc6ReWlrKlVdeSWpqKhEREd7lr7zyil8LC0dxcZCYaLJ7t4OffjI4/XQr2CWJiEgY8Rnqd9xxRyDqsI2UFE+ob97s4PTTNVwXEZHG43P6/cILLyQ2NhaHw4FhGJimyfbt2wNRW1jSfnUREfEXnyP1hx9+mC+++IKDBw/SuXNncnNz6dmzJ8OHDw9EfWFHR8CLiIi/+EyWTz75hDfffJMrrriCxx57jFdeeaXaZWPrYpomkydPJi0tjVGjRrFt27YaX1NSUsKIESPYsmWLd9mQIUMYNWoUo0aNIj09/Tjbafp0X3UREfEXnyP1du3a0axZM7p06cLGjRu5+uqrOXz4sM9vvHr1asrLy1m8eDFZWVnMmDGDjIwM7+vffvstjz76KPn5+d5lZWVlACxcuPBEegkJR6bfnVgW6ExBERFpLD6Hi+3bt2fBggWcd955LFq0iDfffJPy8nKf33jDhg306dMHgB49epCdnV3t9fLycp577jk6d+7sXZabm0tJSQmjR4/mlltuISsr6zjbafrat7eIjbXYv99g3z4luoiINB6fI/UnnniCDz/8kHPOOYdBgwbxxhtvMGXKFJ/fuLCwkNjYWO9zp9NJZWUlLpfnLc8///wa60RFRTFmzBiuv/56tm7dytixY3nnnXe869SmVasYXC6nz3qaku7d4YsvoKAgljPPrP5aQkJccIpqAuzau137Bvv2bte+wb69B6rvOtOyoKCAhIQEDh06xHnnnceuXbsYMGAAAwYMaNA3jo2NpaioyPvcNM16wxkgOTmZpKQkDMMgOTmZ+Ph4CgoKSExMrHOd/fuLG1RPU3L66VF88UUz1q8v5cwzj9zKNiEhjoIC37s2wpFde7dr32Df3u3aN9i398buu74NhDpT9uGHH2bBggXcfPPNGIaBZVnVPr/33nv1vmnPnj354IMPuOqqq8jKyiI1NdVnoZmZmWzatIkpU6aQn59PYWEhCQkJPtcLNTqtTURE/KHOUF+wYAEAjzzyCP369Tvubzxw4EDWrVvHiBEjsCyLadOmsXLlSoqLi0lLS6t1neHDh5Oens6NN96IYRhMmzbN5+g+FOm0NhER8QfDsqx6r1V69dVX8+abbwaqnuMWilM5P/xgcMklsZx2msmGDUd2Udh1agrs27td+wb79m7XvsG+vTeJ6fcqp512Gunp6Zx77rlERUV5l+vWqycuKcnC5bL46ScHxcUQExPsikREJBz4DPVWrVoB8M0331RbrlA/cc2aQefOJps2OdmyxcHZZ5vBLklERMKAz1CfPn16jWUNuaKc1C8lxRPqmzcr1EVEpHH4DPX333+f2bNnU1xcjGVZmKZJaWkpn376aSDqC1upqSZvvQWbNulgORERaRwNGqk/9thjvPjii/zhD39g9erVlJSUBKK2sFZ1BPwPPyjURUSkcfhMlLi4OHr16sW5557L4cOHmThxIp999lkgagtrOlddREQam89EiYqKIi8vjy5duvDFF19QXl5ORUWFr9XEh6qR+o8/OnC7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biIjI4NVml8d2/f+/ft5+umnefDBBwPy/uH/PyoENG/eHIDCwkLuvvtuxo0bF9yCAmTp0qW0bt2aPn368Pzzzwe7nIDav38/u3btYv78+ezYsYM//vGPvPPOOxiGEezS/ComJoadO3fym9/8hv379zN//vxgl+R3V1xxBTt27PA+tyzL+3Nu3rw5hw8fDlZpfnVs31Ub7l999RX/+Mc/+Oc//xms0vzq6L7dbjcPPfQQDz74YMA2YjRSbyJ2797NLbfcwnXXXce1114b7HICYsmSJXzyySeMGjWKnJwcHnjgAQoKCoJdVkDEx8fTu3dvIiIi6Ny5M5GRkfzyyy/BLsvvXnrpJXr37s2qVat4/fXXmTRpUrVpSjtwOI782S0qKqJFixZBrCaw3nrrLR599FGef/55WrduHexy/O67775j27ZtTJkyhfHjx/PDDz/wxBNP+PU9NVJvAvbu3cvo0aOZPHkyF198cbDLCZijt9RHjRrFlClTSEhICGJFgXP++efzyiuvcNttt7Fnzx5KSkqIj48Pdll+16JFC5o1awZAy5YtqaysxO12B7mqwOrevTuff/45F110EWvXrqVXr17BLikgXn/9dRYvXszChQtt8bsOcM455/Dmm28CsGPHDsaPH89DDz3k1/dUqDcB8+fP59ChQ8ybN4958+YBnoMt7HbwmJ3069eP9evXM3z4cCzLYvLkybY4luJ3v/sdDz74ICNHjqSiooJ7772XmJiYYJcVUA888ACPPPIIs2bNonPnzlxxxRXBLsnv3G43TzzxBImJify///f/APif//kf7r777iBXFn50lzYREZEwoX3qIiIiYUKhLiIiEiYU6iIiImFCoS4iIhImFOoiIiJhQqEuEkA7duzgrLPO4rrrrmPIkCFcffXV3HbbbbVe+z0/P5+xY8ee0Ptcd911J7Te559/XuNGFFXWrFnDiBEjGDx4MNdccw2zZ88O+ZuRvPbaa7zxxhvBLkOk0SjURQKsXbt2vP766yxfvpw333yTbt268dRTT9X4uvbt25/wTS9ef/31ky2zmrVr1zJ16lSmT5/OihUryMzMJDc3l2effbZR3yfQvvrqK8rLy4Ndhkij0cVnRILsoosuYtasWQD079+fc845h5ycHGbOnMm4ceN4//33mTRpErGxsXz33Xfk5+dz5513MmzYMA4cOMBDDz3Ejz/+SEREBJMmTeLiiy+mW7dubNy4kblz57Jr1y62bNnC/v37SUtL4/bbb6ewsJAHH3yQ/Px89uzZw8UXX1zv5Svnz5/PH//4R5KTkwGIiopiypQp/PjjjwDk5eUxefJkDhw4QExMDA899BDnnHMOkyZNIjo6mu+//55Dhw4xfvx4Xn/9dXJzc7n88suZNGkSS5cuZc2aNezbt4+CggL69evHpEmTMAyD+fPns2LFCpxOJ5deeikTJ05k9+7d3HXXXXTt2pWcnBzatGnDnDlziI+PZ+3atTz77LNUVlbSsWNHHnvsMVq1akX//v0ZPHgwH3/8MSUlJTz55JMcOnSI999/n88++4yEhAT69Onj/x+2iJ9ppC4SRBUVFaxatYoePXp4l1122WWsWrWqxrWxf/75Z1599VUyMjK8I/s5c+bQqVMn3n77bZ566qlab2WanZ3Niy++yNKlS1m8eDHfffcda9as4cwzz2Tx4sWsWrWK9evX891339VZZ05ODt27d6+27JRTTuGSSy4BYOLEiYwaNYqVK1eSnp7OPffc4x0B79mzh8WLF/P73/+e9PR0/vznP7N8+XJee+01781MNmzYwJw5c3jjjTf45ptv+Pe//82HH37I+++/z5IlS1i2bBnbtm1j0aJFAOTm5nLbbbfxxhtv0KJFC1auXMkvv/zCM888w9/+9jeWL19O7969efrpp731xsfHk5mZyYgRI1iwYAGXXHIJ/fv35+6771agS9jQSF0kwPbs2ePd511eXs4555zDhAkTvK+fe+65ta536aWXYhgGqampHDhwAID169d7g6tbt24sXry4xnrXXHON906A/fv357PPPmPMmDH85z//4aWXXuLHH3/kwIEDFBcX11mzYRh13mWqqKiI7du3M2jQIAB69OhBy5YtvaP4yy67DIBTTz2Vrl270qZNG8ATsgcPHgRgwIABtG3bFoCrrrqKzz77jMjISK6++mqio6MBGDZsGMuXL6dv3760adPGu5HRtWtXDh48yDfffOO9MRKAaZq0bNnSW2dVcHft2pV33323zl5FQplCXSTAqvap16Wu8KxafvTtWV0uV7XnW7Zs8U6RVzn6mvKmaeJ0Olm4cCGrVq3ihhtu4JJLLmHTpk3Ud8Xos846i+zsbFJSUrzL8vLyyMjIYPLkyTW+3rIs741aqm7gUlVvbWqrsbaD8CorK4Hq/0aGYXjfr2fPnt7buZaVlVFUVOT9utr+/UTCjabfRULYBRdc4L0L1JYtWxg7dmyN0Fq9ejXl5eUcPHiQDz74gN69e7Nu3TrS0tIYPHgwZWVl5Obm1nsk++23385f//pXtm7dCnhG5zNmzCAxMZHY2Fg6duzoHf1mZWWxd+9eunbt2uA+PvroIw4fPkxZWRlvvvkml112Gb169eLNN9+ktLSUyspKlixZUu8dzc4991yysrLIy8sDYN68ebUegHg0p9Npu7vESXjTSF0khN199908/PDDDB48GJfLxVNPPVUj1CMjIxk5ciSFhYXccccdpKSkcOuttzJlyhSef/55YmNjOe+889ixYwedOnWq9X0uu+wy7r33Xu69917cbjeVlZVceeWV3HXXXQDMnDmTKVOmMHfuXJo1a8bcuXOJiIhocB+tW7dm7Nix7N+/n8GDB3unynNychg2bBiVlZX07t2bm2++udbT/wASEhKYNm0a48aNwzRN2rdvz8yZM+t930suuYRZs2YRFxfHlVde2eB6RZoq3aVNJIzNnTsXwHu7y6Zo6dKlfPHFF8yYMSPYpYiEPE2/i4iIhAmN1EVERMKERuoiIiJhQqEuIiISJhTqIiIiYUKhLiIiEiYU6iIiImFCoS4iIhIm/j/Hqmd496yFEAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# SCREE PLOT for PCA (pca on x_tune)\n",
    "# Visualize eigenvalues/variances\n",
    "\n",
    "# import matplotlib.pyplot as plt\n",
    "# import numpy as np\n",
    "\n",
    "PC_values = np.arange(pca14.n_components_) + 1\n",
    "plt.plot(PC_values, pca14.explained_variance_ratio_, 'o-', linewidth=2, color='blue')\n",
    "plt.title('Scree Plot')\n",
    "plt.xlabel('Principal Component')\n",
    "plt.ylabel('Variance Explained')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*reference: https://www.kaggle.com/code/robroseknows/pubg-clustering-exploration/notebook* <br/>\n",
    "*reference: https://www.statology.org/scree-plot-python/* <br/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The bend in the elbow of the scree plot (above) is a good indicator for how many eigenvectors to use. Here we see it happens at component 3 or 4, meaning 3 components are sufficient to explain the most variance with the most parsimonious model. Therefore, we'll proceed with cluster analysis with 3 principal components."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.5090363398194313\n"
     ]
    }
   ],
   "source": [
    "# check variance explained by 3 components (using scaled_x_tune)\n",
    "\n",
    "# define PCA model to use (3)\n",
    "pca3 = PCA(n_components=3)\n",
    "\n",
    "# fit PCA model to data\n",
    "pca3_fit = pca3.fit(scaled_x_tune)\n",
    "\n",
    "# how much of the variance is explained by 3 components\n",
    "print(sum(pca3.explained_variance_ratio_))\n",
    "\n",
    "# store the principal components in variable\n",
    "P3 = pca3.transform(scaled_x_tune)\n",
    "\n",
    "# reference: https://www.kaggle.com/code/robroseknows/pubg-clustering-exploration/notebook\n",
    "# reference: https://medium.com/more-python-less-problems/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Looks like around 51% of the variance in the data can be explained by using 3 components. <br/>\n",
    "We will use this data (scaled, PCA = 3) to build our clusters. <br/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we will plot the first 2 components to see if there are any clear clusters that jump out.\n",
    "\n",
    "*reference: https://medium.com/more-python-less-problems/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'PCA 2')"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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wklw8rjdqOjfiTY5RVGUrbBPEq6Ksn6x3kZnrPth4ZtFtbqyV363hr/oRDZFUYaJCrqyr1irWG+r/fNjYyyA2+2K/p2q0FcURdkqKujKvmNQsd5mBsKsZrcr+nUim6Io+wE9Q1cODNuxYN1Ni9qdttNVFEXZChqhKweK/dY/DxwMO11FUUYfjdCVA8d+GwhzqQr9/bJORVFGGxV0RdkBdCKboijDRlPuirJD7MfjAEVRDg8q6Iqyg+wnO11FUQ4XmnJXFEVRlBFABV1RFEVRRgAVdEVRFEUZAVTQFUVRFGUEUEFXFEVRlBFABV1RFEVRRgAVdEVRFEUZAVTQFUVRFGUEMNauNSdKURRFUZSDgkboiqIoijICqKAriqIoygiggq4oiqIoI4AKuqIoiqKMACroiqIoijICqKAriqIoygigk5uHjOu6+OhHP4onnngC4XAYH//4x3HZZZcNe1mHiu9///v4L//lv+C+++7DM888g9/4jd+AMQZXXXUVPvKRj8BxdN+7W7TbbXzgAx/Ac889h1arhXe84x143vOep5/BHtLtdvHBD34QZ86cQSAQwJ133glrrX4GQ2BpaQm/+Iu/iP/5P/8ngsHgpj8D/YSGzF/91V+h1Wrh/vvvxx133IG77rpr2Es6VNx777344Ac/iGazCQC48847cfvtt+Nzn/scrLX42te+NuQVjjYPPvggstksPve5z+Hee+/Fxz72Mf0M9phvfOMbAIDPf/7zeOc734k777xTP4Mh0G638eEPfxjRaBTA1n4XqaAPme9+97t42cteBgC4/vrr8cgjjwx5RYeLkydP4p577ul//eijj+IlL3kJAODlL385Hn744WEt7VDw6le/Gr/+67/e/zoQCOhnsMe86lWvwsc+9jEAwPnz5zExMaGfwRC4++678YY3vAFTU1MAtva7SAV9yFQqFSSTyf7XgUAAnU5niCs6XNx0000IBr2TJ2stjDEAgEQigXK5PKylHQoSiQSSySQqlQre+c534vbbb9fPYAgEg0G8733vw8c+9jHcdNNN+hnsMV/+8pcxNjbWD+6Arf0uUkEfMslkEtVqtf+167orBEbZW/xnVNVqFel0eoirORzMzs7il37pl/DzP//zeO1rX6ufwZC4++678ZWvfAUf+tCH+kdQgH4Ge8EDDzyAhx9+GLfeeisee+wxvO9970M+n+9/f6OfgQr6kHnxi1+Mhx56CADwve99D1dfffWQV3S4ufbaa/Gd73wHAPDQQw/hhhtuGPKKRpvFxUXcdttteM973oPXv/71APQz2Gv+6I/+CJ/+9KcBALFYDMYYvOhFL9LPYA/57Gc/iz/4gz/Afffdhxe84AW4++678fKXv3zTn4EOZxkyUuX+5JNPwlqLT3ziE7jyyiuHvaxDxblz5/Cud70LX/jCF3DmzBl86EMfQrvdxqlTp/Dxj38cgUBg2EscWT7+8Y/jz//8z3Hq1Kn+bf/xP/5HfPzjH9fPYI+o1Wp4//vfj8XFRXQ6Hbz97W/HlVdeqT8HQ+LWW2/FRz/6UTiOs+nPQAVdURRFUUYATbkriqIoygiggq4oiqIoI4AKuqIoiqKMACroiqIoijICqKAriqIoygigDiaKcsg4d+4cXv3qV+PKK6+EMQbtdhtTU1O48847ceTIEQDsTb7vvvvQ6XTgui5uvvlm/NIv/dKK6/ziL/4ipqam8KlPfWrN52q323jb296Gf//v/z1e+tKX7urrUpTDjgq6ohxCpqam8Md//Mf9r++66y785m/+Jj75yU/i/vvvx+c//3l8+tOfxtTUFEqlEm677TbEYjHcfPPNAIDHH38c4XAYjz/+OGZnZ3H06NGLnuOpp57CBz7wAfzwhz/cs9elKIcZTbkrioKXvvSlOH36NADgf/yP/4H3vOc9/SER6XQad9999woXwy9/+cv4mZ/5Gbzyla/EF77whYHX/NKXvoS3ve1t+Imf+IndfwGKoqigK8php91u4ytf+Qquv/565PN5zM7O4tprr11xnyuvvLIvzO12G3/yJ3+C17zmNXjNa16DL33pSwMHCr33ve/Fq171qj15DYqiaMpdUQ4l8/Pz+Pmf/3kAQKvVwnXXXYc77rij//1IJLLmY7/5zW9icnISz3ve82CtheM4+MY3voEbb7xx19etKMraqKAryiFk9Rm6n5mZGTzyyCP4qZ/6qf5tf/d3f4eHHnoI7373u/HAAw9gdnYWr3jFKwBwBPDnP/95FXRFGTKaclcUZQVvfetbcdddd2FhYQEAkM/ncdddd+Gyyy7D4uIiHn74Yfzpn/4pvv71r+PrX/86/uiP/gj/9//+X5w9e3bIK1eUw41G6IqirOCNb3wjOp0ObrvtNhhjYK3FLbfcgptvvhm/93u/h5/7uZ/D9PR0//4zMzN4xStegfvvvx/vfve7h7hyRTnc6LQ1RVEURRkBNOWuKIqiKCOACrqiKIqijAAq6IqiKIoyAqigK4qiKMoIoIKuKIqiKCOACrqiKIqijAAq6IqiKIoyAqigK4qiKMoI8P8DVVhozyA3So4AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# store the principal components (3) into pca3_comp_stg object\n",
    "pca3_comp_stg = pca3.fit_transform(scaled_x_tune)\n",
    "\n",
    "# save components to dataFrame\n",
    "pca3_comp = pd.DataFrame(pca3_comp_stg)\n",
    "\n",
    "# plot the components to see if we can find clusters\n",
    "plt.scatter(pca3_comp[0], pca3_comp[1], alpha = 0.1, color = 'blue')\n",
    "plt.xlabel('PCA 1')\n",
    "plt.ylabel('PCA 2')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There is no readily visible grouping to help a human understand separation within the data. We'll continue by trying 3-dimensional visualization."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### PCA 3D plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sherm\\AppData\\Local\\Temp/ipykernel_22772/1455752897.py:5: MatplotlibDeprecationWarning: Axes3D(fig) adding itself to the figure is deprecated since 3.4. Pass the keyword argument auto_add_to_figure=False and use fig.add_axes(ax) to suppress this warning. The default value of auto_add_to_figure will change to False in mpl3.5 and True values will no longer work in 3.6.  This is consistent with other Axes classes.\n",
      "  ax = Axes3D(fig_p3, elev=17, azim=75)\n",
      "C:\\Users\\sherm\\AppData\\Local\\Temp/ipykernel_22772/1455752897.py:7: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.\n",
      "  fig_p3.show()\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot first 3 componenets - using first 100,000 records\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "fig_p3 = plt.figure()\n",
    "# ax = Axes3D(fig_p3, elev=48, azim=134) # original\n",
    "ax = Axes3D(fig_p3, elev=17, azim=75)\n",
    "ax.scatter(P3[:100000, 0], P3[:100000, 1], P3[:100000, 2])\n",
    "fig_p3.show()\n",
    "\n",
    "# # rotate the axes and update\n",
    "# for angle in range(0, 360):\n",
    "#     ax.view_init(50, angle)\n",
    "#     plt.draw()\n",
    "#     plt.pause(.001)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This does not really look like we are finding the \"clusters\" we expect."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that we have reduced dimensionality of the data using PCA, we can do some clusters and see if they helped. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgkAAAFlCAYAAABhvHtEAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAABBQElEQVR4nO3deXxU9aH//9dkZjKZZCYbCXuCiCCiZUkAUQMWFa0UqiUGUNTLV35FuUWtggVpRbgsoVL0Vraqt/ZBbdWA1O3WWrVAKEjFcI0IKsoeSYCskJksk5k5vz8SRgMBApKZSXg/H488khwOk/dhyXnncz7nc0yGYRiIiIiInCQi1AFEREQkPKkkiIiISJNUEkRERKRJKgkiIiLSJJUEERERaZJKgoiIiDTJEuoAIhJ6l19+Ob169SIiovHPDcuXL+fQoUPMmzeP//3f/2XmzJn07NmTSZMmtWie2tpaVq5cyYYNGzAMA7/fz+jRo/nZz36GyWRq0a8tIt9SSRARAFatWkViYuIp2w8dOhTUHIZh8J//+Z90796dnJwcbDYb5eXl3H///VRVVfGLX/wiqHlELma63CAi52Tbtm2MHTuWkSNHsmDBArxeLwB5eXmMHTuW0aNHM2bMGDZu3IjP52PIkCEcOHAAgOeee47hw4cHXmvixInk5uY2ev2PP/6YvXv38vjjj2Oz2QBISEjgqaeeYtCgQQDcc889vPvuu4Hf893Pr7rqKh5++GFuueUWXnjhBR544IHAfnv27GHo0KH4fD727NnDfffdx5gxY7jtttt47bXXWuBPS6R100iCiADwH//xH40uN3Tt2pXly5efst/hw4f585//jMViYdKkSaxevZpbb72Vhx56iJUrV9KvXz++/vpr7r77bl577TWGDx/Ov/71L7p168a//vUv6urq2LdvH0lJSXz55Zdcc801jV5/x44d9O3bF7PZ3Gj7JZdcwiWXXHLW46irq2P48OH87ne/w+Vy8fzzz1NcXExycjJ//etfGTNmDIZh8NBDD/HUU09x5ZVXUllZybhx47jsssvo37//ef35ibRFKgkiApz+csPJbrvtNqKjowH4yU9+Qm5uLl26dCE1NZV+/foB0LNnT9LS0ti6dSsjRozg1Vdf5fbbb6e4uJhRo0bx4YcfEhcXx9ChQ4mMjGz0+hEREXzf1eIHDhwIgMPhYMSIEbz11ltMnDiRt99+m7/85S/s37+fgwcPMmvWrMDvqamp4fPPP1dJEPkOlQQROSff/QnfMAwsFgs+n++UCYWGYeD1ernuuuv49a9/TW5uLldffTXXXnstr7zyCna7nZEjR57y+v369WPVqlX4fL5GX2v79u289NJLLF68OPD6J9TV1TV6jRMlBmDs2LE88cQT9OjRgx49epCSksKuXbtwOp28+eabgf1KSkpwOp3n+aci0jZpToKInJO//e1veDweamtref311xk2bBj9+/dn7969bN++HYCvv/6ajz/+mMGDB2Oz2Rg0aBDLli3juuuuY/DgweTn55OXl8fQoUNPef0BAwZw6aWXkp2dTW1tLVB/Ap8/fz5du3YFIDExkR07dgCwe/dudu3addq8J0YGli9fTlZWFgDdu3cnKioqUBKKiooYNWpU4DVFpJ5GEkQEOHVOAsCjjz5KVFRUo21du3blrrvuwu12M2LECH76059iMpn43e9+x7x586ipqcFkMpGdnU337t0BGDFiBO+99x5DhgwhKiqK3r17ExcXF5iYeLJnn32WZ555hjFjxmA2m/H7/dx+++2BWy+nTJnCzJkzyc3N5dJLLw1cXjidrKwsVqxYwU033QRAZGQkK1asYMGCBfzP//wPXq+Xhx9+mPT09PP6sxNpq0x6VLSIiIg0RZcbREREpEkqCSIiItIklQQRERFpkkqCiIiINEklQURERJqkWyBPUlxceUFfLyEhmvLyqgv6mi1NmYNDmYOnNeZW5uBQZkhOPv0iYhpJaGEWi/nsO4UZZQ4OZQ6e1phbmYNDmc9MJUFERESapJIgIiIiTVJJEBERkSapJIiIiEiTVBJERESkSSoJIiIi0iSVBBEREWmSSoKIiIg0SSVBREREmqSSICIiIk3SsxtERETCiM/vp7rWR1VNXeB9Va2Xqhov1bVerunfFWdkcH7GV0kQERG5gLw+P1U13sCJvar21JN9VW39CT/w8Yn9a73UenxnfP3dRZX8521XBuVYVBJERESa4PP7cdd4cVfX4a7x4qquq/+4ug5XYHtdw/aGX6+po+YsJ/mTmUwQbbNgt1nokGAPfBwdZSHaZsVuMxMdZQ1sv6Z/FzzVnhY66sZUEkREpE0zDIMajw9Xdd0pb4YpguJSN66ahpN/4MRf/5N+c9msZmLsFpLj7TjsVqKjLMQ0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vr5GElpIamoqS5cuDXWMc3LTTTcxb948AAoLC0lKSgpxorP78ssvqa6u5r777uPee+8lPz8/1JGa7bPPPmP37t2MGzcu1FHOavfu3QwbNgyASy+9lD179oQ40emd/H8vLS2NOXPmhC5QM5yceenSpQwaNAiPx0NxcTHt2rULYbqmnZx5x44dbNiwgQkTJjBr1ixcLlcI0zXtdN+Xly5dyt1330379u1DkOrMTs585MgR0tLSgPp/29u2bWvRr6+S0EJuueUWLJbWN1BjsViYMWMG8+bN45Zbbgl1nLOKiopi0qRJ/OEPf2Du3LlMnz4dr9cb6ljN8txzz/Hzn/881DGa5YorrmD9+vUYhkF+fj5HjhwJ2yHwk//vjRw5EpPJFMJEZ3dyZrPZzKFDhxg1ahTl5eV07949hOmadnLmvn378stf/pK//OUvpKSksHz58hCma1pT35dLS0vZsmVLYHQv3JycOSUlha1btwL1o7/V1dUt+vVVEuQUv/nNb/jHP/7BE088QVVVVajjnFH37t35yU9+gslkonv37sTHx1NcXBzqWGd1/Phx9u7dy5AhQ0IdpVkyMzNxOBzce++9rF+/niuvvBKz2RzqWG1aly5deO+997jzzjtZtGhRqOOc1YgRI7jqqqsCH3/++echTtQ87777LqNGjWo1/54XLlzIc889x+TJk2nXrh0JCQkt+vVUEiTgjTfe4LnnngPAbrdjMpnC/j/Oa6+9FvgGeuTIEVwuF8nJySFOdXYff/wx1157bahjNNtnn31Geno6L730EjfddBMpKSmhjtSmPfDAA4HJoTExMUREhP+36kmTJrF9+3YAtmzZwpVXXhniRM2zZcuWwKW01iA3N5eFCxfy/PPPU1FRwXXXXdeiX6/1jYdLi7n55pt5/PHHmTBhAl6vl1mzZmGz2UId64zuuOMOHn/8ce68805MJhMLFy5sFZd59u3bR9euXUMdo9m6devG7373O1588UWcTicLFiwIdaQ2bfLkycycOROr1Yrdbmf+/PmhjnRWc+bMYd68eVitVpKSkgLzm8Ldvn37WlXp7datG5MnT8Zut3P11Vdz/fXXt+jX01MgRUREpEnhP4YlIiIiIaGSICIiIk1SSRAREZEmqSSIiIhIk1QSREREpEkqCSJtxDfffMPll1/O5s2bG22/4YYb+Oabb77361+o1zmT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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "n_clusters = range(1, 20)\n",
    "kmeans = [KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=0.0001,\n",
    "                 verbose=0, random_state=None, copy_x=True) \n",
    "          for i in n_clusters]\n",
    "score = [kmeans[i].fit(P3).score(P3) for i in range(len(kmeans))]\n",
    "plt.plot(n_clusters,score)\n",
    "plt.xticks([1,3,5,7,9,11,13,15,17,19], [1,3,5,7,9,11,13,15,17,19])\n",
    "plt.xlabel('Number of Clusters')\n",
    "plt.ylabel('Score')\n",
    "plt.title('Elbow Curve')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Visually speaking, we may have reached a plateau in improvement by 13 clusters. We'll proceed under that assumption."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "FIT NEW MODEL WITH 13 CLUSTERS + CHECK PERFORMANCE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "# start with clusters = 13\n",
    "kms = KMeans(n_clusters=13).fit(P3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.style.use(\"seaborn\")\n",
    "plt.scatter(P3[:100000, 0], P3[:100000, 1], c=kms.labels_[:100000])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Again, these are not human-discernible."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pc_0</th>\n",
       "      <th>pc_1</th>\n",
       "      <th>pc_2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1.006365</td>\n",
       "      <td>1.738836</td>\n",
       "      <td>0.291727</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.747356</td>\n",
       "      <td>1.102793</td>\n",
       "      <td>0.418179</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.493516</td>\n",
       "      <td>0.264003</td>\n",
       "      <td>-1.372072</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-1.054105</td>\n",
       "      <td>-1.479501</td>\n",
       "      <td>1.299936</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.697082</td>\n",
       "      <td>-0.811916</td>\n",
       "      <td>-2.419552</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       pc_0      pc_1      pc_2\n",
       "0 -1.006365  1.738836  0.291727\n",
       "1 -0.747356  1.102793  0.418179\n",
       "2 -1.493516  0.264003 -1.372072\n",
       "3 -1.054105 -1.479501  1.299936\n",
       "4  3.697082 -0.811916 -2.419552"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# store our first 3 principal components in dataFrame\n",
    "column_names = ['pc_0','pc_1','pc_2']\n",
    "pca3_slim = pd.DataFrame(P3, columns = column_names)\n",
    "\n",
    "pca3_slim.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "# NEED TO PULL SAMPLE OF 2000 FOR PCA3_SLIM TO MATCH SCALED_X_TUNE\n",
    "\n",
    "x_pca_slim = pca3_slim.loc[indices_slim, :]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Test accuracy with added features = clusters from first 3 principal components."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Clusters 4 Average accuracy: PCA 3 WITH other attributes =  83.0 +- 2.3345235059857496\n",
      "Clusters 5 Average accuracy: PCA 3 WITH other attributes =  83.65 +- 2.7843311584651707\n",
      "Clusters 6 Average accuracy: PCA 3 WITH other attributes =  83.65 +- 2.3774986855937477\n",
      "Clusters 7 Average accuracy: PCA 3 WITH other attributes =  83.39999999999999 +- 2.385372088375312\n",
      "Clusters 8 Average accuracy: PCA 3 WITH other attributes =  83.55 +- 2.161596632121728\n"
     ]
    }
   ],
   "source": [
    "# PCA\n",
    "# this code will add new features to represent PCs 1, 2, 3\n",
    "# IN ADDITION TO the other features in cols_df\n",
    "\n",
    "# pca3_slim\n",
    "\n",
    "# X1 = df_imputed[['Pclass','Fare']]\n",
    "# X2 = df_imputed[['Age','Parch','SibSp']]\n",
    "\n",
    "params = []\n",
    "\n",
    "# we will vary parameters on match-total with # clusters 4 - 9\n",
    "for i in range(4,9):\n",
    "   # for x_heal in range(3,8):\n",
    "        # get the first clustering\n",
    "        cls_p3 = KMeans(n_clusters=i, init='k-means++',random_state=17)\n",
    "        cls_p3.fit(x_pca_slim)\n",
    "        newfeature_p3 = cls_p3.labels_ # the labels from kmeans clustering\n",
    "\n",
    "        # # # append on the second clustering\n",
    "        # cls_heal_kill = KMeans(n_clusters=x_heal, init='k-means++',random_state=17)\n",
    "        # cls_heal_kill.fit(x_heal_kill)\n",
    "        # newfeature_heal_kill = cls_heal_kill.labels_ # the labels from kmeans clustering\n",
    "\n",
    "        y = y_tune_slim.loc[:, ('quart_binary')]\n",
    "        X = x_pca_slim\n",
    "        # X = np.column_stack((X,pd.get_dummies(newfeature_match_total)))\n",
    "        X = np.column_stack((X,pd.get_dummies(newfeature_p3)))\n",
    "\n",
    "        acc = cross_val_score(clf,X,y=y,cv=cv)\n",
    "        # params.append((x_match,acc.mean()*100,acc.std()*100)) # save state\n",
    "        params.append((i,acc.mean()*100,acc.std()*100)) # save state\n",
    "\n",
    "        # print (\"Clusters\",x_match,\"Average accuracy = \", acc.mean()*100, \"+-\", acc.std()*100)\n",
    "        print (\"Clusters\",i,\"Average accuracy: PCA 3 WITH other attributes = \", acc.mean()*100, \"+-\", acc.std()*100)\n",
    "\n",
    "# reference: 09.Clustering and discretization.ipynb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Test accuracy with ONLY clusters from first 3 principal components."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Clusters 4 Average accuracy: PCA 3 WITHOUT other attributes =  84.5 +- 2.3130067012440754\n",
      "Clusters 5 Average accuracy: PCA 3 WITHOUT other attributes =  83.85000000000001 +- 1.8309833423600548\n",
      "Clusters 6 Average accuracy: PCA 3 WITHOUT other attributes =  83.10000000000001 +- 3.088689042296098\n",
      "Clusters 7 Average accuracy: PCA 3 WITHOUT other attributes =  81.89999999999999 +- 2.199999999999998\n",
      "Clusters 8 Average accuracy: PCA 3 WITHOUT other attributes =  81.80000000000001 +- 1.8601075237738247\n"
     ]
    }
   ],
   "source": [
    "# pca3_comp[0], pca3_comp[1],\n",
    "# pca3_slim\n",
    "\n",
    "# X1 = df_imputed[['Pclass','Fare']]\n",
    "# X2 = df_imputed[['Age','Parch','SibSp']]\n",
    "\n",
    "params = []\n",
    "\n",
    "# we will vary parameters on match-total with # clusters 4 - 9\n",
    "for i in range(4,9):\n",
    "   # for x_heal in range(3,8):\n",
    "        # get the first clustering\n",
    "        cls_p3 = KMeans(n_clusters=i, init='k-means++',random_state=17)\n",
    "        cls_p3.fit(x_pca_slim)\n",
    "        newfeature_p3 = cls_p3.labels_ # the labels from kmeans clustering\n",
    "\n",
    "        # # # append on the second clustering\n",
    "        # cls_heal_kill = KMeans(n_clusters=x_heal, init='k-means++',random_state=17)\n",
    "        # cls_heal_kill.fit(x_heal_kill)\n",
    "        # newfeature_heal_kill = cls_heal_kill.labels_ # the labels from kmeans clustering\n",
    "\n",
    "        y = y_tune_slim.loc[:, ('quart_binary')]\n",
    "        X = pd.get_dummies(newfeature_p3)\n",
    "        \n",
    "        # X = np.column_stack((X,pd.get_dummies(newfeature_p3)))\n",
    "\n",
    "        acc = cross_val_score(clf,X,y=y,cv=cv)\n",
    "        # params.append((x_match,acc.mean()*100,acc.std()*100)) # save state\n",
    "        params.append((i,acc.mean()*100,acc.std()*100)) # save state\n",
    "\n",
    "        # print (\"Clusters\",x_match,\"Average accuracy = \", acc.mean()*100, \"+-\", acc.std()*100)\n",
    "        print (\"Clusters\",i,\"Average accuracy: PCA 3 WITHOUT other attributes = \", acc.mean()*100, \"+-\", acc.std()*100)\n",
    "\n",
    "# reference: 09.Clustering and discretization.ipynb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's vary the parameters of our KMEANS clusters to see what works."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Tune Parameters for KMeans - Number of Clusters\n",
    "\n",
    "Jump to [Top](#TOP)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Clusters 4 Average accuracy =  91.4 +- 1.624807680927191\n",
      "Clusters 5 Average accuracy =  90.9 +- 1.593737745050924\n",
      "Clusters 6 Average accuracy =  90.95 +- 1.863464515358422\n",
      "Clusters 7 Average accuracy =  91.25 +- 1.8200274723201288\n",
      "Clusters 8 Average accuracy =  91.19999999999999 +- 1.9131126469708972\n",
      "Clusters 9 Average accuracy =  90.85000000000001 +- 1.8980252896102299\n",
      "Clusters 10 Average accuracy =  90.75000000000001 +- 2.0766559657295183\n",
      "Clusters 11 Average accuracy =  90.75 +- 2.0031225624010123\n",
      "Clusters 12 Average accuracy =  90.95 +- 1.849999999999998\n",
      "Clusters 13 Average accuracy =  90.75000000000001 +- 1.7066048165876004\n",
      "Clusters 14 Average accuracy =  90.64999999999999 +- 2.1336588293351872\n"
     ]
    }
   ],
   "source": [
    "x_match_total = x_tune_slim[['matchDuration','totalDistance']]\n",
    "x_heal_kill = x_tune_slim[['healItems','killsAssist']]\n",
    "\n",
    "# X1 = df_imputed[['Pclass','Fare']]\n",
    "# X2 = df_imputed[['Age','Parch','SibSp']]\n",
    "\n",
    "params = []\n",
    "\n",
    "# we will vary parameters on match-total with # clusters 4 - 14\n",
    "for x_match in range(4,15):\n",
    "   # for x_heal in range(16,19):\n",
    "        # get the first clustering\n",
    "        cls_match_total = KMeans(n_clusters=x_match, init='k-means++',random_state=17)\n",
    "        cls_match_total.fit(x_match_total)\n",
    "        newfeature_match_total = cls_match_total.labels_ # the labels from kmeans clustering\n",
    "\n",
    "        # # append on the second clustering\n",
    "        # cls_heal_kill = KMeans(n_clusters=x_heal, init='k-means++',random_state=17)\n",
    "        # cls_heal_kill.fit(x_heal_kill)\n",
    "        # newfeature_heal_kill = cls_heal_kill.labels_ # the labels from kmeans clustering\n",
    "\n",
    "        y = y_tune_slim.loc[:, ('quart_binary')]\n",
    "        X = x_tune_slim.loc[:, cols_df]\n",
    "        X = np.column_stack((X,pd.get_dummies(newfeature_match_total)))\n",
    "        # X = np.column_stack((X,pd.get_dummies(newfeature_match_total),pd.get_dummies(newfeature_heal_kill)))\n",
    "\n",
    "        acc = cross_val_score(clf,X,y=y,cv=cv)\n",
    "        params.append((x_match,acc.mean()*100,acc.std()*100)) # save state\n",
    "        # params.append((x_match,x_heal,acc.mean()*100,acc.std()*100)) # save state\n",
    "\n",
    "        print (\"Clusters\",x_match,\"Average accuracy = \", acc.mean()*100, \"+-\", acc.std()*100)\n",
    "        # print (\"Clusters\",x_match,x_heal,\"Average accuracy = \", acc.mean()*100, \"+-\", acc.std()*100)\n",
    "\n",
    "# reference: 09.Clustering and discretization.ipynb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It seems that about the best we can do with these new discretization methods is around 91.3%. All the models are within one standard deviation of each other, so most clustering in this range are pretty reasonable."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, let's capture accuracy & standard deviation when we combine 2 clusters. <br/>\n",
    "We will adjust the number of clusters for `matchDuration`/`totalDistance` and `healItems`/`killsAssist` and see what performance looks like."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Cluster definitions:** <br/>\n",
    "1. Cluster 1 (`match_total`): Builds additional attributes from clusters on `matchDuration`/`totalDistance` <br/>\n",
    "2. Cluster 2 (`heal_kill`): Builds additional attributes from clusters on `healItems`/`killsAssist` <br/>\n",
    "\n",
    "The loop below builds a model that includes **both** clusterings, and returns the average accuracy score (from cross validation), and the average standard deviation for every cluster-count combination of the 2 groupings (Cluster 1, Cluster 2). <br/>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Clusters 4 3 Average accuracy =  91.39999999999999 +- 1.496662954709575\n",
      "Clusters 4 4 Average accuracy =  91.10000000000001 +- 1.5132745950421544\n",
      "Clusters 4 5 Average accuracy =  91.25 +- 1.6007810593582112\n",
      "Clusters 4 6 Average accuracy =  91.05000000000001 +- 1.6039014932345428\n",
      "Clusters 4 7 Average accuracy =  91.05000000000001 +- 1.6499999999999995\n",
      "Clusters 5 3 Average accuracy =  91.05000000000001 +- 1.4908051515875564\n",
      "Clusters 5 4 Average accuracy =  91.20000000000002 +- 1.5198684153570656\n",
      "Clusters 5 5 Average accuracy =  91.1 +- 1.5620499351813322\n",
      "Clusters 5 6 Average accuracy =  90.9 +- 1.8275666882497057\n",
      "Clusters 5 7 Average accuracy =  90.85000000000001 +- 1.7755280904564699\n",
      "Clusters 6 3 Average accuracy =  90.9 +- 1.8275666882497057\n",
      "Clusters 6 4 Average accuracy =  91.05000000000001 +- 1.916376789673679\n",
      "Clusters 6 5 Average accuracy =  90.7 +- 1.5524174696260016\n",
      "Clusters 6 6 Average accuracy =  90.7 +- 1.661324772583614\n",
      "Clusters 6 7 Average accuracy =  91.05000000000001 +- 1.6499999999999986\n",
      "Clusters 7 3 Average accuracy =  91.4 +- 1.8138357147217048\n",
      "Clusters 7 4 Average accuracy =  90.64999999999999 +- 2.1100947846009186\n",
      "Clusters 7 5 Average accuracy =  91.3 +- 1.8867962264113203\n",
      "Clusters 7 6 Average accuracy =  91.10000000000001 +- 1.6999999999999995\n",
      "Clusters 7 7 Average accuracy =  91.25000000000001 +- 1.9137659209004634\n",
      "Clusters 8 3 Average accuracy =  91.05000000000001 +- 1.7095320997278753\n",
      "Clusters 8 4 Average accuracy =  90.8 +- 1.8193405398660243\n",
      "Clusters 8 5 Average accuracy =  91.10000000000001 +- 1.6552945357246842\n",
      "Clusters 8 6 Average accuracy =  90.99999999999999 +- 1.8708286933869684\n",
      "Clusters 8 7 Average accuracy =  91.14999999999999 +- 1.7895530168173277\n"
     ]
    }
   ],
   "source": [
    "x_match_total = x_tune_slim[['matchDuration','totalDistance']]\n",
    "x_heal_kill = x_tune_slim[['healItems','killsAssist']]\n",
    "x_items_distance = x_tune_slim[['totalItems','totalDistance']]\n",
    "\n",
    "# X1 = df_imputed[['Pclass','Fare']]\n",
    "# X2 = df_imputed[['Age','Parch','SibSp']]\n",
    "\n",
    "params_k = []\n",
    "\n",
    "# we will vary parameters on match-total with # clusters 4 - 9\n",
    "for x_match in range(4,9):\n",
    "   for x_heal in range(3,8):\n",
    "        # get the first clustering\n",
    "        cls_match_total = KMeans(n_clusters=x_match, init='k-means++',random_state=17)\n",
    "        cls_match_total.fit(x_match_total)\n",
    "        newfeature_match_total = cls_match_total.labels_ # the labels from kmeans clustering\n",
    "\n",
    "        # # append on the second clustering\n",
    "        cls_heal_kill = KMeans(n_clusters=x_heal, init='k-means++',random_state=17)\n",
    "        cls_heal_kill.fit(x_heal_kill)\n",
    "        newfeature_heal_kill = cls_heal_kill.labels_ # the labels from kmeans clustering\n",
    "\n",
    "        y = y_tune_slim.loc[:, ('quart_binary')]\n",
    "        X = x_tune_slim.loc[:, cols_df]\n",
    "        # X = np.column_stack((X,pd.get_dummies(newfeature_match_total)))\n",
    "        X = np.column_stack((X,pd.get_dummies(newfeature_match_total),pd.get_dummies(newfeature_heal_kill)))\n",
    "\n",
    "        acc = cross_val_score(clf,X,y=y,cv=cv)\n",
    "        # params.append((x_match,acc.mean()*100,acc.std()*100)) # save state\n",
    "        params_k.append((x_match,x_heal,acc.mean()*100,acc.std()*100)) # save state\n",
    "\n",
    "        # print (\"Clusters\",x_match,\"Average accuracy = \", acc.mean()*100, \"+-\", acc.std()*100)\n",
    "        print (\"Clusters\",x_match,x_heal,\"Average accuracy = \", acc.mean()*100, \"+-\", acc.std()*100)\n",
    "\n",
    "# reference: 09.Clustering and discretization.ipynb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Creating multiple clustering scenarios doesn't produce apparent benefit. These clusters are slightly tighter in terms of accuracy results. We appear to have two peaks in accuracy—one low at 4 & 3 clusters and one high at 7 & 3 clusters."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### AGGLOMERATIVE CLUSTERS\n",
    "\n",
    "Jump to [Top](#TOP)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "params = []\n",
    "difference = 100\n",
    "\n",
    "def avg(list):\n",
    "    avg = sum(list)/len(list)\n",
    "    return avg\n",
    "\n",
    "for aff in ['euclidean', 'cosine']:\n",
    "    for link in ['ward', 'complete', 'average']:\n",
    "        for n in range(1,10):\n",
    "            try:\n",
    "                cls_hac = AgglomerativeClustering(n_clusters=n, affinity= aff, linkage=link)\n",
    "                cls_hac.fit(x_tune_slim)\n",
    "                newfeature = cls_hac.labels_\n",
    "\n",
    "                x=np.column_stack((x_tune_slim, pd.get_dummies(newfeature)))\n",
    "\n",
    "                acc = cross_val_score(clf,x,y=y_tune_slim['quart_binary'], cv=cv)\n",
    "                params.append((n,aff,link,acc.mean()*100,acc.std()*100))\n",
    "\n",
    "                # print(\"C=\",n,aff,link,\"Average accuracy = \", acc.mean()*100, \"+-\", acc.std()*100)\n",
    "            except:\n",
    "                continue\n",
    "\n",
    "hac_result = pd.DataFrame(params, columns=['n', 'affinity', 'linkage', 'mean', 'standard_deviation'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "hac_result['difference'] = abs((hac_result['mean'] + hac_result['standard_deviation']) - (hac_result['mean'] - hac_result['standard_deviation']))\n",
    "# hac_result[hac_result.difference == hac_result.difference.min()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>n</th>\n",
       "      <th>affinity</th>\n",
       "      <th>linkage</th>\n",
       "      <th>mean</th>\n",
       "      <th>standard_deviation</th>\n",
       "      <th>difference</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>2</td>\n",
       "      <td>cosine</td>\n",
       "      <td>average</td>\n",
       "      <td>90.40</td>\n",
       "      <td>1.545962</td>\n",
       "      <td>3.091925</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>6</td>\n",
       "      <td>euclidean</td>\n",
       "      <td>complete</td>\n",
       "      <td>90.90</td>\n",
       "      <td>1.562050</td>\n",
       "      <td>3.124100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>4</td>\n",
       "      <td>cosine</td>\n",
       "      <td>complete</td>\n",
       "      <td>90.85</td>\n",
       "      <td>1.566046</td>\n",
       "      <td>3.132092</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>euclidean</td>\n",
       "      <td>ward</td>\n",
       "      <td>91.10</td>\n",
       "      <td>1.593738</td>\n",
       "      <td>3.187475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>44</th>\n",
       "      <td>9</td>\n",
       "      <td>cosine</td>\n",
       "      <td>average</td>\n",
       "      <td>90.45</td>\n",
       "      <td>1.634778</td>\n",
       "      <td>3.269557</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    n   affinity   linkage   mean  standard_deviation  difference\n",
       "37  2     cosine   average  90.40            1.545962    3.091925\n",
       "14  6  euclidean  complete  90.90            1.562050    3.124100\n",
       "30  4     cosine  complete  90.85            1.566046    3.132092\n",
       "3   4  euclidean      ward  91.10            1.593738    3.187475\n",
       "44  9     cosine   average  90.45            1.634778    3.269557"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hac_result.sort_values(by='difference', ascending = True).head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='ME2'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Modeling and Evaluation 2**\n",
    "\n",
    "Jump to [Top](#TOP)\n",
    "\n",
    "*Assignment: Evaluate and compare models.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**T-Test to compare best accuracy record with best standard deviation record.** <br/>\n",
    "This demonstrates that the difference in these scores is not of statistical significance. (We can use any one for our model.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>cluster_x_match</th>\n",
       "      <th>cluster_x_heal</th>\n",
       "      <th>avg_accuracy</th>\n",
       "      <th>avg_stdev</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>91.40</td>\n",
       "      <td>1.496663</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>91.10</td>\n",
       "      <td>1.513275</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>91.25</td>\n",
       "      <td>1.600781</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>6</td>\n",
       "      <td>91.05</td>\n",
       "      <td>1.603901</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>7</td>\n",
       "      <td>91.05</td>\n",
       "      <td>1.650000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   cluster_x_match  cluster_x_heal  avg_accuracy  avg_stdev\n",
       "0                4               3         91.40   1.496663\n",
       "1                4               4         91.10   1.513275\n",
       "2                4               5         91.25   1.600781\n",
       "3                4               6         91.05   1.603901\n",
       "4                4               7         91.05   1.650000"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "column_names = ['cluster_x_match','cluster_x_heal','avg_accuracy','avg_stdev']\n",
    "params_df = pd.DataFrame(params_k, columns = column_names)\n",
    "\n",
    "#print(params)\n",
    "params_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    cluster_x_match  cluster_x_heal  avg_accuracy  avg_stdev\n",
      "15                7               3          91.4   1.813836\n"
     ]
    }
   ],
   "source": [
    "# look at that record in params_df\n",
    "print(params_df[params_df.avg_accuracy == params_df.avg_accuracy.max()])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Our best accuracy is shown here (91.4%) when we have 7 clusters for `matchDuration`/`totalDistance` and 3 clusters for `healItems`/`killsAssist`.."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   cluster_x_match  cluster_x_heal  avg_accuracy  avg_stdev\n",
      "5                5               3         91.05   1.490805\n"
     ]
    }
   ],
   "source": [
    "# MOVE TO MODEL-EVAL-2\n",
    "# look at that record in params_df\n",
    "print(params_df[params_df.avg_stdev == params_df.avg_stdev.min()])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Our best standard deviation is (1.49%) when we have 4 clusters for `matchDuration`/`totalDistance` and 5 clusters for `healItems`/`killsAssist`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "# MOVE TO MODEL-EVAL-2\n",
    "# get standard deviation scores to compare using T-test\n",
    "a_stdev_stg = params_df[params_df.avg_stdev == params_df.avg_stdev.min()]\n",
    "a_avg = a_stdev_stg['avg_accuracy']\n",
    "a_stdev = a_stdev_stg['avg_stdev']\n",
    "\n",
    "b_stdev_stg = params_df[params_df.avg_accuracy == params_df.avg_accuracy.max()]\n",
    "b_avg = b_stdev_stg['avg_accuracy']\n",
    "b_stdev = b_stdev_stg['avg_stdev']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a_stdev not different from b_stdev\n"
     ]
    }
   ],
   "source": [
    "from scipy import stats\n",
    "t_check=stats.ttest_ind_from_stats(mean1=a_avg, std1=a_stdev, nobs1=10,\n",
    "                                   mean2=b_avg, std2=b_stdev, nobs2=10)\n",
    "alpha=0.05\n",
    "\n",
    "if(t_check[1]<alpha):\n",
    "    print('a_stdev different from b_stdev')\n",
    "else: \n",
    "    print('a_stdev not different from b_stdev')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The **t-test** suggests that there is not a statistically significant difference between the accuracy with `cluster_x_match`=7/`cluster_x_heal`=3 as compared to `cluster_x_match`=5/`cluster_x_heal`=3. This is as expected; we can't *tease* out any additional information with clustering with KMeans."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='ME3'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Modeling and Evaluation 3**\n",
    "\n",
    "Jump to [Top](#TOP)\n",
    "\n",
    "*Assignment: Visualize results of model comparisons.*\n",
    "\n",
    "SUGGESTED ACTION: \n",
    "\n",
    "Make subsections: \n",
    "* (1) silhouette plot for KMEANS\n",
    "* (2) silhouette plot for HAC\n",
    "* (3) accuracy for KMEANS on TEST data\n",
    "* (4) accuracy for HAC on TEST data. (if takes long time to run, sample 5000 records from TEST)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Evaluation of clusters = silhouette coefficient"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The *\"silhouette coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1. 1: Means clusters are well apart from each other and clearly distinguished.\"*\n",
    "\n",
    "*reference: https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "# do it for the k-means\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn import metrics\n",
    "from sklearn.metrics import silhouette_score\n",
    "\n",
    "seuclid = []\n",
    "scosine = []\n",
    "\n",
    "k = range(2,10)\n",
    "for i in k:\n",
    "    kmeans_model = KMeans(n_clusters=i, init=\"k-means++\").fit(x_tune_slim)\n",
    "    labels = kmeans_model.labels_\n",
    "    seuclid.append(metrics.silhouette_score(x_tune_slim, labels, metric='euclidean'))\n",
    "    scosine.append(metrics.silhouette_score(x_tune_slim, labels, metric='cosine'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "plt.plot(k,seuclid,label='euclidean')\n",
    "plt.plot(k,scosine,label='cosine')\n",
    "plt.ylabel(\"Silhouette\")\n",
    "plt.xlabel(\"Cluster\")\n",
    "plt.title(\"Silhouette vs Cluster Size\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Hierarchical Agglomerative Clustering - Visualizations\n",
    "\n",
    "Jump to [top](#TOP)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "hc = AgglomerativeClustering(n_clusters=2, affinity = 'cosine', linkage = 'average')\n",
    "hc.fit(x_tune_slim)\n",
    "hac_labels = hc.labels_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'HAC (3 clusters, Euclidean, Average)')"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAe0AAAFlCAYAAADGV7BOAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAACL3ElEQVR4nOzdd3gU1dfA8e+mhxRC6L333os0KdI7KCACgv5EQeBVUVQURMACNrA3xAIiLSBNaugdQu899JpK6s77x03b7Oxm0zbZcD7Pk0d3Znfmzibs2bnlHIOmaRpCCCGEyPWccroBQgghhLCNBG0hhBDCQUjQFkIIIRyEBG0hhBDCQUjQFkIIIRyEBG0hhBDCQUjQFrlO1apVuX//vsm2pUuX8tJLL5ls27RpE1WrVmX16tVmx7h16xYTJ06kR48e9OzZkwEDBrBhwwaL5wwPD2fkyJFERUURFhbG2LFj6d69O127duXHH39MV/snTpzIL7/8kq7XpDRixAiz688uVatWpUePHvTq1cvkJzg4OEPHmzNnDlOnTgXgxRdf5Ny5c2bPWbt2Lc8991ym2p1R9+/fp06dOkyePDlHzp8eH3/8MXv27MnpZohcxiWnGyBERs2fP58ePXrw22+/0bVr16Tt9+/fZ+DAgYwbN46PPvoIg8HAqVOneP755/H09OSJJ54wO9asWbMYMGAAHh4ezJo1i6JFizJ79mwiIyPp3r07jRs3pn79+na5rh07dtjlPInmzZuHv79/lh/3p59+yvJjZtbixYtp3749K1eu5P/+7//w8/PL6SZZNHr0aAYPHsyiRYvw8PDI6eaIXELutIVDunr1Knv37uXtt9/m8uXLBAUFJe2bP38+DRo0oHfv3hgMBgCqVavG7NmzKVSokNmxbty4webNm+nQoQMA7777Lm+99RYAd+7cISYmBh8fH7PXRURE8Pbbb9OpUye6du3K559/TupcRal7DRIfR0REMHbsWHr16kWfPn2YNGkSRqORt99+G4Bhw4Zx48YNbt26xejRo+nbty89evTg+++/ByA4OJg2bdowYsQIOnXqxI0bN5g8eTI9evSgb9++jB07loiIiEy8w7Bnzx66d++u+zguLo6PPvoo6drfffddYmJiTF7frl07jh49CsBXX31Fhw4d6N+/P+vXr096TkxMDDNmzKBPnz707NmTiRMnEh4eDsDmzZsZOHAgffv2pW3btnz55ZdJ7Rg4cCATJkygd+/edO/enQMHDqR5PUajkYULF9KnTx8aNWrEP//8A0BYWBgNGjTgzp07Sc8dMGAAW7Zssdq+du3aMX78eLp06cL69estthfgxx9/5KmnnqJPnz5Mnz6ddu3apXn9Pj4+1K9fn4ULF6b9yxKPDQnaIlcaNmyYSXft7NmzTfYvWLCAtm3bUrBgQbp27cpvv/2WtO/YsWM0aNDA7JiNGzematWqZts3btxI8+bNcXFRHU8GgwEXFxfeeOMNunfvTpMmTShfvrzZ62bPnk10dDSrV68mICCAgwcPsnfvXpuub/369URERLB8+XIWL14MqC8iH330EaDufosXL86ECRPo168fS5cuZfHixezcuTNpOODmzZu88sor/Pfff1y7do29e/eyYsUKli5dSunSpTl9+rRNbUn9Xo8ePTrN18yfP5/jx4+zfPlyVq5cSUREhO4wBcCGDRtYt24dAQEB/P3330lBCVQwc3Z2ZunSpaxYsYIiRYowa9YsNE3j119/5eOPP2bp0qUsXLiQH3/8MekL0JEjRxgxYgQBAQH07duXL774Is02b9u2jaioKFq0aEHv3r35888/iYuLw8fHh44dO7JixQoAzp8/z927d2nVqpXF9iWqXLkya9asoUOHDhbbu23btqTf39KlS02+TKV1/JYtW5p8yRFCusdFrpS6y3bp0qX8999/gLo7Wbp0KTNmzACgT58+DBo0iBs3blC8eHEMBoPZHa81Fy5coEyZMmbbZ82axQcffMDYsWP55ptvGDt2rMn+nTt38vbbb+Ps7IyzszN//vknAMuWLUvznA0bNuSLL77gueeeo0WLFgwbNoyyZcuaPCcyMpJ9+/YREhLCV199lbTt1KlT1KlTBxcXF+rVqwdAlSpVcHZ2ZsCAAbRs2ZJOnTpRp04dm64/I93jO3fupFevXkndtol3lXPmzDF77q5du+jYsSPe3t4A9OvXjz/++AOAwMBAwsLC2LlzJwCxsbEULFgQg8HA999/T2BgICtXruT8+fNomsajR48AKFGiBNWrVwegRo0aNr3nCxYsoEePHri4uNC+fXsmT57M2rVr6d69OwMGDOCDDz5g5MiRLFmyhH79+uHk5GSxfYkaNWoEYLW9W7ZsoXPnzvj6+gLw7LPPsnv3bqvXn6hUqVJc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4JxuRpbx8vIC1O9q7NixjB8/PmcblAVcXFx46623WL9+PbNnz87p5mTK0qVL8ff3p1WrVvz444853ZxM8/DwYOTIkQwYMIBLly7x4osvsnbtWof9THjw4AHXr1/n+++/Jzg4mJdffpm1a9diMBhyummZ9sMPPzB69Oicbka6yZ12Jnh7exMREZH02Gg0Ouw/zrzsxo0bDB06lF69etGjR4+cbk6W+OSTT/jvv/947733iIyMzOnmZNiSJUvYuXMnzz33HCdPnuStt97izp07Od2sDCtfvjw9e/bEYDBQvnx5/Pz8HPp6/Pz8aNmyJW5ublSoUAF3d3fu37+f083KtNDQUC5cuECzZs1yuinpJkE7Exo0aMDWrVsBCAoKokqVKjncIpHa3bt3GTFiBBMmTKB///453ZxMCwgI4IcffgDA09MTg8Hg0F3+f/31F3/++Sd//PEH1atX55NPPqFw4cI53awMW7x4MR9//DEAt27dIjw83KGvp2HDhmzbtg1N07h16xaPHj3Cz88vp5uVafv27aNFixY53YwMkdvCTOjYsSM7duxg4MCBaJrGjBkzcrpJIpXvv/+e0NBQvv32W7799ltATbZz1ElPTz31FG+//TbPPvsscXFxvPPOO7i7u+d0s0SC/v378/bbbzNo0CAMBgMzZsxw6N63J598kn379tG/f380TeP999936C+JiS5evEipUqVyuhkZIlW+hBBCCAch3eNCCCGEg5CgLYQQQjgICdpCCCGEg5CgLYQQQjgICdpCCCGEg5CgLYQDWbBgAQsWLDDbvnTpUiZOnGj1tRMnTqRt27ZJSWb69OnD6tWrk/a/++67HD161OLrZ8+ezf79+zPeeCFEpjnuAkIhHkODBg3K1OvHjh2bVCzh6tWrDB48GD8/P1q0aMH06dOtvnbfvn00bdo0U+cXQmSOBG0hcrk9e/Ywc+ZMjEYjlStXplSpUrz66qsEBATw3Xff4e3tTcmSJcmXLx8AR44c4aOPPiIqKooCBQrwwQcfULp0abPjli5dmqFDhzJ//nxatGjBc889x5gxYyhbtixvvPEGkZGRODk5MWnSJC5dusSxY8eYNGkSX3/9NSEhIXzxxRdERUURGhrK22+/TYcOHZg4cSLe3t4cP36cW7duMXr0aPr168fDhw959913uXDhAm5ubkycOJHmzZuzdetWZs+eTVxcHKVKleLDDz+kQIEC9n6LhXAY0j0uhAO4dOkS8+bNS8ridOvWLWbNmsVff/3FwoULk3Lgx8TEMGnSJD777DOWLVvG888/z3vvvWfxuFWqVOHChQsm2xYvXkzbtm1ZunQpY8eO5cCBA/Tu3ZtatWoxbdo0qlatyp9//sm0adNYtmwZ06ZN46uvvkp6/c2bN5k/fz7fffcdn376KQBfffUVZcqUYc2aNXz66ad8+eWX3L9/n88++4xffvmFgIAAWrZsyaxZs7L6rRMiT5E7bSEcQPny5fHx8Ul6fOjQIerXr59UmrNHjx7s3r2bS5cucfXqVV5++eWk54aHh1s9duqUrs2bN+fVV1/l5MmTtGnThiFDhpi9ZubMmWzevJm1a9dy+PBhk8I5TzzxBAaDgSpVqvDw4UNAda0nBuSqVauycOFCNm/enFTMBVTBnfz586fjXRHi8SNBWwgHkDqwGgwGUmYgTsxvbTQaKVWqFMuXLwcgPj6eu3fvWjzu6dOnqVixosm2hg0bsmrVKgIDA1m9ejXLli1j7ty5Js8ZPHgwTZs2pWnTpjRv3pw33ngjaV9iLvSU5RtdXFxMHp8/f574+HgaNGjA999/D0B0dLRJ8BdCmJPucSEcUMOGDQkKCuLWrVsYjcakWeAVKlQgJCQkaZb3kiVLTAJqSpcuXWL+/Plmk9s+/fRTVqxYQZ8+fXj//fc5ceIEoOrFx8fH8/DhQy5dusS4ceNo3bo1GzduJD4+3mp7GzVqxKpVqwAVsF988UXq1KlDUFAQFy9eBODbb79N6k4XQuiTO20hHFChQoWYNGkSw4cPx9PTk0qVKgHg5ubGV199xfTp04mOjsbb25tPPvkk6XWzZ89m3rx5SSU933rrLRo0+P/27hBVQigKwPAB7xp0H5MsrkAwWd2CCFbBJqh7cX0mmfbglWkT7vB9/Z5wyw+nnNe/2cMwxDzPcV1XFEXx975pmljXNfZ9j77vo23bSClFXddx3/fHu97jOMayLNF1XaSU4jiOKMsytm2LaZrieZ6oqirO8/zCb8HvcOULADJhPQ4AmRBtAMiEaANAJkQbADIh2gCQCdEGgEyINgBkQrQBIBNvfrI9pfUeTvYAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x_tune_slim['rideDistance'],x_tune_slim['matchDuration'], c=hc.labels_, cmap = plt.cm.rainbow)\n",
    "plt.xlabel('rideDistance')\n",
    "plt.ylabel('matchDuration')\n",
    "plt.grid(c='black',linewidth=0.5)\n",
    "ax=plt.gca()\n",
    "ax.set_facecolor('white')\n",
    "plt.title('HAC (3 clusters, Euclidean, Average)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Based on the scatter plot we created, we can determine that the clustering method doesn't produce a high degree of separation within our dataset. It seems that the clustering is very random since there isn't a clearly defined linear separation between our red and purple classes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAl8AAAFKCAYAAAAjTDqoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAABEQElEQVR4nO3dd3hUZf7+8feZmkwmmQRI6EkIXVARrKvoqrAqoiiIoAhWVFZ/6FeWFREUpaqsuouKZWVFdEVFRHEtK4plWUVhRUCK9BKKAdLb1N8fE4aEwCSUmQnhfl1Xrpx+PvMYyZ3nOcUIBAIBRERERCQqTLEuQERERORkovAlIiIiEkUKXyIiIiJRpPAlIiIiEkUKXyIiIiJRpPAlIiIiEkUKXyInoWXLljF48GCuuuoqevfuzR133MG6desAWLFiBcOHDwdg1KhRvPrqqwC0b9+effv2RaW+2267LXSud999lzfffDMq561s4cKFDB48mD59+nDllVdy//33s3PnTgDmzp3LXXfdddTHfu6551iwYMEx1zh37lz69u3L1VdfzZVXXsnDDz9MYWEhAG+99RYvv/zyMZ9DRI4/S6wLEJHocrvd3HXXXcyYMYNOnToB8MEHHzB06FC++OILTj31VP72t7/FtMZFixaFppcuXUrbtm2jev758+czffp0pk+fTkZGBoFAgJdffpkhQ4bwr3/965iPv3jxYtq0aXNMx1i+fDnPP/887733HsnJyfh8Ph577DHGjRvHX/7yF2644YZjrlNEIkPhS+QkU1paSmFhISUlJaFlV199NU6nE5/Px5IlSxg/fjwfffRRtX2nTZvGzz//TF5eHrfffjuDBg0C4Pnnn+df//oXZrOZVq1aMXbsWFJTUxk8eDCDBg3i8ssvB6gyv2HDBiZOnEheXh4+n4/Bgwdz3XXX8dBDDwFw8803c/vtt/Pll1+yaNEi4uLiGDRoENOnT+ff//43fr+f5s2b8+ijj9K4ceMqdQ4cOJBbb72Vyy67DICnnnoKgFtuuYUHH3yQ3NxcAC666CLuv//+ap/zmWeeYfz48WRkZABgGAZ33nknTZs2xe12V9k23Gf829/+xueff47VaiUlJYXJkyfz+eefs3LlSp588knMZjMXXXQRU6dO5ccff8Tn83HKKacwZswYnE4nl1xyCaeddhpr167lgQceoGfPnqHz5uTkEAgEKCsrA8BsNnPfffeFejCnTZtGbm4uQ4cO5e677w7tt2fPHiwWC19//TW7d+/m8ccfZ+fOnXg8Hq688soq24pIZCh8iZxkXC4XI0eO5I477qBRo0Z07dqVc845hyuvvBKbzRZ235YtW/Loo4+yatUqBgwYwPXXX8+HH37It99+y5w5c3A4HEybNq3KcOWheL1ehg8fzpNPPkmnTp0oLCxkwIABtGnThsmTJzN37lxmzpxJgwYN+P7772nbti2DBg1i3rx5/Prrr7z77rtYLBbefvttxowZwyuvvFLl+P3792fu3Llcdtll+Hw+PvzwQ2bNmsU777xDixYtmDFjBiUlJaFhusTExNC+ubm5ZGdn07Vr1yrHNAyDq6++utbtvHPnTmbOnMl3332HzWZjxowZLF++nEGDBvHpp58yaNAgevbsyXPPPYfZbGbu3LkYhsHTTz/N1KlTGTduHABt27bl2WefrXb8Cy+8kI8//phLLrmE9u3bc8YZZ3DhhRdy0UUXVdmuadOmfPDBBwBs27aNm2++mSeeeAKAkSNHcsstt3DJJZdQXl7O0KFDSU9Pp1evXrX+nCJy5BS+RE5Ct956K/379+fHH3/kxx9/5JVXXuGVV15hzpw5Yffr3bs3AB07dsTtdlNUVMQ333xD3759cTgcAAwZMoQXX3yxWg9RZZs3b2br1q2MHj06tKysrIxVq1bRpUuXw+63cOFCVqxYQb9+/QDw+/2UlpZW265Xr148+eST5OTksGrVKjIzM8nMzKR79+7ceeed7Ny5k9/97neMGDGiSvACMJlMoWMfi8aNG9OhQweuvfZaLrzwQi688ELOO++8att99dVXFBYW8t///hcAj8dDw4YNQ+vPPPPMQx7farXyl7/8hT//+c8sXryYH3/8kQcffJDzzjvvkGFt3759DB06lAceeICzzjqLkpISfvzxR/Lz8/nrX/8KQElJCWvWrFH4EokwhS+Rk8zSpUv56aefuOOOO7j44ou5+OKLeeCBB+jduzeLFi0iJSXlsPtaLMF/MgzDACAQCOD3+0PzEAwtXq83NF/59bEejwcAn89HYmJiqEcGgsNhBwehg/n9fu644w5uvPFGIHj9Wn5+frXt4uPjueyyy/joo4/46aef6N+/PwCnnXYaX3zxBd999x3ff/89/fv355VXXqFz586hfV0uF5mZmfz888/87ne/q3Lc++67j2HDhlU736E+o8lk4o033mDFihV89913TJo0ie7du/PnP/+52mcaPXp0qMequLiY8vLy0Pr9ofZgc+bMISUlhUsvvZSrr76aq6++mmHDhnHJJZdUuzGitLSUu+++m2uvvTYUoP1+P4FAgNmzZxMfHw8EA5rdbj/k+UTk+NHdjiInmQYNGjB9+nSWLFkSWpaTk0NRURHt2rU74uN1796d9957L3QN2axZszjrrLOw2Ww0aNCAlStXArB+/XrWrl0LQKtWrYiLiwuFr507d9K7d+/QtmazORTgKk9fcMEFzJkzh6KiIgD++te/Vgsz+11//fW8//77/O9//wtd+zV16lReeOEFevTowcMPP0ybNm1C10hVdu+99zJx4kS2bNkCBMPiCy+8wJo1a8jKyqqy7eE+45o1a+jduzetW7fmrrvu4pZbbmHFihWH/Exvvvkmbrcbv9/P2LFjefrpp2tsd5PJxNSpU9m1a1do2bp162jWrBkulyu0zOfzcf/999OhQ4cqd2g6nU66dOnCP/7xDwAKCgq44YYb+OKLL2o8t4gcG/V8iZxkWrVqxfPPP88zzzzDrl27sNvtJCYmMmnSJLKyssjJyTmi41133XXs3LmT/v374/f7ycjIYOrUqQAMGzaMUaNG8fXXX5OVlRUaQrPZbLzwwgtMnDiRv//973i9Xu677z66desGwOWXX87gwYOZNm0aF154IVOmTAFg6NCh7N69m+uvvx7DMGjatGlo3cE6d+6M2Wzm8ssvD/Xm3HzzzYwaNYrevXtjs9lo3749V155ZbV9r7rqKgKBAA888ABer5fy8nI6derEzJkzq10Xd7jP2KFDB6644gr69euHw+EgLi6OMWPGAHDJJZfw9NNP4/F4+OMf/8gTTzzBtddei8/no2PHjowaNarGdu/bty+lpaUMHToUt9uNYRhkZmby6quvYjabQ9t98sknfPXVV3Tu3Jlrrrkm1Ev38ssvM3XqVMaPH89VV12F2+2md+/eR3Rdm4gcHSNQub9cRERERCJKw44iIiIiUaTwJSIiIhJFCl8iIiIiUaTwJSIiIhJFCl8iIiIiUXTCPGoiJ6cwKudJSXGQm1tS84YnKbVPeGqfmqmNwlP71ExtFJ7ap2bRaKPU1MM/NFo9XwexWMw1b3QSU/uEp/apmdooPLVPzdRG4al9ahbrNlL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERE54f3vf0t49NGHABg9emS19fPmzeHVV1+KdlmHpPAlIiIi9cqkSU/FuoSwTpjXC0Wax+vny/9tp0GKgzizQWpyPA2T4rBalE9FRESiwev18tRTk9i+fRt+v5+hQ4cxadJjvPnmHOx2O9OnTyMjI5PLL7+SZ599itWrf8Hj8XL77XeSkOAMHefqqy/jww8/4+efl/HXv04lKSkJk8lMp06dAZg1axbvv/8BhmFw6aV/oH//gWzcuJ5p057B7w9QVFTI/ff/iVNPPZ2BA6/l1FNPZ+vWLTRo0IAJE57EbD62J+QrfFXYnlPE21+ur7LMAJIT7TRyxdHIFU9qctXvKYl2TCYjNgWLiIhEyDtfrufHNb8d12Oe1SGN6y9pE3ab+fPn4XIl89BDj5Cfn8c999x5yO2+/fZr8vPzeOWV19m7dw/vvfcOZ555drXtpk17mnHjJpKensHUqZMB2LRpIx9//DEvvPB3DMPg/vv/yDnnnMumTRu5997/o3XrNvz735/y8cfzOfXU09mxI5u//nU6jRs3Ydiw21i9ehWdO596TG2h8FWhVdMkxt16FgXlPjZuy2VPXhl78kvJyStjfXY+67bnV9vHbDJomBQXDGPJ8TRyxZGaHE8jVzyNkuNIjLdiGApnIiIitbFhw3qWL/+JVatWAuDzecnPP/D7NxAIALB16xY6dToNgIYNG3HnnX/kf/9bUu14OTm/kZ6eAcCpp57O9u3b2LhxAzt27OC++4YBUFhYyPbt22nUKI3XXvs7drudkpISEhISAHC5kmncuAkAaWmNcbvLj/lzKnxVkt44kdTURDqnJ1dZ7vX52VdYzp68Uvbkl5GTV0pOxfSevFJ+2ZwL5FY7nt1qplFyHKmuYDBrlBzsNUutCGdxNjW/iIjUPddf0qbGXqpIyMjIJC0tjSFDbqO8vIyZM2fw5ZcL2Lt3D02bNmP9+l/JzGxFZmYmCxd+AUBRURGPPDKKm266pdrxGjZsyObNm8jMbMXq1atITEwkPT2DNm3aMHnyMxiGwdtvv0lWVhtGjx7BI49MIDOzFa+++hI7d+4AiEgnin7714LFbCItOZ605PhDri93+4K9ZBVh7EBAC/aeZecUH3I/Z7w1NITZqFIoS3XF09AVh8Ws681EROTk0adPX554YgL33nsnxcVFXHttf2666WZGjryPJk2akZiYCMAFF1zEkiU/MGzY7fh8Pm69deghjzd27HgmTnwUhyMBh8NBYmIibdu247zzzuOPf7wdt9tDx46dSE1N5Q9/uIJRo0bQoEEDUlPTyM/Pi9jnNAL7+/DquJycwqicJzU18bieKxAIUFzmrdJTllPp+978Ury+6v8JDCAlyR68xsx18LBmHMmJdkwxGNI83u1T36h9aqY2Ck/tUzO1UXhqn5pFo41SUxMPu049XxFmGAbOeCvOeCutmiZVW+8PBMgvclcbyszJD/aarduWx6/bqh/XYg5eb9YoORjOUpPjqwS0hDiLrjcTERGpgxS+YsxkGKQk2klJtNOuZXK19V6fn70FwWHMPXll5OSXVrkZYPemfYc8bpzNXOXOzIOHNe22Y7tNVkRERI6OwlcdZzGbaJzioHGK45DrS8u97M0/EMr2957l5Ad70rbnFB1yvySHtdpQ5v5etAZJut5MREQkUhS+TnDxdgst0py0SHNWWxcIBCgs9VTqKas6rLllVyEbdxRU288woEGi/UCPWXL8gTs0E+zR+FgiIiL1lsJXPWYYBkkOG0kOG1nNDnG9mT9AbmF5aAiz8vc9+WX8ui2PtdsOPiY0b+SkfXoy7Vsm0y49mSSHLUqfSERE5MSn8HUSM5kMGrriaOiKo3169fUeb+XrzYK9ZTv2lrB68z625xTxxdLtADRrlED7lsmhQOZyqndMRETkcBS+5LCsFhNNGjho0uDA9WapqYns2JnPpp0FrN2Wx69bc1mXnc+OPcUs/CkbgCYNHKEg1j49hZREhTEREYmsWbNeo1u3MznllM6xLqVGCl9yxKwWE+1aJgfvzvxdJl6fny27ClmzNZe12/JYtz2fr5ft4OtlwacDpyXH0y4UxpJp5Dr0w2pFRESO1uDBt8S6hFpT+JJjZjGbaN3cRevmLq48D3x+P1t3F7F2ax5rt+by6/Z8/rN8J/9ZvhOARq640PVi7dNTSHXF6ZlkIiJCeXkZkyY9xq5du/B6vQwf/gAffjiX7OxsfD4fAwcO4tJL/8Dcue/yyScfYTKZOO20Ltxzz31MnDiOSy/9A/v27eW77xZRXl5GdvZ2Bg26mV69rmLDhvU8++xTBAIB0tIa8cADo3E6q9+sFg0KX3LcmU0mWjVNolXTJC4/Jx2/P8C234pYW9Ez9uu2PBat3MWilbsASEm0VxmmbJwSrzAmIhJDc9d/xE+/rTiuxzwj7VT6tukddpt5896jSZNmPPbYZDZuXM8333yFy5XM2LHjKSkp5rbbbqJbt7P5+OP53H//SDp3PpX335+D1+utcpzi4iKefvo5tm3byoMP/h+9el3FE09M4KGHHqFVqyy++upT3nxzJnfddc9x/Yy1pfAlEWcyGWQ0SSSjSSJ/ODsdfyBAdk5xKIyt3ZrH97/s5vtfdgPgctpCQax9y2SaNnQojImInAS2bt3Cuef+DoCsrDa8//57nHnm2QA4HAlkZrYiO3s7o0c/wltvvcGLL06jU6dTqx2nTZt2AKSlNcbtdgOwZcsm/vKXKQAYRoCmTVtE4yMdksKXRJ3JMGiZ5qRlmpMeZ7YkEAiwY28Jv27NZc3WPNZuy+OH1b/xw+rfgOADYdtVCmPNUhNi8l5LEZGTRd82vWvspYqEjIxWrF69iu7df0929nYWLPgMm83KRRddTElJMRs2bKBZs2bMnDmDP/3pIex2Ow88cC8rVvxc5TiH+oM9PT2DMWMep0mTJmzd+isbNmyN1seqJmLhy+/3M27cONauXYvNZmPChAlkZGSE1n/44Yf84x//wGQy0a9fP2688cZIlSJ1nGEYNG+UQPNGCVzctQWBQIBd+0oq7qYMhrEla3NYsjYHAGe8lbYtXHRIT6F9ejIt0pwKYyIi9UCfPn2ZPPlx7r33Tnw+H3/5y9+YO/ddhg27nfLycm67bSgpKQ1o3boNQ4cOITk5hdTUVE45pTMffzw/7LFHjHiICRMewe/3Y7WaGTFidJQ+VXVGIBAIROLA//73v/nyyy+ZMmUKy5Yt46WXXmL69Omh9RdccAEfffQRDoeDK6+8kjlz5uByuQ57vGi9oV1vgw8vFu0TCATIySsN9optzePXbbnsLSgPrXfYLaG7L9unJ5Pe2InZFJvXI+nnp2Zqo/DUPjVTG4Wn9qlZNNooNTXxsOsi1vO1dOlSunfvDkCXLl1YuXJllfXt27ensLAQi8VCIBDQNT1yWIZhkJbiIC3FwYWnNwNgT15p6HqxtdtyWbZ+D8vW7wGCLxVv1/LAE/gzGifqXZUiIlJnRCx8FRUVVbmF02w24/V6sViCp2zbti39+vUjPj6enj17kpRU/fU3IofTKDmeRsnxnH9qUwD2FZRVhLFc1m7NY/mGvSzfsBcAu9VMmxau0HPGWjVNUhgTEZGYiVj4cjqdFBcXh+b9fn8oeK1Zs4avvvqKL774AofDwciRI/nkk0+44oorDnu8lBQHFos5UuVWEa6rUOpm+6SmJtK+dWpofm9+Kb9s3MvKDXtZuXEPv2zaxy+b9gFgs5rpkJHCqW0a0TmrIe3SU7BZj9/PVl1sn7pGbRSe2qdmaqPw1D41i2UbRSx8de3alYULF9KrVy+WLVtGu3btQusSExOJi4vDbrdjNptp0KABBQUFYY+Xm1sSqVKr0Fh5eCdS+3Rs4aJjCxf9L8qioNgdfFH41jzWbMtl+fo9LK8YprSYTbRulhR61ljr5q6jDmMnUvvEitooPLVPzdRG4al9alZvr/nq2bMnixYtYuDAgQQCASZNmsT8+fMpKSlhwIABDBgwgBtvvBGr1Up6ejrXXnttpEoRISnBxpkd0jizQxoAhSVuft2Wz9ptufy6Nfjg17Xb8gAwmwxaNUuifctkOqSn0Ka5C7stOr2uIiJS/0XsbsfjTXc71g31tX2Kyzys25Yfej/l1t2F7P8/w2wyyGySWPF+yhTatnARbz/03y31tX2OJ7VReGqfmqmNwlP71Kze9nyJnEgS4qx0aduILm0bAVBS5mV99v67KfPYtLOQDTsK+OT7rRgGZDRODA5TpqfQroULR5w1xp9AREROFApfIofgiLNwWutGnNY6GMbK3F7WZ+cfCGM7Cti8q5DPftiGAbRs7KR9yxS6ndIET7kHs8nAZDIwm43gtGFgNpswm4wqXyaTgdlUsdwcnNcDY0VE6jeFL5FaiLNZ6NyqIZ1bNQSg3ONjQ6UwtnFHPlt3F/H5km3HfC4DQkEsFMxCQe3woc1y8PJD7WM2YTaMSsc/eDtT1eBYZdsD6ywH72PeHzDDHK/i6wS50kFEJGIUvkSOgt1q5pTMBpyS2QAAj9fHxh0F/FbgJi+/BJ8/EPryV5n24/MF8AUC+HyHWFdp3uev2CYQwOcLrvMHAnjc/oOOHZw/UTJNZtMkBlzcmvbpKbEuRUQkJhS+RI4Dq8VM+/QULojhha7+QKUwd1Bo2x/UvBXfg9P+4PbVwqC/WmgMzfv8hwmOlY53cA0V+/n9Aco9PtZuy+OJf/7E+ac2of/FbUhy2GLSXiIisaLwJVJPmAwDk9nAYgbq8PX/e0s8TJv9E4tW7GLZuj1c9/vWdD+9ma51E5GTht6xIiJR1SGjAWNvOZMbLm2Lzx9g5qdrmfzGUrbu1q3xInJyUPgSkagzm0z0PKslE4eey1kd0tiQXcDjry1h9hfrKC33xro8EZGIUvgSkZhJSbQz7JrOPHD96TRyxfHvH7cx5u+LWbLmN90VKSL1lsKXiMRc56yGjL/jbK4+P5PCEjcvzFvJM+/+zG9ReqeriEg0KXyJSJ1gtZi5pnsWj99+DqdkprBy4z7GvvoDHy7ahMfrj3V5IiLHjcKXiNQpTRo4GDGgC3f36YQjzsK8bzfxyIwfWLV5X6xLExE5LhS+RKTOMQyDszs2ZuId59KjWwt+yy1h6uxlvPThL+QXlce6PBGRY6LnfIlIneWIs3Bjz3acf2pTXv9sDYtX7Wb5hj30vbA1F5/RHJNJzwYTkROPer5EpM7LaJLIw4PPZPAf2gEGb37+K+NfX8KmnQWxLk1E5IgpfInICcFkMri4awsm3Xku53VqzJZdhUyYuYRZ/15LSZkn1uWJiNSawpeInFBcCTaGXtWJkTecQZOGDhb+L5vRryzmu1926dlgInJCUPgSkRNSx4wUHrvtbPpemEVpuZdX5q9i6uxl7NxbHOvSRETCUvgSkROWxWyi9+8ymXDHOZzWuiGrt+TyyKs/MPebDbg9vliXJyJySApfInLCS02O577rTuOea08lKcHGR//dwpi/L2b5hj2xLk1EpBqFLxGpFwzDoFv7VCYOPYfLz05nX0E5z767nOffX8G+grJYlyciEqLnfIlIvRJns3D9JW34XecmvP7vtSxdm8PKjfu4pnsrepzZArNJf3OKSGzpXyERqZdapDkZNagrt17RAavFxNtfruexfyxh/fb8WJcmIic5hS8RqbdMhkH305sxceg5dD+tKdtzipj0xlJe+2Q1RaV6NpiIxIbCl4jUe4kOG7f26shDN3WlRWoC3/y8k9Evf8+3y3fg17PBRCTKFL5E5KTRtkUyj9xyFtdf3AaP188/Pl7DE2/+j+05RbEuTUROIgpfInJSsZhNXH5OOhOHnkO3dqms257PuBk/8s7C9ZS5vbEuT0ROAgpfInJSapAUxz19T+X+/qfRIMnOp4u3Mubvi/nfrzl6TZGIRJTCl4ic1E5r3Yjxd5zDledlkF/k5rm5K/jbnOXsySuNdWkiUk/pOV8ictKzW830u6g153Vqwhv/XsvPG/ayestirjo/k8vOTsdi1t+pInL86F8UEZEKzRolMPKGMxh61SnE2cy89/VGHp3xA2u25Ma6NBGpRxS+REQqMQyD8zo1YeKd53LxGc3ZtbeEJ9/6iVfmr6Kg2B3r8kSkHtCwo4jIISTEWRl8WXvOP7Upsz5by3e/7OLn9Xvo9/vWXNSlGSbDiHWJInKCUs+XiEgYWc2SGHvzmdzYoy0BAsz6bC0TX1/Kll2FsS5NRE5QCl8iIjUwmQx6nNmSiUPP5ZxTGrNpZwGPz/yRf37+K6XlejaYiBwZhS8RkVpKdtq56+pOjBjYhbTkeBYs3c7oV77nh9W79WwwEak1hS8RkSPUKbMBj99+Ntdc0IriUi8vfvALT7/zM7v3lcS6NBE5ASh8iYgcBavFzNUXtGL8HWfTuVUDftm0j7Gv/sC8bzfi8fpiXZ6I1GEKXyIix6BxioP/u/50hl3TGWe8hQ8XbWbsqz+wctPeWJcmInWUwpeIyDEyDIOzOqQxcei59DyzJTl5pTz99s9Mn7eS3MLyWJcnInWMnvMlInKcxNst3NCjLeef2oRZn63lxzW/sWLjXq7tnsUl3ZpjNunvXRFRz5eIyHGX3jiRhwZ3Y8jl7TGbDN76Yh3jZy5hw478WJcmInWAwpeISASYDIPfd2nOxKHncn7nJmzdXcSk15fy+qdrKC7zxLo8EYkhhS8RkQhKSrBxe+9TePDGM2jaKIGvlu1g9Mvfs2jFTj0bTOQkFbHw5ff7eeSRRxgwYACDBw9my5YtVdYvX76cG2+8kRtuuIHhw4dTXq6LUkWk/mqfnsK4W8/iut+3ptzt49V/reapt35ix57iWJcmIlEWsfC1YMEC3G43b7/9NiNGjGDKlCmhdYFAgLFjxzJ58mTeeustunfvTnZ2dqRKERGpEyxmE73OzWDC0HPo0qYRa7bm8eiMH3jv6w2Ue/RsMJGTRcTC19KlS+nevTsAXbp0YeXKlaF1mzZtIjk5mZkzZ3LTTTeRl5dHVlZWpEoREalTGrniGX7dafy/fqeS7LTxr++2MOaVxSxbvyfWpYlIFETsURNFRUU4nc7QvNlsxuv1YrFYyM3N5aeffmLs2LFkZGRw991307lzZ84777zDHi8lxYHFYo5UuVWkpiZG5TwnKrVPeGqfmqmNgv6QmsiF3dKZ/fla5n29gb/NWc6ilbvIaJKE1WrCbjVjtZixWU3YKr5bLebgcqsJm8WEzWrGZjVjtezfPrjMYq7fl/TqZyg8tU/NYtlGEQtfTqeT4uID1zL4/X4sluDpkpOTycjIoE2bNgB0796dlStXhg1fubnReWdaamoiOTmFUTnXiUjtE57ap2Zqo+quPCedLlkNmPXZWpau+Y2la3475mOaDAOrxVTty2YxYTXvnzdXWW4JTVcsN5uwWg9sb6u0fdVjVtreYsJkMo5DqxyefobCU/vULBptFC7cRSx8de3alYULF9KrVy+WLVtGu3btQutatmxJcXExW7ZsISMjgyVLlnDddddFqhQRkTqveaqTBwd1pTxgkL0rH6/Xj8frx13xPfjlC373+XF7gt+rLK+8vc+PJ7SNj3KPj+JST2ibSDKbjANBz2LCYjGHpo820FVeXuTx4y5144izYLeaMYzIhj2R4y1i4atnz54sWrSIgQMHEggEmDRpEvPnz6ekpIQBAwYwceJERowYQSAQ4IwzzuD3v/99pEoRETkhGIZBy7RE4iI8YhgIBPD6ApVCmi8U5A4Z9g4OdQctrxoUfQcFPz+l5V4KioPTXt/xDX5mk4EjzoLDbsERZyUhzhKcrzSdEGetWF8xXbE83m7BpOAmMWAETpAHzUSrC1XdteGpfcJT+9RMbRRefW8ffyAQDGsH9d55vX7clUKdp1LYc1cKhx6vH0wm9uaVUFLmpbjMU/HdS0mZB6+v9r/SDIKvhDo4lCVUhDeH/cB0QpyF+Mrb2S119rq6+v4zdDzU22FHERGRg5kMI3STQELc0R0j3C9Ot8cXCmIl5QdCWXGZl9KyqvMl5Qemd+0rOeLHfdit5rBhzVGtF+7ANjZrdG4gk7pJ4UtEROqN/cEuJdF+xPt6ff6KQHagR62kclgr81JSfmB6/zb7CsrJzinmSIaRLGZTlYBWuUetpiHTOJuuczvRKXyJiIgQDERJDhtJDtsR7+v3Byh1eyv1sHkOBLRQoDsoyJV5KCzxsHtfKf4juALIZBhVh0jtVXvXUhsm4Pd4ibNbiLcFw1q83UKc3Ryat1pMCnAxpPAlIiJyjEwmg4Q4Kwlx1iPeNxAIUO7xVQlohwxrB/e6lXvJLSwPXgd3hMwm40Aos5lDQS3ebg7O24I3JMRXrNu/7f7wFmc/sK/ZVDevfavLFL5ERERiyDAM4mwW4mwWGiQd+f4er69Sb5oXi93C7pwiSt1eysp9lLm9lJb7gvNuH2Xl3irr9hWUU+ou5mhvv7NZTBXhzXzg+/4gtz+4VYS5uErrDp4/mR4bovAlIiJyArNazCQ7zSQ7g9e5Hc2dfIFAALfHHwxqbh+l5V7KyoNhrbQivJVVhLfS8kPMu4PzeUXuo35PqWFQvdetUq9cnL1ScKs8HxpWPRD0rJa63Run8CUiInKSMwwDu82M3WbGdYzH8vn9lLt9VYNZuffQ825vMOiFeuR8Fc+Fc7N7nxef/+i64yzm/b2J5mrDpw67hasuakNKfOwikMKXiIiIHDdmkwlHnAnHUVz/djCP118xRBouvFWdPzAd3H5Pfill5b4qd6PGO2z0vzDrmOs7WgpfIiIiUicFXyt1dHegVuYPBHB7fJSWB1+1dUqbVPbtK655xwhR+BIREZF6zVTppgYAc4zfTlC3r0gTERERqWcUvkRERESiSOFLREREJIoUvkRERESiSOFLREREJIoUvkRERESiSOFLREREJIoUvkRERESiSOFLREREJIoUvkRERESiSOFLREREJIoUvkRERESiSOFLREREJIoUvkRERESiSOFLREREJIpqFb7cbjfTp0/nz3/+M0VFRTz33HO43e5I1yYiIiJS79QqfD3++OOUlpayatUqzGYzW7duZfTo0ZGuTURERKTeqVX4+uWXX3jggQewWCzEx8fzxBNPsGbNmkjXJiIiIlLv1Cp8GYaB2+3GMAwAcnNzQ9MiIiIiUnuW2mw0ZMgQbr31VnJycpg4cSILFizgj3/8Y6RrExEREal3ahW+rrnmGjp37szixYvx+XxMnz6dDh06RLo2ERERkXqnVuHr//2//8e0adNo06ZNaNnNN9/MzJkzI1aYiIiISH0UNnzde++9rF69mt27d3PppZeGlvt8Ppo0aRLx4kRERETqm7Dha8qUKeTl5TFx4kTGjBlzYCeLhYYNG0a8OBEREZH6Juzdjk6nkxYtWtCsWTOaN28e+mrcuDEPP/xwtGoUERERqTfC9nw9/PDDbNu2jZUrV7Ju3brQcq/XS2FhYcSLExEREalvwoavYcOGkZ2dzcSJE7n33ntDy81mM61bt454cSIiIiL1TdhhxxYtWnDOOefw4Ycf0qxZM0pKSujWrRtpaWkkJydHqUQRERGR+qNWT7j/+OOPGTZsGBMmTCAvL4+BAwfywQcfRLo2ERERkXqnVuHrlVde4a233sLpdNKwYUPef/99Xn755UjXJiIiIlLv1Cp8mUwmnE5naD4tLQ2TqVa7ioiIiEgltXrCfdu2bXnjjTfwer2sXr2af/7zn3q9kIiIiMhRqFX31SOPPMLu3bux2+2MHj0ap9PJo48+GunaREREROqdWvV8ORwORowYwYgRI2p9YL/fz7hx41i7di02m40JEyaQkZFRbbuxY8ficrn405/+VPuqRURERE5QtQpfHTp0wDCMKstSU1P55ptvDrvPggULcLvdvP322yxbtowpU6Ywffr0KtvMnj2bX3/9lbPOOusoShcRERE58dQqfK1ZsyY07fF4WLBgAcuWLQu7z9KlS+nevTsAXbp0YeXKlVXW//TTT/z8888MGDCAjRs3HmHZIiIiIiemWoWvyqxWK1dccQUvvvhi2O2Kioqq3CFpNpvxer1YLBZ+++03nnvuOZ577jk++eSTWp03JcWBxWI+0nKPSmpqYlTOc6JS+4Sn9qmZ2ig8tU/N1EbhqX1qFss2qlX4mjdvXmg6EAiwbt06LJbwuzqdToqLi0Pzfr8/tM+nn35Kbm4ud955Jzk5OZSVlZGVlUXfvn0Pe7zc3JLalHrMUlMTycnReysPR+0TntqnZmqj8NQ+NVMbhaf2qVk02ihcuKtV+Fq8eHGV+ZSUFJ599tmw+3Tt2pWFCxfSq1cvli1bRrt27ULrhgwZwpAhQwCYO3cuGzduDBu8REREROqLWoWvyZMn4/F42LRpEz6fj7Zt29bY89WzZ08WLVrEwIEDCQQCTJo0ifnz51NSUsKAAQOOS/EiIiIiJ5paha+VK1cyfPhwkpOT8fv97Nmzh+eff57TTz/9sPuYTCYef/zxKstat25dbTv1eImIiMjJpFbha8KECTzzzDOhsLVs2TLGjx/PnDlzIlqciIiISH1Tqyfcl5SUVOnl6tKlC+Xl5RErSkRERKS+qlX4crlcLFiwIDS/YMECkpOTI1WTiIiISL1Vq2HH8ePHM3LkSB5++GEAWrZsyZNPPhnRwkRERETqo1qFr8zMTN59911KSkrw+/1VHp4qIiIiIrVXq/C1fPlyZsyYQW5uLoFAILT89ddfj1hhIiIiIvVRrcLXgw8+yE033USbNm2qvWBbRERERGqvVuErLi6OQYMGRboWERERkXovbPjasWMHAB07duS1117j0ksvxWw+8HLrZs2aRbY6ERERkXombPi66aabQtPff/99lWu8DMPgiy++iFxlIiIiIvVQ2PD15ZdfRqsOERERkZNC2PD10EMPhd158uTJx7UYERERkfoubPg6++yzo1WHiIiIyEkhbPi64IILSE1NDV14LyIiIiLHJmz4GjNmDC+99BI33XTTIZ/vpQvuRURERI5M2PD10ksvsXDhQl577TXS09P5/PPPmTNnDqeccgrDhg2LVo0iIiIi9YYp3MoZM2bw3HPP4Xa7WbNmDSNHjqRHjx7k5+czderUaNUoIiIiUm+E7fmaN28eb7/9NvHx8UydOpVLLrmE/v37EwgE6NWrV7RqFBEREak3wvZ8GYZBfHw8AIsXL6Z79+6h5SIiIiJy5ML2fJnNZgoKCigpKWH16tWcf/75AGRnZ2Ox1Oq1kCIiIiJSSdgEdeedd3LNNdfg9Xq57rrrSEtL4+OPP+aZZ57hnnvuiVaNIiIiIvVG2PB1+eWXc8YZZ5Cbm0uHDh0ASEhIYMKECZxzzjlRKVBERESkPqlx7LBx48Y0btw4NH/RRRdFtCARERGR+izsBfciIiIicnwpfImIiIhEkcKXiIiISBQpfImIiIhEkcKXiIiISBQpfImIiIhEkcKXiIiISBQpfImIiIhEkcKXiIiISBQpfImIiIhEkcKXiIiISBQpfImIiIhEkcKXiIiISBQpfImIiIhEkcKXiIiISBQpfImIiIhEkcKXiIiISBQpfImIiIhEkcKXiIiISBQpfImIiIhEkcKXiIiISBRZInVgv9/PuHHjWLt2LTabjQkTJpCRkRFa/9FHHzFz5kzMZjPt2rVj3LhxmEzKgiIiIlK/RSztLFiwALfbzdtvv82IESOYMmVKaF1ZWRnPPvssr7/+OrNnz6aoqIiFCxdGqhQRERGROiNi4Wvp0qV0794dgC5durBy5crQOpvNxuzZs4mPjwfA6/Vit9sjVYqIiIhInRGxYceioiKcTmdo3mw24/V6sVgsmEwmGjVqBMCsWbMoKSnh/PPPD3u8lBQHFos5UuVWkZqaGJXznKjUPuGpfWqmNgpP7VMztVF4ap+axbKNIha+nE4nxcXFoXm/34/FYqky/9RTT7Fp0yamTZuGYRhhj5ebWxKpUqtITU0kJ6cwKuc6Eal9wlP71ExtFJ7ap2Zqo/DUPjWLRhuFC3cRG3bs2rUr33zzDQDLli2jXbt2VdY/8sgjlJeX88ILL4SGH0VERETqu4j1fPXs2ZNFixYxcOBAAoEAkyZNYv78+ZSUlNC5c2fmzJnDmWeeyc033wzAkCFD6NmzZ6TKEREREakTIha+TCYTjz/+eJVlrVu3Dk2vWbMmUqcWERERqbP0YC0RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8RERGRKLLEuoC6oshdzFtr52Kzm7H4bCRYHRVfCaFpp9WBw+ogweLAbDLHumQRERE5ASl8VSj2FLNm3zrKfGW12j7OHFcpoFX6slQNbAcCXDxx5jgMw4jwJxEREZG6TOGrQuOENJ66cBzxLjNbd+6m2FtCsefgr+ID0xXrdxbvxuP31OocJsNUEc6q9qo5rPE4LYcKbMGeNqtJ/5lERETqC/1Wr8RkmEiyO2mcEDii/dw+zyGDWbXA5imh2FtMoaeI3SU5BKjdeWxmGwmW4LDnwb1qjlBv24F1TquDOEscJkOX9ImIiNQ1Cl/Hgc1sxWZOJiUuudb7+AN+yrxlFB0c0rwllByix63IU8Lu0j24i3bU6vgGBg5rfMVQ6KGHQROsCdV64mxm61G2goiIiNSGwleMmAwTjoqeqyPh8XsrhbOqvWpF3mJKPKXV1u0p3Yc/4K/V8a0m60HXr1XtVWtW3Aiz24bL7iLZ7sJuth3NxxcRETlpKXydYKwmCy57Ei57Uq33CQQClPnKDnENW0VIO8T1bXtLc8n27ax+sPVVZ+Mt8STbk0iuCGPJ9iRcdhcpdldFQEvCaU3QjQYiIiIVFL5OAoZhEG+JJ94ST6P4hrXez+f3Vbt+LWDzsn3vbvLKC8grz6/4KmBn8e7DHsdisuCyJR06pMW5cNlcuOyJWHRjgYiInAT0204Oy2wyk2RLJMmWGFqWmppIjquw2rblPjd55fnkV4SxvLJ88tyVpsvz2Zi/JexNBolWJ8lxh+492z8db4mLyGcVERGJFoUvOS7sZhuNHak0dqQedhuf30ehp4jcskohLdR7lk9+eQG7in9jW2F22PMc6D1z4aoUzPb3rCXanLrTU0RE6iyFL4kas8kcCk2HEwgEKPWWkldeQG6oJ61qUMsvL2B3Sc5hj2EyTNWGOV1VhjyDQc2qOztFRCQGFL6kTjEMI3QXaDNnk8Nu5/F5yHcXVAxr5pHnLjgQ0iqGObcUbmdTwdbDHiPB6jgQzGyu0JBn5ZDmsMTrZgERETmuFL7khGQ1W2kU3zDsDQT+gJ9Cd/Fhe89yy/PZW7qP7KJD3NW5/zwmS6Xrz6r3niXbXSTZEvWuTxERqTWFL6m3TIYJlz0Rlz2RdFocdrsyb1m1uzcPhLTg/Pq8TYe9WcDAIMnmxGV3kZbYAHwmrCZr8MtswRaaDn63maxYTBZsFfOH285qsmA2zOp5ExGpZxS+5KQXZ4mjiSWOJglph93G5/cdGOY86CaB/TcQ7CjexdbC7ce1NgMDq9laKZhZQgHOarJiqRzaKoKbrSK4VQ5xVddVDoLVt1MvnohIZCl8idSC2WSmQVwKDeJSDrtNIBDAkWxm52+5uH0ePP6KL58Ht98bmg4u91Za56m0zhuc93nw7p+utF+pt4wCXxEevwdfwBeRz2oyTIcNaZb90wf15AWnLVV68qquq9jWbCXepTtRReTkFrHw5ff7GTduHGvXrsVmszFhwgQyMjJC67/88kuef/55LBYL/fr14/rrr49UKSJRYRgGTlsCyfbavcrpWPkDftwVIc3j94RC2/7Q5/ZVCnl+Dx7f4bYLLvf6K0LiQcco9hSTVxEYa/uaqpo0TWhMliuT1q5MWidn0jCugYZXReSkEbHwtWDBAtxuN2+//TbLli1jypQpTJ8+HQCPx8PkyZOZM2cO8fHx3HDDDVx88cWkph7+GVEiUpXJMBFnsQP2qJ3T5/eFeu6q9O6Fwl5F4Dto2l1pm1zvXtbu2cTO4t0s2rEYAJctMRjGkluR5cqghbOZhj9FpN6KWPhaunQp3bt3B6BLly6sXLkytG7Dhg2kp6fjcgWf99StWzeWLFnCFVdcEalyROQ4MJvMmE1mjuU9A6mpiezanUd20U425G9mQ/5mNuZt4qecFfyUswIAm9lGZlJ6sGfMlUmmK11vNxCReiNi4auoqAin0xmaN5vNeL1eLBYLRUVFJCYeeGVNQkICRUVFYY+XkuLAYonOX8KpqYk1b3QSU/uEp/apWZPGyTRpnEw3OgLB6+VyiveyZs8G1uzZwNqc9fyaG/yC4JBuhqs5HRq1oX1qFh0ataGh4/DX353o9DNUM7VReGqfmsWyjSIWvpxOJ8XFxaF5v9+PxWI55Lri4uIqYexQcnNLIlPoQVJTE8nJqf7uQglS+4Sn9qnZ4drIwE7HhFPomHAKZECxp4RN+VuCvWN5m9lSsI3Nedv5dP1XAKTYk2mdHOwZy3Jl0szZpF68Vko/QzVTG4Wn9qlZNNooXLiLWPjq2rUrCxcupFevXixbtox27dqF1rVu3ZotW7aQl5eHw+FgyZIl3H777ZEqRUROQAlWB50bdaRzo2DvmMfvZVthNhsrwtjG/M0s2b2MJbuXARBnjiPLlVFx7VgmmUktsZltMfwEIiKHFrHw1bNnTxYtWsTAgQMJBAJMmjSJ+fPnU1JSwoABAxg1ahS33347gUCAfv360bhx40iVIiL1gNVkqQhXGfRIv4hAIMBvJTlsyN/ChvxNbMzfzKp9a1m1by0QvCGhZWLzUM9YlisTl11DMSISe0YgEDj0Y7vrmGh1oaq7Njy1T3hqn5pFso0K3UVVesa2FmZXeR5ao/iGoYv4Wydn0tiRVucecaGfoZqpjcJT+9Ss3g47iohEW6LNyempnTk9tTMAbp+HLQXbgndUVnwt3rWUxbuWApBgcZCVXDFU6WpFemJzrGZrLD+CiJwEFL5EpN6yma20TcmibUoWEHww7a7i39iQv4kNeVvYmL+JFXtWs2LPagAshpn0pJahnrFWrgyc1oRYfgQRqYcUvkTkpGEyTDRzNqGZswndm58HQF55fmiYckP+Zjblb2Fj/mY+3xrcp4kjLXQRf5Yrk9T4hnVuqFJETiwKXyJyUku2u+jW+HS6NT4dgDJvGZv3D1XmbWZTwRb+u/MH/rvzByA4tLn/urGs5ExaOpvrafwickQUvkREKomzxNGhQVs6NGgLBF+ptKN4V5XesWU5K1mWE3xrh9VkJbNiqDIruRVZrnTiLfGx/AgiUscpfImIhGE2mWmZ2JyWic35fcvzCQQC7CvLq3i8xRY25G1ifd4m1uVthC1gYNDM2aTKi8MbxNXfp/GLyJFT+BIROQKGYdAwPoWG8Smc3aQrACWeUjYVbGFjXrBnbHPBNrKLdvJt9ndAcGhz/zBla1cmzZ1N68XT+EXk6Ch8iYgcI4c1nk4NO9CpYQcAvH4v2wp3hIYpN+ZtZulvP7P0t58BiDPbgy8Or7iIPzMpnTiLPZYfQUSiSOFLROQ4s5gstHKl08qVzqVcGHxxeOkeNuRvYWPeJjbkb2FN7jrW5K4DgndhtnA2pbWrFZ2L21JW4sMgOIRpGAYGwbsrDcMUXF6x7MD64LrgsUJ7YhhU2Q4MTJWOF9p2/zEwhfbZvy2Vz3fI81Y+3/61Vfeh0mcREYUvEZGIMwyDNEcqaY5Uzmt6JgBF7uIDPWP5m9lasJ2thdks3P6fGFcbOQdC5EEBDgMMAxMHhcHDhMwEWzwOswOnzUmSzUmiNTH43eYk0ZZIYsVyvdtT6iqFLxGRGHDaEjgttROnpXYCwOPzsKVwO/nso6CwlEDATwAIECAQCFT6DsG3wgXwE4D96yqW798OwI8fAgcdgwCB/ctC21Y9dtXt/JWOF+58++s5VM1Vz3u48xGoOEbF8qrnq9gnEKCgvJBs964a29hutpFoqwhm1gPhLKlSSNsf1OLMceqZk6hR+BIRqQOsZittkluRmnqa3stXg9TURHbtzqPIU0yBu4hCdyGF7iIKPUUU7J92B6eL3EVsLtiGP+APe0yLyUKi1UmSLZFEW0KlHrSKkGZ1huYd1njdMCHHROFLREROOGaTGZc9CZc9qcZt/QE/JZ7SA8HMcyCcFVaEt4KKwJZdvBNvoTfs8UyGiURrQpXes1BQO6iHzWlN0EN4pRqFLxERqddMhgmnLQGnreb3dAYCAcp8ZaEwVjWcVe5hKyKndA/bi3aEPZ6BQYK14vq0Sr1nB65PS6gybzXp1/LJQP+VRUREKhiGQbwlnnhLPI0dqTVu7/a5KwW1wooetSIKPZWHP4soKC9gV/HuGo8Xb4k7xPVpVW8kSLQGp+va40n8AT++gB+f34c/4AtOB3z4/L7g94p1voCvyrbBdb6D5v34/ZWOEfDh9/sPv23Ahy/M+so1BQJ++nT6Ax0TTolZWyl8iYiIHCWb2Uaj+AY0im9Q47Zev/egYc9KvWnuqkOhOaV7K25OCHNuk7V6MLMlkrYvhcKi0kOGnv1hpmqw2R90DrPtIYLMoY5VU72xZjbMmAwTFpOZgvJCqLkjNGIUvkRERKLAYrKQEpdMSlxyjdv6A36KPSWhMFZwcEir1LO2rTAbX8B3XGs1MDAbJkwmM2bDjNkwBecNM1azlTgjLrisyvpguKmyrNK06aBtqy07eL2p4niHWl8xbapUW5Vlpv37VZzHMFW5mzU1NTGmN7YofImIiNQxJsMUupC/JoFAgFJvaagnzeKAosLySsHEXBFMqk9XC0uVQo9EjsKXiIjICcwwDBxWBw6rgyYJaTHv1ZGaKdqKiIiIRJHCl4iIiEgUKXyJiIiIRJHCl4iIiEgUKXyJiIiIRJHCl4iIiEgUKXyJiIiIRJHCl4iIiEgUKXyJiIiIRJHCl4iIiEgUGYFAoG6/hlxERESkHlHPl4iIiEgUKXyJiIiIRJHCl4iIiEgUKXyJiIiIRJHCl4iIiEgUKXyJiIiIRJEl1gXUBR6Ph9GjR5OdnY3b7WbYsGFceumlsS6rTvH5fIwZM4ZNmzZhNpuZPHky6enpsS6rztm7dy99+/ZlxowZtG7dOtbl1DnXXHMNiYmJALRo0YLJkyfHuKK65aWXXuLLL7/E4/Fwww030L9//1iXVKfMnTuX999/H4Dy8nJWr17NokWLSEpKinFldYPH42HUqFFkZ2djMpkYP368/h2qxO1289BDD7Ft2zacTiePPPIImZmZMalF4Qv48MMPSU5O5qmnniI3N5drr71W4esgCxcuBGD27NksXryYyZMnM3369BhXVbd4PB4eeeQR4uLiYl1KnVReXg7ArFmzYlxJ3bR48WJ++ukn3nrrLUpLS5kxY0asS6pz+vbtS9++fQF47LHH6Nevn4JXJV9//TVer5fZs2ezaNEinn32WaZNmxbrsuqMd955B4fDwTvvvMPGjRsZP348r776akxq0bAjcPnll3PfffeF5s1mcwyrqZt69OjB+PHjAdixYweNGjWKcUV1zxNPPMHAgQNJS0uLdSl10po1aygtLeW2225jyJAhLFu2LNYl1Sn/+c9/aNeuHffccw933303v//972NdUp21YsUK1q9fz4ABA2JdSp3SqlUrfD4ffr+foqIiLBb1r1S2fv16LrzwQgCysrLYsGFDzGrRfxkgISEBgKKiIoYPH879998f24LqKIvFwoMPPsjnn3/O3/72t1iXU6fMnTuXBg0a0L17d15++eVYl1MnxcXFcfvtt9O/f382b97M0KFD+fTTT/ULokJubi47duzgxRdfZPv27QwbNoxPP/0UwzBiXVqd89JLL3HPPffEuow6x+FwkJ2dzRVXXEFubi4vvvhirEuqUzp27MjChQvp0aMHP//8M7t378bn88Wkw0U9XxV27tzJkCFD6NOnD1dddVWsy6mznnjiCT777DPGjh1LSUlJrMupM9577z3++9//MnjwYFavXs2DDz5ITk5OrMuqU1q1asXVV1+NYRi0atWK5ORktVElycnJXHDBBdhsNrKysrDb7ezbty/WZdU5BQUFbNy4kXPPPTfWpdQ5r732GhdccAGfffYZH3zwAaNGjQoN9wv069cPp9PJkCFDWLhwIZ06dYrZSJfCF7Bnzx5uu+02Ro4cyXXXXRfrcuqkefPm8dJLLwEQHx+PYRganq3kzTff5I033mDWrFl07NiRJ554gtTU1FiXVafMmTOHKVOmALB7926KiorURpV069aNb7/9lkAgwO7duyktLSU5OTnWZdU5P/74I7/73e9iXUadlJSUFLqhxeVy4fV68fl8Ma6q7lixYgXdunVj1qxZ9OjRg5YtW8asFr1YG5gwYQKffPIJWVlZoWWvvPKKLpyupKSkhIceeog9e/bg9XoZOnQoPXr0iHVZddLgwYMZN26c7jI6yP47jXbs2IFhGPzpT3+ia9eusS6rTnnyySdZvHgxgUCA//u//6N79+6xLqnO+fvf/47FYuGWW26JdSl1TnFxMaNHjyYnJwePx8OQIUM0klPJvn37eOCBBygtLSUxMZGJEyfSuHHjmNSi8CUiIiISRRp2FBEREYkihS8RERGRKFL4EhEREYkihS8RERGRKFL4EhEREYkihS8ROaEVFRXx2GOP0bt3b/r06cPgwYP55ZdfWLx4MYMHDz7i4xUWFurp6SISUQpfInLC8vv9DB06FJfLxbx58/jggw+45557GDp0KHl5eUd1zPz8fFavXn18CxURqUThS0ROWIsXL2bnzp0MHz489I7Ic889l8mTJ1d5svfgwYNZvHgxANu3b+eSSy4BYP78+fTp04e+ffsyfPhwysvLmTBhAr/99luo92vevHlce+219OnTh9GjR4de13Luuedyxx130KdPHzweTzQ/toic4BS+ROSEtWrVKjp06IDJVPWfsosuuoiGDRvWuP+zzz7LjBkzmDt3Ls2bN2fjxo2MGTOGtLQ0nn/+edatW8c777zD7Nmz+eCDD2jYsCGvvvoqEHwR9tChQ/nggw+wWq0R+XwiUj9ZYl2AiMjRMplM2O32o97/4osv5oYbbqBHjx5cdtlldOzYke3bt4fWL168mC1btnD99dcD4PF4OOWUU0LrTz/99KMvXkROWgpfInLC6ty5M//85z8JBAIYhhFa/vTTT1d7+fL+N6l5vd7QsjFjxrBmzRq+/vprRo4cyb333ku3bt1C630+H1dccQVjxowBgu/Oqzycqfe/isjR0LCjiJywzjzzTBo2bMhzzz0XCkXffvstc+fOZd++faHtUlJSWL9+PQALFiwAgiHsD3/4AykpKdx111306dOH1atXY7FYQgHtnHPO4fPPP2fv3r0EAgHGjRvHzJkzo/wpRaS+Uc+XiJywDMPghRdeYPLkyfTu3RuLxUJKSgovv/wyhYWFoe3uuOMORo0axXvvvcell14KgMViYfjw4dx2223Y7XYaNmzIlClTSEpKolmzZgwePJhZs2Zx7733cvPNN+P3++nYsSN33nlnrD6uiNQTRmB/X7yIiIiIRJyGHUVERESiSOFLREREJIoUvkRERESiSOFLREREJIoUvkRERESiSOFLREREJIoUvkRERESiSOFLREREJIr+Pw3Qjopg3RBWAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "seuclid = []\n",
    "scosine = []\n",
    "\n",
    "k = range(2,10)\n",
    "for i in k:\n",
    "    hc_model = AgglomerativeClustering(n_clusters=i, linkage = 'average').fit(x_tune_slim)\n",
    "    labels = hc_model.labels_\n",
    "    seuclid.append(metrics.silhouette_score(x_tune_slim, labels, metric='euclidean'))\n",
    "    scosine.append(metrics.silhouette_score(x_tune_slim, labels, metric='cosine'))\n",
    "    \n",
    "plt.figure(figsize=(10,5))\n",
    "plt.plot(k,seuclid,label='euclidean')\n",
    "plt.plot(k,scosine,label='cosine')\n",
    "plt.ylabel(\"Silhouette\")\n",
    "plt.xlabel(\"Cluster\")\n",
    "plt.title(\"Silhouette vs Cluster Size\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Comparing the Silhouette score for both Euclidean and Cosine, there is a clear similar trend between the distance metrics. However, Cosine's Silhouette score shows that its trend is hovering near 0, denoting overlapping clusters. Additionally, it goes below 0 which tells us that the data points are wrongly assinged to a cluster as we can see (above) in the scatter plot."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average accuracy (with kmeans for matchDuration/totalDistance)=  92.10392019663368 +- 0.17108095385156977\n"
     ]
    }
   ],
   "source": [
    "# MATCH DURATION + TOTAL DISTANCE (ON X_TEST)\n",
    "\n",
    "x_match_total = x_test[['matchDuration','totalDistance']]\n",
    "cls_match_total = KMeans(n_clusters=8, init='k-means++',random_state=17)\n",
    "cls_match_total.fit(x_match_total)\n",
    "newfeature_match_total = cls_match_total.labels_ # the labels from kmeans clustering\n",
    "cv_match_total = StratifiedKFold(n_splits=10)\n",
    "x_match_total = x_test.loc[:, cols_df]\n",
    "y_match_total = y_test.loc[:, ('quart_binary')]\n",
    "x_match_total = np.column_stack((x_match_total,pd.get_dummies(newfeature_match_total)))\n",
    "acc_match_total = cross_val_score(clf,x_match_total,y=y_match_total,cv=cv_match_total)\n",
    "print (\"Average accuracy (with kmeans for matchDuration/totalDistance)= \", acc_match_total.mean()*100, \"+-\", acc_match_total.std()*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Agglomerative Clustering has memory-space error if run on full x_test\n",
    "indices_slim2 = x_test.sample(n=2000, replace=False, random_state=17).index\n",
    "x_test_slim = x_test.loc[indices_slim2, :]\n",
    "y_test_slim = y_test.loc[indices_slim2, :]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average accuracy (with kmeans for matchDuration/totalDistance)=  90.75 +- 0.17108095385156977\n"
     ]
    }
   ],
   "source": [
    "cv_match_total = StratifiedKFold(n_splits=10)\n",
    "x_test = x_test_slim.loc[:, cols_df]\n",
    "y_test = y_test_slim.loc[:, ('quart_binary')]\n",
    "acc_match_total1 = cross_val_score(clf,x_test,y=y_test,cv=cv_match_total)\n",
    "print (\"Average accuracy (with kmeans for matchDuration/totalDistance)= \", acc_match_total1.mean()*100, \"+-\", acc_match_total.std()*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline average accuracy =  90.75 +- 1.8200274723201293\n"
     ]
    }
   ],
   "source": [
    "X = x_test.loc[:, cols_df]\n",
    "y = y_test\n",
    "cv = StratifiedKFold(n_splits=10)\n",
    "\n",
    "# clf is the Random classifier model\n",
    "acc1 = cross_val_score(clf, X=X, y=y, cv=cv)\n",
    "\n",
    "print (\"Baseline average accuracy = \", acc1.mean()*100, \"+-\", acc1.std()*100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Based on our final externally cross-validated test analysis, we find the KMeans process with `matchDuration` + `totalDistance` clusters (n=8) to be the best model for predicting `quart_binary`, our response of interest."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='ME4'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Modeling and Evaluation 4**\n",
    "\n",
    "Jump to [Top](#TOP)\n",
    "\n",
    "*Assignment: Summarize the ramifications.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Clustering may help identify hidden trends and produce improved analysis, leveraging it in predicting `quart_binar` (our developed feature from the original `winPlacePerc` continuous variable). This ends up as the case for us as well, since it produced an improved prediction. Our final accuracy score from this model was **92.1039 +- 0.1711** as compared with the \"baseline\" RandomForestClassifier which had a score of **90.75 +- 1.8200** on the new, `x_test` data.\n",
    "\n",
    "Due to the wider variance found in the RandomForest model, we do not expect the results to be verifiable via Student's t-test analysis. However, we would be hesitant to recommend this model to a client alone. While there *was* additional information to tease out of the data with clustering, it is unclear whether this is the most useful tool for PUBG placement predictions.\n",
    "\n",
    "We suggest that the results we have found here could be used as a start for additional feature selection—specifically for ranked matches.\n",
    "\n",
    "For future analysis, we may switch to F1-score analysis as an improved metric in prediction capability. Also, we would be interested in pursuing the Agglomerative Clustering mechanism if it was capable of processing larger amounts of information."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='DEPLOYMENT'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **DEPLOYMENT**\n",
    "\n",
    "Jump to [Top](#TOP)\n",
    "\n",
    "*Assignment: Be critical of your performance and tell the reader how you current model might be usable by other parties.* "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Q1: Did you achieve your goals?**\n",
    "\n",
    "One of the limitations we artificially imposed on ourselves was removing the `matchID` and `groupID` classifiers. These categoricals were essential to the original winners of the Kaggle competition in helping them assign `winPlacePerc` in the most accurate fashion.\n",
    "\n",
    "Our approach focused more on individual player stats, so this is a benefit of our model over theirs: ours is more generalizable.\n",
    "\n",
    "While we did improve our prediction capabilities over the baseline *RandomForestClassifier*, the result isn't statistically significant. It is *practially* significant, and for that we are grateful for this exercise."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Q2: If not, can you reign in the utility of your modeling?**\n",
    "\n",
    "One of the strongest pieces of evidence for this model was the fact that we produced a smaller standard deviation than the Random Forest classifier."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Q3: How useful is your model for interested parties (i.e., the companies or organizations that might want to use it)?**\n",
    "\n",
    "While this approach by itself isn't more useful than the RandomForestClassifier, it is important to note that not much more information can be extracted from player-stats alone."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Q4: How would your deploy your model for interested parties?**\n",
    "\n",
    "One possibility for deployment would be to generate an R-shiny application or convert our Python-based modeling into equivalent Javascript for a web-based deployment. This would be a tool that could be managed and updated by a team rather than giving full access to the client."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Q5: What other data should be collected?**\n",
    "\n",
    "If clients were interested in betting on well-known teams and individuals, we may be able to tailor this analysis to those specific requests. One of the downsides of the Kaggle data is that it is anonymized.\n",
    "\n",
    "A different approach would be to utilize clustering and classification or regression models to generate strong team-pairings based on individual e-sport competitors' stats and strategies."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Q6: How often would the model need to be updated, etc.?**\n",
    "\n",
    "Thankfully, the model need only be updated whenever there is a major tournament or event in PUBG. It could be run more frequently depending on frequency of events, but typically this approach would only need to be done once, prior to the event."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='EXCEPTIONAL'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **EXCEPTIONAL WORK**\n",
    "\n",
    "Jump to [Top](#TOP)\n",
    "\n",
    "*Assignment: You have free reign to provide additional analyses or combine analyses.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Performed additional EDA on attributes that might be useful for clusters.\n",
    "* Performed additional T-Tests to assess KMEANS & HAC model parameters performance.\n",
    "* Created additional (new) attributes to aid with our clusters.\n",
    "* Utilized Silhouette score to inform modeling."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='APPENDIX'></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **APPENDIX**\n",
    "\n",
    "Jump to [Top](#TOP)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## DATA DEFINITIONS"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Original variables\n",
    "\n",
    "The first three attributes (see above)—*Id*, *groupID*, and *matchID*—are alpha-numeric identifiers for the players, the teams/groups, and the individual matches that players participated in. These are stored as objects but could also be identified as strings.\n",
    "\n",
    "The variables of type integer are *assists*, *boosts*, *DBNO*s, *headshotKills*, *heals*, *killPlace*, *killPoints*, *kills*, *killStreaks*, *matchDuration*, *maxPlace*, *numGroups*, *rankPoints*, *revives*, *roadKills*, *teamKills*, *vehicleDestroys*, *weaponsAcquired*, and *winPoints*.\n",
    "\n",
    "* DBNOs = Number of enemy players knocked. DBNO means “down but not out.” Essentially, “knocked” means to be knocked out. When a player is knocked, they lose their ability to shoot, move quickly, and to hold a gun. Players do not die instantly when knocked, when playing in a group (on a team) - other members of the team can revive a knocked player.\n",
    "\n",
    "* assists = Number of enemy players this player damaged that were killed by teammates.\n",
    "\n",
    "* boosts = Number of boost items used. Boost items are things that improve a player’s health, examples are energy drinks, painkillers, and adrenaline syringes.\n",
    "\n",
    "* damageDealt = Total damage dealt. Note: Self inflicted damage is subtracted.\n",
    "\n",
    "* headshotKills = Number of enemy players killed with headshots. \n",
    "\n",
    "* heals = Number of healing items used.\n",
    "* Id - Player’s Id\n",
    "\n",
    "* killPlace = Ranking in match of number of enemy players killed.\n",
    "\n",
    "* killPoints = Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.\n",
    "\n",
    "* killStreaks = Max number of enemy players killed in a short amount of time.\n",
    "\n",
    "* kills = Number of enemy players killed.\n",
    "\n",
    "* longestKill = Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.\n",
    "\n",
    "* matchDuration = Duration of match in seconds.\n",
    "\n",
    "* matchId = ID to identify match. There are no matches that are in both the training and testing set.\n",
    "\n",
    "* matchType = String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.\n",
    "\n",
    "* rankPoints = Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.\n",
    "\n",
    "* revives = Number of times this player revived teammates.\n",
    "\n",
    "* rideDistance = Total distance traveled in vehicles measured in meters.\n",
    "\n",
    "* roadKills = Number of kills while in a vehicle.\n",
    "* swimDistance - Total distance traveled by swimming measured in meters.\n",
    "\n",
    "* teamKills = Number of times this player killed a teammate.\n",
    "\n",
    "* vehicleDestroys = Number of vehicles destroyed.\n",
    "\n",
    "* walkDistance = Total distance traveled on foot measured in meters.\n",
    "\n",
    "* weaponsAcquired = Number of weapons picked up.\n",
    "\n",
    "* winPoints = Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.\n",
    "\n",
    "* groupId = ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.\n",
    "\n",
    "* numGroups = Number of groups we have data for in the match.\n",
    "\n",
    "* maxPlace = Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.\n",
    "\n",
    "* winPlacePerc = The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match. (Higher is better.)\n",
    "\n",
    "*Definitions pulled from Kaggle*: https://www.kaggle.com/c/pubg-finish-placement-prediction/data\n",
    "\n",
    "*Additional details added to elaborate on/clarify some of the variables.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Variables we created <br/>\n",
    "\n",
    "* healItems = sum of heals + boosts\n",
    "\n",
    "* quart_binary = value is 0/1; if winPlacePerc in top quartile, value = 1, else 0.\n",
    "\n",
    "* quart_int = numeric representation of the winPlacePerc quartile.\n",
    "\n",
    "* quartile = varchar representation of quartile of winPlacePerc.\n",
    "\n",
    "* totalDistance = sum of rideDistance + swimDistance + walkDistance.\n",
    "\n",
    "* totalItems = sum of heals + boosts + weaponsAcquired.\n"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "b3ba2566441a7c06988d0923437866b63cedc61552a5af99d1f4fb67d367b25f"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}