# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:47:30.087649Z","iopub.execute_input":"2023-03-07T03:47:30.088657Z","iopub.status.idle":"2023-03-07T03:47:30.125568Z","shell.execute_reply.started":"2023-03-07T03:47:30.088587Z","shell.execute_reply":"2023-03-07T03:47:30.124454Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## importing database

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:48:15.010336Z","iopub.execute_input":"2023-03-07T03:48:15.010712Z","iopub.status.idle":"2023-03-07T03:48:17.321634Z","shell.execute_reply.started":"2023-03-07T03:48:15.010680Z","shell.execute_reply":"2023-03-07T03:48:17.320485Z"}}
import plotly.express as px
import pandas as pd  
import numpy as np

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:49:31.424385Z","iopub.execute_input":"2023-03-07T03:49:31.424819Z","iopub.status.idle":"2023-03-07T03:49:31.493303Z","shell.execute_reply.started":"2023-03-07T03:49:31.424777Z","shell.execute_reply":"2023-03-07T03:49:31.492218Z"}}
base = pd.read_csv('/kaggle/input/mushroom-attributes/mushroom.csv')
base

# %% [markdown]
# ## analysing the database

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:49:46.975872Z","iopub.execute_input":"2023-03-07T03:49:46.976277Z","iopub.status.idle":"2023-03-07T03:49:46.997291Z","shell.execute_reply.started":"2023-03-07T03:49:46.976241Z","shell.execute_reply":"2023-03-07T03:49:46.995804Z"}}
base.info

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:50:58.340452Z","iopub.execute_input":"2023-03-07T03:50:58.340815Z","iopub.status.idle":"2023-03-07T03:50:58.442748Z","shell.execute_reply.started":"2023-03-07T03:50:58.340782Z","shell.execute_reply":"2023-03-07T03:50:58.441559Z"}}
np.unique(base)

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:51:22.944822Z","iopub.execute_input":"2023-03-07T03:51:22.946417Z","iopub.status.idle":"2023-03-07T03:51:23.015680Z","shell.execute_reply.started":"2023-03-07T03:51:22.946360Z","shell.execute_reply":"2023-03-07T03:51:23.014040Z"}}
base.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:52:55.144815Z","iopub.execute_input":"2023-03-07T03:52:55.146493Z","iopub.status.idle":"2023-03-07T03:52:55.166314Z","shell.execute_reply.started":"2023-03-07T03:52:55.146430Z","shell.execute_reply":"2023-03-07T03:52:55.165126Z"}}
base.isnull().sum()

# %% [markdown]
# ## spliting the predicts and the classfiers

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:53:11.465420Z","iopub.execute_input":"2023-03-07T03:53:11.465828Z","iopub.status.idle":"2023-03-07T03:53:11.473500Z","shell.execute_reply.started":"2023-03-07T03:53:11.465791Z","shell.execute_reply":"2023-03-07T03:53:11.472246Z"}}
x_mush = base.iloc[:, 0:22].values
x_mush

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:53:39.114779Z","iopub.execute_input":"2023-03-07T03:53:39.115172Z","iopub.status.idle":"2023-03-07T03:53:39.123013Z","shell.execute_reply.started":"2023-03-07T03:53:39.115140Z","shell.execute_reply":"2023-03-07T03:53:39.121779Z"}}
y_mush = base.iloc[:, 22].values
y_mush

# %% [markdown]
# ## preprocessing using the LabeEncoder to transform in numeric all atributes

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:53:37.260713Z","iopub.execute_input":"2023-03-07T03:53:37.261080Z","iopub.status.idle":"2023-03-07T03:53:37.516933Z","shell.execute_reply.started":"2023-03-07T03:53:37.261050Z","shell.execute_reply":"2023-03-07T03:53:37.515570Z"}}
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:53:48.146463Z","iopub.execute_input":"2023-03-07T03:53:48.146853Z","iopub.status.idle":"2023-03-07T03:53:48.159501Z","shell.execute_reply.started":"2023-03-07T03:53:48.146820Z","shell.execute_reply":"2023-03-07T03:53:48.157986Z"}}
label_encoder = LabelEncoder()
label_encoder

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:54:00.711178Z","iopub.execute_input":"2023-03-07T03:54:00.711747Z","iopub.status.idle":"2023-03-07T03:54:00.758848Z","shell.execute_reply.started":"2023-03-07T03:54:00.711673Z","shell.execute_reply":"2023-03-07T03:54:00.757459Z"}}
for i in range(22):
  x_mush[:,i] = label_encoder.fit_transform(x_mush[:,i])

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:54:10.428690Z","iopub.execute_input":"2023-03-07T03:54:10.429036Z","iopub.status.idle":"2023-03-07T03:54:10.437149Z","shell.execute_reply.started":"2023-03-07T03:54:10.429009Z","shell.execute_reply":"2023-03-07T03:54:10.435717Z"}}
x_mush

# %% [markdown]
# ## scaling all values

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:54:25.985280Z","iopub.execute_input":"2023-03-07T03:54:25.985670Z","iopub.status.idle":"2023-03-07T03:54:26.010060Z","shell.execute_reply.started":"2023-03-07T03:54:25.985634Z","shell.execute_reply":"2023-03-07T03:54:26.008507Z"}}
from sklearn.preprocessing import StandardScaler
scaler_mush = StandardScaler()
x_mush = scaler_mush.fit_transform(x_mush)
x_mush

# %% [markdown]
# ## spliting the data for trainning = 85% and test = 15%

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:54:43.154471Z","iopub.execute_input":"2023-03-07T03:54:43.154870Z","iopub.status.idle":"2023-03-07T03:54:43.225452Z","shell.execute_reply.started":"2023-03-07T03:54:43.154838Z","shell.execute_reply":"2023-03-07T03:54:43.224371Z"}}
from sklearn.model_selection import train_test_split

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:54:54.775747Z","iopub.execute_input":"2023-03-07T03:54:54.776111Z","iopub.status.idle":"2023-03-07T03:54:54.788773Z","shell.execute_reply.started":"2023-03-07T03:54:54.776078Z","shell.execute_reply":"2023-03-07T03:54:54.787537Z"}}
x_mush_treinamento, x_mush_teste, y_mush_treinamento, y_mush_teste = train_test_split(x_mush, y_mush, test_size = 0.15, random_state=0)
x_mush_treinamento.shape, y_mush_treinamento.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:55:07.270268Z","iopub.execute_input":"2023-03-07T03:55:07.270626Z","iopub.status.idle":"2023-03-07T03:55:07.279737Z","shell.execute_reply.started":"2023-03-07T03:55:07.270592Z","shell.execute_reply":"2023-03-07T03:55:07.278393Z"}}
x_mush_teste.shape, y_mush_teste.shape

# %% [markdown]
# ## trainning by using RandomForestClassifier

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:56:07.776715Z","iopub.execute_input":"2023-03-07T03:56:07.777157Z","iopub.status.idle":"2023-03-07T03:56:08.120806Z","shell.execute_reply.started":"2023-03-07T03:56:07.777117Z","shell.execute_reply":"2023-03-07T03:56:08.119311Z"}}
from sklearn.ensemble import RandomForestClassifier

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:56:11.200644Z","iopub.execute_input":"2023-03-07T03:56:11.201037Z","iopub.status.idle":"2023-03-07T03:56:11.206825Z","shell.execute_reply.started":"2023-03-07T03:56:11.201004Z","shell.execute_reply":"2023-03-07T03:56:11.205065Z"}}
random_forest_mush = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 0)

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:56:20.634240Z","iopub.execute_input":"2023-03-07T03:56:20.634598Z","iopub.status.idle":"2023-03-07T03:56:20.982850Z","shell.execute_reply.started":"2023-03-07T03:56:20.634566Z","shell.execute_reply":"2023-03-07T03:56:20.981475Z"}}
random_forest_mush.fit(x_mush_treinamento, y_mush_treinamento)

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:56:30.526606Z","iopub.execute_input":"2023-03-07T03:56:30.526967Z","iopub.status.idle":"2023-03-07T03:56:30.555550Z","shell.execute_reply.started":"2023-03-07T03:56:30.526937Z","shell.execute_reply":"2023-03-07T03:56:30.554161Z"}}
previsoes = random_forest_mush.predict(x_mush_teste)
previsoes

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:56:43.150408Z","iopub.execute_input":"2023-03-07T03:56:43.150784Z","iopub.status.idle":"2023-03-07T03:56:43.158577Z","shell.execute_reply.started":"2023-03-07T03:56:43.150748Z","shell.execute_reply":"2023-03-07T03:56:43.156675Z"}}
y_mush_teste

# %% [markdown]
# ## using metrics to show the results

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:57:04.018356Z","iopub.execute_input":"2023-03-07T03:57:04.018763Z","iopub.status.idle":"2023-03-07T03:57:04.031469Z","shell.execute_reply.started":"2023-03-07T03:57:04.018709Z","shell.execute_reply":"2023-03-07T03:57:04.029824Z"}}
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_mush_teste, previsoes)
accuracy_score(y_mush_teste, previsoes)

# %% [markdown]
# ### The results is 100% of precision

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:57:13.632929Z","iopub.execute_input":"2023-03-07T03:57:13.633283Z","iopub.status.idle":"2023-03-07T03:57:13.916157Z","shell.execute_reply.started":"2023-03-07T03:57:13.633251Z","shell.execute_reply":"2023-03-07T03:57:13.915356Z"}}
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(random_forest_mush)
cm.fit(x_mush_treinamento, y_mush_treinamento)
cm.score(x_mush_teste, y_mush_teste)

# %% [code] {"execution":{"iopub.status.busy":"2023-03-07T03:57:24.639458Z","iopub.execute_input":"2023-03-07T03:57:24.639847Z","iopub.status.idle":"2023-03-07T03:57:24.677390Z","shell.execute_reply.started":"2023-03-07T03:57:24.639809Z","shell.execute_reply":"2023-03-07T03:57:24.676395Z"}}
print(classification_report(y_mush_teste, previsoes))