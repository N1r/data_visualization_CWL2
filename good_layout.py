import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.add_vertical_space import add_vertical_space



#st.header('Generate the useful plot for analysis and interputation : thermometer: star:')
st.set_page_config(page_title = "useful plots for analysis and interpretation", layout = 'centered')

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 2, 0.2, 1, 0.1)
)
row0_1.title("Quick overview of perception data")

with row0_2:
    add_vertical_space()

row0_2.subheader(
    "A Streamlit web app by yiran-ding for convenience"
)


row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))

with row1_1:
    st.markdown(
        "Hey there! this app is designed for quick analysis and visualization data from Gorilla app!"
    )

    st.markdown(
        "**To begin, please upload the excel files (better be well-organzied) (or just use default (a small pilot study) so far).** 👇"
    )
    st.subheader('upload csv/excel files ')
    uploaded_files = st.file_uploader('Choose a file')

# Processing parts

df = pd.read_excel('data_exp_104329-v9_task-sh6f.xlsx')

need_columns = ['Spreadsheet: Audio_Canonical','Spreadsheet: Audio_Accent','Spreadsheet: Audio_base_line','Spreadsheet: Interspread_1','Spreadsheet: Interspread_2',
               'Participant Private ID','Display','Response','Spreadsheet: display'
               ]

new_columns = ['Canonical','Accent','Base_line','Interspread_1','Interspread_2',
               'Participant Private ID','Display','Response','Spreadsheet: display']

df_need = df[need_columns]
df_need.columns = new_columns

# 获得有反馈的 单元
df_respone = df_need.dropna(subset=['Response'])


# value counts between columns
# analysis paticipate one by one
all_person = df_respone['Participant Private ID'].unique()

def get_person_data(df_need):
    df_respone = df_need.dropna(subset=['Response'])
    df_base_line = df_respone[['Base_line','Response']].dropna(subset=['Base_line']).reset_index()
    #df_base_line.shape
    pitch = []
    duration = []
    for i in range(df_base_line.shape[0]):
        wav_name = df_base_line.Base_line[i]
        pitch_value = int(wav_name.split('&')[0][-2:])/16 + 1
        duration_value = int(wav_name.split('&')[1][-6:-4])/16 + 1
        pitch.append(pitch_value)
        duration.append(duration_value)

    df_base_line['pitch'] = pitch
    df_base_line['duration'] = duration
    return df_base_line

# second part
df_respone = df_need.dropna(subset=['Response'])
df_base_line = df_respone[['Base_line','Response']].dropna(subset=['Base_line']).reset_index()

df_base_line = get_person_data(df_base_line)


def lg_regress(df_base_line):
    from sklearn.linear_model import LogisticRegression

    y = df_base_line['Response']
    X = df_base_line[['pitch','duration']]

    #X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf.coef_[0]

all_coef = []
for i in all_person:
    df_person = df_respone[df_respone['Participant Private ID'] == i]
    df_temp = get_person_data(df_person)
    coef = lg_regress(df_temp)
    all_coef.append(coef)

# normalize
dd = np.array(all_coef)/np.array(all_coef).sum(axis=1,keepdims=1)

# 数据小幅度清理
to_plot = dd[0:-1]
df = pd.DataFrame(to_plot,columns=['pitch','duration'])
# df['start'] = np.zeros(4)
# df['end'] = np.ones(4)
# first plot

import plotly.express as px

#df = px.data.tips()
fig1 = px.violin(df,box=True,title = 'Normalized perceptual weights',width=600, height=600)
#fig1.show()
# second plot

df['start'] = np.zeros(4)
df['end'] = np.ones(4)

from itertools import chain
x = list(chain.from_iterable(zip(df['start'], df['end'])))
y = list(chain.from_iterable(zip(df['pitch'], df['duration'])))
line_df = pd.DataFrame(dict(
    x = x,
    y = y
))
fig2 = px.line(line_df, x="x", y="y",width=600, height=600)
fig2.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))
#f#ig2.show()

# third plot

df_respone = df_need.dropna(subset=['Response'])
df_base_line = df_respone[['Base_line','Response']].dropna(subset=['Base_line']).reset_index()
#df_base_line.shape

all_base_line = pd.crosstab(df_respone.Base_line,df_respone.Response)

qq = all_base_line[0]
colored_grid = np.array(qq).reshape(5,5)
new_heatmap = np.flip(colored_grid,axis=1)


import plotly.express as px

fig3 = px.imshow(new_heatmap, text_auto=True, width=500, height=500,color_continuous_scale=px.colors.sequential.Blues)
fig3.update_layout(title="Correlation heatmap",
                  yaxis={"title": 'Pitch'},
                  width=500,
                  height=500,
                  xaxis={"title": 'Duration',"tickangle": 45}, )



tab1, tab2, tab3 = st.tabs(["Violin and Connections lines", "Heatmap", "Raw data"])

#col1, col2, col3 = st.columns(3)

with tab1:

    row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
        (0.1, 1, 0.1, 1, 0.1)
    )

    with row3_1:
        st.subheader("Violin plots")
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        st.markdown(
            "This fig will show the perceptual normalized weights generated by LogisticRegression, value 0 indicates no reliance of certain cues and 1 for full reliance of certain cues "
        )

    with row3_2:
        st.subheader("Connections lines")
        # plots a bar chart of the dataframe df by book.publication year by count in plotly. columns are publication year and count
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
        st.markdown(
            "Trying to find a solution to overlay two figs later"
        )
        add_vertical_space()

with tab2:
    st.subheader("Heatmap")
    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
    st.markdown(
        "Heatmap indicates the perceptual weights of duration and pitch in baseline condition."
    )

with tab3:
    st.header('GLMER and raw_data')
    st.code('''
import bambi as bmb
# multi-level LogisticRegression by Bambi and Lmer comparsion

model_hierarchical = bmb.Model('Reponse ~ Condition + duration + (duration|Condition)', df, family = 'bernoulli')

model_hierarchical.fit(draws=1000, random_seed =1234, target_accept = 0.95, idata_kwargs={"log_likelihood":True})
''')

    st.markdown(
    "Next step: how to interpret results and anova test."
    )
    st.write(df_respone[0:20])
