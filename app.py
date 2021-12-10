import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects  as go
import pickle
import streamlit as st

class ExpressoApp():
	def __init__(self):
		model_name = 'CATBOOST_0xcb'
		self.model = pickle.load(open(model_name, 'rb'))

	def user_inputs(self):
		REGION = st.sidebar.selectbox('Location', ('DAKAR', 'THIES', 'SAINT-LOUIS', 'LOUGA',
			'KAOLACK', 'DIOURBEL', 'TAMBACOUNDA', 'KAFFRINE', 'KOLDA', 'FATICK', 'MATAM', 'ZIGUINCHOR',
			'SEDHIOU', 'KEDOUGOU'))
		TENURE = st.sidebar.selectbox('Duration in network', ("> 24 month", "18-21 month", 
	          "12-15 month", "15-18 month", "21-24 month", "9-12 month", "6-9 month", "3-6 month"))
		MONTANT = st.sidebar.number_input('Top-up amount', min_value=20.0)
		FREQUENCE_RECH = st.sidebar.number_input('Number of times refilled', min_value=1.0)
		REVENUE = st.sidebar.number_input('Monthly income')
		ARPU_SEGMENT = st.sidebar.number_input('Revenue over 90days / 3')
		FREQUENCE = st.sidebar.number_input('Number of times income made', min_value=1.0)
		DATA_VOLUME = st.sidebar.number_input('Number of connections')
		ON_NET = st.sidebar.number_input('Inter expresso call')
		ORANGE = st.sidebar.number_input('Call to orange')
		REGULARITY = st.sidebar.slider('Number of times active in 90 days', min_value=1, max_value=65, step=1)
		TOP_PACK = st.sidebar.text_input('The most active packs')
		FREQ_TOP_PACK = st.sidebar.number_input('Number of times TOP_PACK activated', min_value=0.0)

		columns = ['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME',
		'ON_NET', 'ORANGE', 'REGULARITY', 'TOP_PACK', 'FREQ_TOP_PACK'] 
		values = [[REGION, TENURE, MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE, DATA_VOLUME,
		ON_NET, ORANGE, REGULARITY, TOP_PACK, FREQ_TOP_PACK]]

		features = pd.DataFrame(values, columns=columns)
		return features

	def preprocess_file(self, data):
		cat_col = ['REGION', 'TOP_PACK']
		def fillna(data):
			for col in data.columns:
				if col in cat_col:
					data[col].fillna('N/A', inplace=True)
				else:
					data[col].fillna(-1, inplace=True)
				return data
		def fit(data):
			data.drop(['user_id', 'TIGO', 'ZONE1', 'ZONE2', 'MRG'], inplace=True, axis=1)
			data = fillna(data)
			return data
		features = fit(data)
		return features


	def predict(self, features):
		pred = self.model.predict(features)
		proba = self.model.predict_proba(features)
		return pred, proba

	def plot_pie_chart(self, probabilities, churn_labels):
		fig = go.Figure(
			data = [go.Pie(
				labels = list(churn_labels),
				values=probabilities[0]
				)]
			)
		fig = fig.update_traces(
			hoverinfo='label+percent',
			textinfo='value',
			textfont_size=15
			)
		return fig

	def filedownload(self, file):
		b64 = base64.b64encode(file.encode()).decode()  # strings <-> bytes conversions
		href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
		return href

	def construct_app(self):
		st.set_page_config(
		    page_title="Expresso Churn",
		    page_icon=":expresso_logo:",
		    layout="wide",
		    initial_sidebar_state="expanded",
		)

		st.markdown(
		    f'''
		        <style>
		        	.reportview-container .main .block-container{{
				        padding-top: {2}rem;
				        padding-right: {5}rem;
				        padding-left: {5}rem;
				        padding-bottom: {3}rem;
				    }}

		            .sidebar .sidebar-content {{
		                width: 375px;
		            }}
		        </style>
		    ''',
		    unsafe_allow_html=True
		)

		st.image('expresso_logo.png', width=100)
		st.title('EXPRESSO CUSTOMER PREDICTION CHURN')
		st.subheader("""
			Expresso is an African telecommunications services company that provides telecommunication services in two African markets: Mauritania and Senegal. Expresso offers a wide range of products and services to meet the needs of customers. This application predicts the likelihood of each Expresso customer “churning,” i.e. becoming inactive and not making any transactions for 90 days.
			""")
		st.markdown(
		    '<hr/>',
		    unsafe_allow_html=True
		)

		st.sidebar.markdown(
		    '<h3 class="header-style">Expresso Churn Classification</h3>',
		    unsafe_allow_html=True
		)
		st.sidebar.markdown(
		    '<hr/>',
		    unsafe_allow_html=True
		)


		uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
		if uploaded_file is not None:
		    data = pd.read_csv(uploaded_file)
		    id_ = data.user_id
		    with st.spinner('Preprocessing uploaded file! please wait...'):
		    	features = self.preprocess_file(data)
		    st.table(features)

		    # with st.spinner('Predicting uploaded file! please wait...'):
		    # 	_, proba = self.predict(features)
		    # st.success('Done!')
		    # churn_file = pd.DataFrame({'user_id': id_, 'CHURN': proba})
		    # churn_file.to_csv('Churn.csv', index=False)

		    # st.set_option('deprecation.showPyplotGlobalUse', False)
		    # st.markdown(self.filedownload(churn_file), unsafe_allow_html=True)
		else:
			features = self.user_inputs()

			if st.sidebar.button('Predict'):
				# with st.spinner('Predict, please wait...'):
				st.subheader('Input features predicted on')
				st.dataframe(features)
				pred, proba = self.predict(features)

				if pred == 0:
					prediction_str = 'No'
				elif pred == 1:
					prediction_str = 'Yes'

				pred_col, proba_col = st.columns(2)
				pred_col.subheader('Prediction')
				if prediction_str == 'Yes':
					pred_col.info(f'{prediction_str}! customer will churn')
				elif prediction_str == 'No':
					pred_col.info(f'{prediction_str}! customer will not churn')

				proba_col.subheader('Prediction Probability Distribution')
				churn_labels = ['No', 'Yes']
				fig = self.plot_pie_chart(proba, churn_labels)
				proba_col.plotly_chart(fig, use_container_width=True)

		return self

def main():
	expresso = ExpressoApp()
	expresso.construct_app()


main()