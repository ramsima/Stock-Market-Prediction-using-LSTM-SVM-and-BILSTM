import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import streamlit as st

#for image
import base64

#utils pkgs
import codecs

# Components pkgs
import streamlit.components.v1 as stc


with open('style.css') as design:
    source = design.read()

# converting img into python file
def get_img_as_base64(file):
    with open(file, "rb") as g:
        data = g.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("bull.jpg")
img1 = get_img_as_base64("team-1.jpg")
img2 = get_img_as_base64("team-2.jpg")
img3 = get_img_as_base64("team-3.jpg")
img4 = get_img_as_base64("team-4.jpg")
img5 = get_img_as_base64("team-4.jpg")
bar_chart = get_img_as_base64("bar-chart-line.jpg")
visualize = get_img_as_base64("visualize.jpg")
data1 = get_img_as_base64("data.jpg")
instagram = get_img_as_base64("instagram.jpg")
facebook = get_img_as_base64("facebook.jpg")
linkedin = get_img_as_base64("linkedin.jpg")
twitter = get_img_as_base64("twitter.jpg")
location = get_img_as_base64("location.jpg")
phone = get_img_as_base64("phone.jpg")
envelope = get_img_as_base64("envelope.jpg")


def get_fill_color1(label):
    if label>=1:
        return 'rgba(0,255,0,0.6)'
    else:
       return 'rgba(255,0,0,0.6)' 



def app():
  stock_list = ['NEPSE','HYDROPOWER','MICROFINANCE','RHGCL','CYC','KLBS','AKPL','GHL','NTC','UPPER','NABIL','MLBSL','MLBS','NICA','UNL']

  st.header('Select desired Stock')
  stock = st.selectbox('', stock_list, placeholder='NEPSE')
  str = '.csv'
  path = './CSV/'
  stock = path+stock + str


  # Read the CSV file
  data = pd.read_csv(stock)
  #data = data.set_index('Date')
  df = pd.read_csv(stock)

  indicator_list = ['Moving Average','Exponential Moving Average','Bollinger Band','Ichimoku Cloud']

  st.header('Stock Market Solution')
  indicator_selected = st.selectbox('Select desired Stock', indicator_list, placeholder='Moving Average')

  if(indicator_selected == "Moving Average"):
    st.subheader('Price vs MA50')
    df1 = pd.DataFrame(columns = ['Close', 'Moving Average 50 days', 'Moving Average 100 days','Moving Average 200 days'])
    ma_50_days = data.Close.rolling(50).mean()

    df1['Moving Average 50 days'] = ma_50_days
    df1['Close'] = data['Close']

    st.line_chart(df1[['Close','Moving Average 50 days']],height=500, color=["#5BEA0A", "#F60AEF"])


    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    df1['Moving Average 100 days'] = ma_100_days

    st.line_chart(df1[['Close','Moving Average 50 days', 'Moving Average 100 days']],color=["#5BEA0A", "#F60AEF","#F64A0A"],height=500)



    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()

    df1['Moving Average 200 days'] = ma_200_days

    st.line_chart(df1[['Close','Moving Average 200 days', 'Moving Average 100 days']],height=500, color=["#5BEA0A", "#F60AEF","#F64A0A"])


  elif(indicator_selected == "Exponential Moving Average"):

    df2 = pd.DataFrame(columns = ['Close', 'Exponential Moving Average 20 days', ' Exponential Moving Average 50 days'])

    df2['Close'] = data['Close']
    df = data['Close']


    # Calculate the EMA with a period of 20 days and a weighting factor of 0.2
    df2['Exponential Moving Average 20 days'] = df.ewm(span=20, adjust=False, min_periods=20).mean()


    st.subheader('Price vs EMA20')

    st.line_chart(df2[['Close','Exponential Moving Average 20 days']], color=["#5BEA0A", "#0000FF"], height=500)
    


    df1 = data['Close']
    # Calculate the EMA with a period of 50 days and a weighting factor of 0.2
    df2['Exponential Moving Average 50 days'] = df1.ewm(span=50, adjust=False, min_periods=50).mean()

    st.subheader('Price vs EMA50')

    st.line_chart(df2[['Close','Exponential Moving Average 50 days']], color=["#5BEA0A", "#F60AEF"], height=500)

  elif(indicator_selected == "Bollinger Band"):

    st.subheader("Bollinger Band")
    df = pd.DataFrame(data[['Date','Close']])

    # compute the datapoints for moving average, upper band and the lower band

    def bollinger_band(price, length = 20, num_stdev = 2):
      mean_price = price.rolling(length).mean()
      stdev = price.rolling(length).std()
      upband = mean_price + num_stdev*stdev
      dwnband = mean_price - num_stdev*stdev
    
      return np.round(mean_price, 3), np.round(upband, 3), np.round(dwnband, 3)
    
    df['Moving_avg'], df['Upper_band'], df['Lower_band'] = bollinger_band(df['Close']) 
    st.line_chart(df[['Close','Moving_avg','Upper_band','Lower_band']],height=500)

  elif(indicator_selected == "Ichimoku Cloud"):
     
     #Conversion line
     hi_val = df['High'].rolling(window=9).max()
     low_val = df['Low'].rolling(window=9).min()
     df['Conversion'] = (hi_val + low_val) /2

     # Base line
     hi_val2 = df['High'].rolling(window=26).max()
     low_val2 = df['Low'].rolling(window=26).min()
     df['Baseline'] = (hi_val2 + low_val2) /2

     #leading span A
     df['SpanA'] = (df['Conversion']+ df['Baseline'])/2

     #leading span B
     hi_val3 = df['High'].rolling(window=52).max()
     low_val3 = df['Low'].rolling(window=52).min()
     df['SpanB'] = (hi_val3 + low_val3) /2

     #lagging span
     df['Lagging'] = df['Close'].shift(-26)
  
     #create Candlestick
     candle = go.Candlestick(x=df.index, open=df['Open'].shift(-26),high=df['High'].shift(-26),low=df['Low'].shift(-26),close=df['Close'].shift(-26), name='Candlestick')
     df1 = df

     fig = go.Figure()
     fig = make_subplots(rows=2,cols=1,shared_xaxes="columns", shared_yaxes="rows", column_width=[1], row_heights=[0.6,0.1], subplot_titles=["Candlestick", "Volume"])

     df['label'] = np.where(df['SpanA'] > df['SpanB'], 1,0)
     df['group'] = df['label'].ne(df['label'].shift()).cumsum()

     df = df.groupby('group')

     dfs = []
     for name, data in df:     
      dfs.append(data)

     for df in dfs:
      fig.add_traces(go.Scatter(x=df.index, y= df.SpanA, line=dict(color='rgba(0,250,0,0)'), name ='Cloud'))
      fig.add_traces(go.Scatter(x=df.index, y= df.SpanB, line=dict(color='rgba(250,0,0,0)'), fill='tonexty', fillcolor = get_fill_color1(df['label'].iloc[0]), name='Cloud'))

     baseline = go.Scatter(x=df1.index, y=df1['Baseline'].shift(-26), line=dict(color='red',width = 1),name="Baseline")
     conversion = go.Scatter(x=df1.index, y=df1['Conversion'].shift(-26), line=dict(color='blue',width = 1),name="Conversion")
     lagging = go.Scatter(x=df1.index, y=df1['Lagging'].shift(-26), line=dict(color='green',width = 1),name="Lagging")
     span_a = go.Scatter(x=df1.index, y=df1['SpanA'].shift(0), line=dict(color='green',width = 1),name="SpanA")
     span_b = go.Scatter(x=df1.index, y=df1['SpanB'].shift(0), line=dict(color='red',width = 1),name="SpanB")

     fig.add_trace(candle)
     fig.add_trace(baseline)
     fig.add_trace(conversion)
     fig.add_trace(lagging)
     fig.add_trace(span_a)
     fig.add_trace(span_b)

     fig.add_trace(go.Bar(x=df1.index, y=df1['Volume'], name='Volume'),col=1,row=2)
     fig.update_layout(width=1200,height=800,xaxis_rangeslider_visible=False,showlegend=True)

     fig.show()


  with open('style1.css') as design:
    source = design.read()

    
  stc.html(f"""
            <!DOCTYPE html>             
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Document</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css">
             
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
             
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.2/css/fontawesome.min.css">
             
            <style>
            {source}
            </style>
            </head>
            <body>  

    <!-- ======= Contact Section ======= -->
    <section id="contact" class="contact">
      <div class="container">

        <div class="section-title" data-aos="fade-up">
          <h2></h2>
        </div>

        <div class="row">

          <div class="col-lg-4 col-md-6" data-aos="fade-up" data-aos-delay="100">
            <div class="contact-about">
              <h3>Nepal Stock Solutions</h3>
              <p>Discover the Power of Investing</p>
              <div class="social-links">
                <i class="bi bi-twitter"><img style="padding-right:10px" height=35px src="data:bull/jpg;base64,{twitter}" alt="h"></i>
                <i class="bi bi-facebook"><img style="padding-right:10px" height=35px src="data:bull/jpg;base64,{facebook}" alt="h"></i>
                <i class="bi bi-instagram"><img style="padding-right:10px" height=35px src="data:bull/jpg;base64,{instagram}" alt="h"></i>
                <i class="bi bi-linkedin"><img style="padding-right:10px" height=35px src="data:bull/jpg;base64,{linkedin}" alt="h"></i>
              </div>
            </div>
          </div>

          <div class="col-lg-3 col-md-6 mt-4 mt-md-0" data-aos="fade-up" data-aos-delay="200">
            <div class="info">
              <div>
                <i class="ri-map-pin-line"><img style="padding-right:20px" height=35px src="data:bull/jpg;base64,{location}" alt="h"></i>
                <p>Libali-6, Bhaktapur<br>Nepal</p>
              </div>

              <div>
                <i class="ri-mail-send-line"><img style="padding-right:20px" height=35px src="data:bull/jpg;base64,{envelope}" alt="h"></i>
                <p>info@NepalStockSolutions.com</p>
              </div>

              <div>
                <i class="ri-phone-line"><img style="padding-right:20px" height=35px src="data:bull/jpg;base64,{phone}" alt="h"></i>
                <p>+977 9841 **** **</p>
              </div>

            </div>
          </div>

          <div class="col-lg-5 col-md-12" data-aos="fade-up" data-aos-delay="300">
            <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d883.375782876981!2d85.43842601964027!3d27.67084043623559!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x39eb0553197e9d2f%3A0x73b807bbe91781af!2sLiwali%2C%20Bhaktapur%2044800!5e0!3m2!1sen!2snp!4v1686130436489!5m2!1sen!2snp" width="400" height="200" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
          </div>

        </div>

      </div>
    </section><!-- End Contact Section -->
    </footer>


</body>
</html>

    
    
    """ ,scrolling = False,height=340)




  
  
