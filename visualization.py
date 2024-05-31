import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#for image
import base64

#utils pkgs
import codecs

# Components pkgs
import streamlit.components.v1 as stc

# importing pages
import home,data,visualization,predict,indicators


with open('style.css') as design:
    source = design.read()

# converting img into python file
def get_img_as_base64(file):
    with open(file, "rb") as g:
        data = g.read()
    return base64.b64encode(data).decode()

instagram = get_img_as_base64("instagram.jpg")
facebook = get_img_as_base64("facebook.jpg")
linkedin = get_img_as_base64("linkedin.jpg")
twitter = get_img_as_base64("twitter.jpg")
location = get_img_as_base64("location.jpg")
phone = get_img_as_base64("phone.jpg")
envelope = get_img_as_base64("envelope.jpg")


def app():
  stock_list = ['NEPSE','HYDROPOWER','MICROFINANCE','RHGCL','CYC','KLBS','AKPL','GHL','NTC','UPPER','NABIL','MLBSL','MLBS','NICA','UNL']

  st.subheader('Select Desired Stock')
  stock = st.selectbox('', stock_list, placeholder='NEPSE')
  str = '.csv'
  path = './CSV/'
  stock = path + stock + str


  # Read the CSV file
  data = pd.read_csv(stock)
  data['Open'] = data['Open'].astype(float)
  data['High'] = data['High'].astype(float)
  data['Low'] = data['Low'].astype(float)
  data['Close'] = data['Close'].astype(float)
  

  df = data['Close']
  st.markdown("""<h2>Closing and Opening Price Visualization</h2>""",unsafe_allow_html=True)
  st.line_chart(data[['Close','Open']] , color=['#0A1FF6','#F64A0A'])
  

  df = pd.read_csv(stock)
  df['Volume'] = df['Volume'].str.replace('.00', '')
  df['Volume'] = df['Volume'].str.replace(',', '')
  df['Volume'] = df.Volume.astype(float)

  st.markdown("""<h2>Volume Visualization</h2>""",unsafe_allow_html=True)
  fig11 = plt.figure(figsize=(25,5))
  plt.plot(df['Volume'], 'blue', label='Volume')
  plt.xlabel('Time')
  plt.ylabel('Volume')
  plt.legend()
  plt.show()
  st.pyplot(fig11)


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

  





