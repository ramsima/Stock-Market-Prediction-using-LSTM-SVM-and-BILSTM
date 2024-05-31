import numpy as np
import numpy
from numpy import array
import pandas as pd
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as stc
with open('style1.css') as design:
  source = design.read()

#for image
import base64

#utils pkgs
import codecs

# Components pkgs
import streamlit.components.v1 as stc

# importing pages
import home,data,visualization,predict,indicators


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




# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


def app():

  stock_list = ['NEPSE','HYDROPOWER','MICROFINANCE','AKPL','NTC','UPPER','NABIL','MLBSL','MLBS','NICA','UNL']

  st.header('Select Desired Stock')
  stock = st.selectbox('', stock_list, placeholder='NEPSE')
  str = '.csv'
  path = './CSV/'
  stock = path + stock + str


  # Read the CSV file
  data = pd.read_csv(stock)

  model_list = ['LSTM','BILSTM','SVM']


  model_path1 = 'D:\project\Stock-MArket-Forecasting-master\Stock-MArket-Forecasting-master\Model\\'
  model_path2 = '.keras'

  st.subheader('MODEL')
  model_name = st.selectbox('Select desired model', model_list, placeholder='SVM')



  if (model_name == "LSTM" ):

    model_true_path = model_path1 + model_name + model_path2

    model = load_model(model_true_path)
    ### Delete all the null values in stock data
    data.dropna(inplace=True)

    data=data.reset_index()['Close']

    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(np.array(data).reshape(-1,1))

    ##splitting dataset into train and test split
    training_size = int(len(data)*0.7)
    test_size=len(data)-training_size
    data_train = data[0:training_size,:]
    data_test = data[training_size:len(data),:]

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(data_train, time_step)
    X_test, y_test = create_dataset(data_test, time_step)

    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    # For training and validation loss of LSTM model

    #loss=pd.DataFrame(columns=['loss','val_loss'])

    #loss = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1).history

    #st.subheader('Training Loss vs Validation Loss')
    #fig1112 = plt.figure(figsize=(20,7))
    #plt.plot(loss['loss'], 'r', label='Training Loss')
    #plt.plot(loss['val_loss'], 'b', label='Validation Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()
    #st.pyplot(fig1112)
    

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    
    

    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    #For error Calculations
    y_test.reshape(1,-1)
    original_y_test=scaler.inverse_transform(pd.DataFrame(y_test))



    y_train.reshape(-1,1)

    y_train = scaler.inverse_transform(pd.DataFrame(y_train))

    dataaa = pd.read_csv(stock)

    df1 = pd.DataFrame(columns = ['Close', 'train_predict', 'test_predict'])


    # shift train predictions for plotting
    look_back=100
    trainPredictPlot = numpy.empty_like(data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(data)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(data)-1, :] = test_predict
    # plot baseline and predictions

    df1['train_predict'] = trainPredictPlot.tolist()
    df1['test_predict'] = testPredictPlot.tolist()
    df1['Close'] = dataaa['Close']

    df1['train_predict'].dropna()

    st.subheader('Prediction vs Original Price')
    fig12 = plt.figure(figsize=(20,7))
    plt.plot(df1['Close'], 'r', label='Closing Price')
    plt.plot(trainPredictPlot, 'g', label='Training Predictions')
    plt.plot(testPredictPlot, 'b', label='Testing Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig12)

    st.subheader('Prediction vs Validation')
    fig13 = plt.figure(figsize=(20,7))
    plt.plot(trainPredictPlot, 'g', label='Training Predictions')
    plt.plot(testPredictPlot, 'b', label='Testing Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig13)

    x_input=data_test[-time_step:].reshape(1,-1)
    x_input.shape

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=100
    i=0

    while(i<30):
         if(len(temp_input)>100):
              #print(temp_input)
              x_input = np.array(temp_input[1:])
              print("{} day input {}".format(i,x_input))
              x_input=x_input.reshape(1,-1)
              x_input = x_input.reshape((1, n_steps, 1))
              #print(x_input)
              yhat = model.predict(x_input, verbose=0)
              print("{} day output {}".format(i,yhat))
              temp_input.extend(yhat[0].tolist())
              temp_input=temp_input[1:]
              #print(temp_input)
              lst_output.extend(yhat.tolist())
              i=i+1

         else:
              
              x_input = x_input.reshape((1,n_steps,1))
              yhat = model.predict(x_input, verbose=0)
              print(yhat[0])
              temp_input.extend(yhat[0].tolist())
              print(len(temp_input))
              lst_output.extend(yhat.tolist())
              i=i+1


    day_new=np.arange(1,501)
    day_pred=np.arange(501,531)


    st.subheader('30 days prediction')

    fig14 = plt.figure(figsize=(20,4))
    plt.plot(day_new,scaler.inverse_transform(data[-500:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))

    st.pyplot(fig14)

    #st.write(scaler.inverse_transform(lst_output))
    

    # Error Calculations
    st.subheader("Errors in LSTM")

    rmse = np.sqrt(mean_squared_error(original_y_test, test_predict))
    mae= mean_absolute_error(original_y_test, test_predict)
    mape = np.mean(np.abs((original_y_test - test_predict) / original_y_test)) * 100
    r_squared = r2_score(original_y_test, test_predict)
    with open('style2.css') as design:
      source1 = design.read()

    stc.html(f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                <style>
                  {source1}
                </style>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                </head>
                <body> 
                <div class=rmse>
                <h4>1. Root Mean Squared Error</h4>
             
                
                <h5>Range of RMSE is 0 to ∞ </h5>
                <h5>Root Mean Squared Error = {rmse} </h5>

                </div>

                <div class=mae>
                <h4>2. Mean Absolute Error</h4>
    
                
                <h5>Range of MAE is 0 to ∞ </h5>
                <h5>Mean Absolute Error = {mae} </h5>

                </div>
                <div class=mape>
                <h4>3. Mean Absolute Percentage Error</h4>
        
                
                <h5>Range of MAE is 0% to ∞% but generally it is 0% to 100%</h5>
                <h5>Mean Absolute Percentage Error = {mape}% </h5>

                </div>  

                <div class=r2>
                <h4>4. R Squared Error</h4>
                
                <h5>Range of MAE is 0 to 1 </h5>
                <h5>Mean Absolute Error = {r_squared} </h5>

                </div>        
        
                </body>
                </html>

    """,scrolling=False,height=650)
  
    
    

  elif(model_name == "SVM"):
       
       # load
       #with open('SVM.keras', 'rb') as f:            
       #   model1 = pickle.load(f)

       df = pd.read_csv(stock)

       # Convert the Date column to a datetime object
       df['Date'] = pd.to_datetime(df['Date'])

       # Sort the dataframe by date
       df = df.sort_values('Date')

       # Convert '--' to 0 in the 'Percent Change' column
       df['Percent Change'] = df['Percent Change'].str.replace('--', '0')
       # Convert '%' to '' in the 'Percent Change' column
       df['Percent Change'] = df['Percent Change'].str.replace('%', '')

       # Convert 'Percent Change' column to float
       df['Percent Change'] = df['Percent Change'].astype(float)

       # Create features and target variables
       X = df[['Open', 'High', 'Low', 'Percent Change']]
       y = df['Close']

       # Split the data into train and test sets
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       # Create an SVR model
       svr = SVR()

       # Fit the model
       svr.fit(X_train, y_train)

       # Make predictions on the test set
       y_pred = svr.predict(X_test)




       # Calculate RMSE, MAE, and R2 for training set
       train_rmse = np.sqrt(mean_squared_error(y_train, svr.predict(X_train)))
       train_mae = mean_absolute_error(y_train, svr.predict(X_train))
       train_r2 = r2_score(y_train, svr.predict(X_train))

       # Calculate RMSE, MAE, and R2 for test set
       test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
       test_mae = mean_absolute_error(y_test, y_pred)
       test_r2 = r2_score(y_test, y_pred)

       # Print evaluation metrics for training set
       print("Training RMSE:", train_rmse)
       print("Training MAE:", train_mae)
       print("Training R2:", train_r2)

       # Print evaluation metrics for test set
       print("Test RMSE:", test_rmse)
       print("Test MAE:", test_mae)
       print("Test R2:", test_r2)

       # Forecast close prices for the upcoming week
       last_day = df['Date'].max()
       forecast_dates = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=5, freq='D')
       forecast_features = df[['Open', 'High', 'Low', 'Percent Change']].tail(1).values

       predictions = []
       for _ in range(5):
           prediction = svr.predict(forecast_features)[0]
           predictions.append(prediction)
           forecast_features = np.roll(forecast_features, -1, axis=0)
           forecast_features[-1] = [df['Open'].iloc[-1], df['High'].iloc[-1], df['Low'].iloc[-1], predictions[-1]]

       # Create a DataFrame for the predictions
       df_predictions = pd.DataFrame(predictions, columns=['Close'])
       df_predictions['Date'] = forecast_dates

       df_original=pd.DataFrame(columns=['Close'])
       df_original['Close'] = df['Close']
       

       day_new=np.arange(1,101)
       day_pred=np.arange(101,106)


       st.subheader('FIVE days prediction')

       fig14 = plt.figure(figsize=(20,4))
       plt.plot(day_new,df_original[-100:])
       plt.plot(day_pred,df_predictions['Close'])

       st.pyplot(fig14)

       # Print the dataframe
       print(df_predictions)

       st.subheader('Predictions for 5 days')

       st.write(df_predictions)

       st.subheader('Errors in SVM')

       with open('style2.css') as design:
        source1 = design.read()

       stc.html(f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                <style>
                  {source1}
                </style>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                </head>
                <body> 
                <div class=rmse>
                <h4>1. Root Mean Squared Error</h4>
             
                
                <h5>Range of RMSE is 0 to ∞ </h5>
                <h5>Root Mean Squared Error = {test_rmse} </h5>

                </div>

                <div class=mae>
                <h4>2. Mean Absolute Error</h4>
    
                
                <h5>Range of MAE is 0 to ∞ </h5>
                <h5>Mean Absolute Error = {test_mae} </h5>

                </div>
               

                <div class=r2>
                <h4>3. R Squared Error</h4>
                
                <h5>Range of MAE is 0 to 1 </h5>
                <h5>Mean Absolute Error = {test_r2} </h5>

                </div>        
        
                </body>
                </html>

       """,scrolling=False,height=480)
       #st.write("Training RMSE:", train_rmse)
       #st.write("Training MAE:", train_mae)
       #st.write("Training R2:", train_r2)
       #st.write("RMSE_Score:", test_rmse)
       #st.write("MAE_Score:", test_mae)
       #st.write("R2_Score:", train_r2)


  elif(model_name == "BILSTM"):
       
       data = pd.read_csv(stock)

       model_true_path = model_path1 + model_name + model_path2
       model = load_model(model_true_path)
       ### Delete all the null values in stock data
       data.dropna(inplace=True)

       data=data.reset_index()['Close']

       scaler = MinMaxScaler(feature_range=(0,1))
       data = scaler.fit_transform(np.array(data).reshape(-1,1))

       ##splitting dataset into train and test split
       training_size = int(len(data)*0.7)
       test_size=len(data)-training_size
       data_train = data[0:training_size,:]
       data_test = data[training_size:len(data),:]

       # reshape into X=t,t+1,t+2,t+3 and Y=t+4
       time_step = 100
       X_train, y_train = create_dataset(data_train, time_step)
       X_test, y_test = create_dataset(data_test, time_step)

       X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
       X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

       ### Lets Do the prediction and check performance metrics
       train_predict=model.predict(X_train)
       test_predict=model.predict(X_test)

       # For Training and Validation loss of BILSTM model

       #loss=pd.DataFrame(columns=['loss','val_loss'])

       #loss = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1).history

       #st.subheader('Training Loss vs Validation Loss')
       #fig11121 = plt.figure(figsize=(20,7))
       #plt.plot(loss['loss'], 'r', label='Training Loss')
       #plt.plot(loss['val_loss'], 'b', label='Validation Loss')
       #plt.xlabel('Epochs')
       #plt.ylabel('Loss')
       #plt.legend()
       #plt.show()
       #st.pyplot(fig11121)

       train_predict=scaler.inverse_transform(train_predict)
       test_predict=scaler.inverse_transform(test_predict)

       #For error Calculations
       y_test.reshape(1,-1)
       original_y_test=scaler.inverse_transform(pd.DataFrame(y_test))

       y_train.reshape(-1,1)

       y_train = scaler.inverse_transform(pd.DataFrame(y_train))

       dataaa = pd.read_csv(stock)

       df1 = pd.DataFrame(columns = ['Close', 'train_predict', 'test_predict'])


       # shift train predictions for plotting
       look_back=100
       trainPredictPlot = numpy.empty_like(data)
       trainPredictPlot[:, :] = np.nan
       trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
       # shift test predictions for plotting
       testPredictPlot = numpy.empty_like(data)
       testPredictPlot[:, :] = numpy.nan
       testPredictPlot[len(train_predict)+(look_back*2)+1:len(data)-1, :] = test_predict
       # plot baseline and predictions

       df1['train_predict'] = trainPredictPlot.tolist()
       df1['test_predict'] = testPredictPlot.tolist()
       df1['Close'] = dataaa['Close']

       df1['train_predict'].dropna()

       st.subheader('Prediction vs Original Price')
       

      
       fig12 = plt.figure(figsize=(20,7))
       plt.plot(df1['Close'], 'r', label='Closing Price')
       plt.plot(trainPredictPlot, 'g', label='Training Predictions')
       plt.plot(testPredictPlot, 'b', label='Testing Predictions')
       plt.xlabel('Time')
       plt.ylabel('Price')
       plt.legend()
       plt.show()
       st.pyplot(fig12)

       st.subheader('Prediction vs Validation')
       fig13 = plt.figure(figsize=(20,7))
       plt.plot(trainPredictPlot, 'g', label='Training Predictions')
       plt.plot(testPredictPlot, 'b', label='Testing Predictions')
       plt.xlabel('Time')
       plt.ylabel('Price')
       plt.legend()
       plt.show()
       st.pyplot(fig13)

       x_input=data_test[-time_step:].reshape(1,-1)
       x_input.shape

       temp_input=list(x_input)
       temp_input=temp_input[0].tolist()

       lst_output=[]
       n_steps=100
       i=0

       while(i<30):
         if(len(temp_input)>100):
              #print(temp_input)
              x_input = np.array(temp_input[1:])
              print("{} day input {}".format(i,x_input))
              x_input=x_input.reshape(1,-1)
              x_input = x_input.reshape((1, n_steps, 1))
              #print(x_input)
              yhat = model.predict(x_input, verbose=0)
              print("{} day output {}".format(i,yhat))
              temp_input.extend(yhat[0].tolist())
              temp_input=temp_input[1:]
              #print(temp_input)
              lst_output.extend(yhat.tolist())
              i=i+1

         else:
              
              x_input = x_input.reshape((1,n_steps,1))
              yhat = model.predict(x_input, verbose=0)
              print(yhat[0])
              temp_input.extend(yhat[0].tolist())
              print(len(temp_input))
              lst_output.extend(yhat.tolist())
              i=i+1


       day_new=np.arange(1,501)
       day_pred=np.arange(501,531)


       st.subheader('30 days prediction')

       fig14 = plt.figure(figsize=(20,4))
       plt.plot(day_new,scaler.inverse_transform(data[-500:]))
       plt.plot(day_pred,scaler.inverse_transform(lst_output))

       st.pyplot(fig14)

       #st.write(scaler.inverse_transform(lst_output))


       st.subheader("Errors in BILSTM")


       rmse = np.sqrt(mean_squared_error(original_y_test, test_predict))
       mae= mean_absolute_error(original_y_test, test_predict)
       mape = np.mean(np.abs((original_y_test - test_predict) / original_y_test)) * 100
       r_squared = r2_score(original_y_test, test_predict)

       with open('style2.css') as design:
        source1 = design.read()

        stc.html(f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                <style>
                  {source1}
                </style>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                </head>
                <body> 
                <div class=rmse>
                <h4>1. Root Mean Squared Error</h4>
             
                
                <h5>Range of RMSE is 0 to ∞ </h5>
                <h5>Root Mean Squared Error = {rmse} </h5>

                </div>

                <div class=mae>
                <h4>2. Mean Absolute Error</h4>
    
                
                <h5>Range of MAE is 0 to ∞ </h5>
                <h5>Mean Absolute Error = {mae} </h5>

                </div>
                <div class=mape>
                <h4>3. Mean Absolute Percentage Error</h4>
        
                
                <h5>Range of MAE is 0% to ∞% but generally it is 0% to 100%</h5>
                <h5>Mean Absolute Percentage Error = {mape}% </h5>

                </div>  

                <div class=r2>
                <h4>4. R Squared Error</h4>
                
                <h5>Range of MAE is 0 to 1 </h5>
                <h5>Mean Absolute Error = {r_squared} </h5>

                </div>        
        
                </body>
                </html>

          """,scrolling=False,height=650)


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



       








    


    




    


  