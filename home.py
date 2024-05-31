import streamlit as st

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


def app():

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
  

  
            <!-- ======= Hero Section ======= -->
            <section id="hero" class="d-flex align-items-center">

            <div class="container">
            <div class="row">
                <div class="col-lg-6 pt-5 pt-lg-0 order-2 order-lg-1 d-flex flex-column justify-content-center">
                <h1 data-aos="fade-up">
                    Discover the Power of Investing</h1>
                <div data-aos="fade-up" data-aos-delay="800">
                    
                </div>
                </div>
                <div class="col-lg-6 order-1 order-lg-2 hero-img" data-aos="fade-left" data-aos-delay="200">
                <img src="data:bull/jpg;base64,{img}" class="img-fluid animated" alt="h">
                </div>
            </div>
            </div>

  </section><!-- End Hero -->

  <main id="main">
    <!-- ======= About Us Section ======= -->
    <section id="about" class="about">
      <div class="container">

        <div class="section-title" data-aos="fade-up">
          <h2>About Us</h2>
        </div>

        <div class="row content">
          <div class="col-lg-6" data-aos="fade-up" data-aos-delay="150">
            <p>
              Welcome to our cutting-edge platform, where artificial intelligence meets the exciting world of stock
              market prediction. At Nepal Stock Solutions
            </p>
            <ul>
              <li><i class="ri-check-double-line"></i> We pride ourselves on harnessing the power of advanced AI models to provide you with accurate and insightful forecasts for the ever-changing stock market.</li>
              <li><i class="ri-check-double-line"></i>Make informed investment decisions and navigate the complexities
                of the stock market with confidence</li>
            </ul>
          </div>
          <div class="col-lg-6 pt-4 pt-lg-0" data-aos="fade-up" data-aos-delay="300">
            <p>
              Join us on this revolutionary journey, where technology meets finance, and unlock the potential of your
              investments.
            </p>
          </div>
        </div>

      </div>
    </section><!-- End About Us Section -->

   
    <!-- ======= Services Section ======= -->
    <section id="services" class="services">
      <div class="container">

        <div class="section-title" data-aos="fade-up">
          <h2>Services</h2>
        </div>

        <div class="row">
          <div class="col-md-6 col-lg-3 d-flex align-items-stretch mb-5 mb-lg-0">
            <div class="icon-box" data-aos="fade-up" data-aos-delay="100">
              <div class="icon"><i class="bi bi-bar-chart-line"><img height=30px src="data:bar-chart/jpg;base64,{bar_chart}"></i></div>
              <h4 class="title">Stock Prediction</h4>
              <p class="description">Predict stock prices for upcoming days with help of Machile learning models
              </p>
            </div>
          </div>

          <div class="col-md-6 col-lg-3 d-flex align-items-stretch mb-5 mb-lg-0">
            <div class="icon-box" data-aos="fade-up" data-aos-delay="200">
              <div class="icon"><i class="bx bx-file"><img height=30px src="data:bar-chart/jpg;base64,{visualize}"></i></div>
              <h4 class="title">Data Visualization</h4>
              <p class="description">Visualize your stock data to see how it has been performing in a graphical manner</p>
            </div>
          </div>

          <div class="col-md-6 col-lg-3 d-flex align-items-stretch mb-5 mb-lg-0">
            <div class="icon-box" data-aos="fade-up" data-aos-delay="400">
              <div class="icon"><i class="bx bx-world"><img height=30px src="data:bar-chart/jpg;base64,{data1}"></i></div>
              <h4 class="title"><a href="https://nepsealpha.com/nepse-data" target="_blank">Nepse Data</a></h4>
              <p class="description">Get data for any stock to visualize or predict and gain upperhand over others</p>
            </div>
          </div>

        </div>

      </div>
    </section><!-- End Services Section -->


    <!-- ======= F.A.Q Section ======= -->
    <section id="faq" class="faq">
      <div class="container">

        <div class="section-title" data-aos="fade-up">
          <h2>Frequently Asked Questions</h2>
        </div>

        <div class="row faq-item d-flex align-items-stretch" data-aos="fade-up" data-aos-delay="100">
          <div class="col-lg-5">
            <i class="ri-question-line"></i>
            <h4>Is the prediction accurate?</h4>
          </div>
          <div class="col-lg-7">
            <p>
              The predictions made do not guarantee the accuracy but gives a general idea of what the stock price can be
              in general scenario. So, common assumptions and market analysis must be done to that might affect general
              conditons.
            </p>
          </div>
        </div><!-- End F.A.Q Item-->

        <div class="row faq-item d-flex align-items-stretch" data-aos="fade-up" data-aos-delay="200">
          <div class="col-lg-5">
            <i class="ri-question-line"></i>
            <h4>Who will be responsible if I go in loss?</h4>
          </div>
          <div class="col-lg-7">
            <p>
              We only provide people with the tool that can help predict assume the prices and any loss or profit
              incurred is not in account for Nepal Stock Solutions. Nepal Stock Solutions takes no responsibility.
            </p>
          </div>
        </div><!-- End F.A.Q Item-->
      </div>
    </section><!-- End F.A.Q Section -->

      </main><!-- End #main -->

    
    
    <!-- ======= Contact Section ======= -->
    <section id="contact" class="contact">
      <div class="container">

        <div class="section-title" data-aos="fade-up">
          <h2>Contact Us</h2>
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

    
    
    """ ,scrolling = False,height=2350)