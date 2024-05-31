import streamlit as st

from streamlit_option_menu import option_menu

import home,data,visualization,predict,indicators,contacts,Team
with open('style.css') as design:
    st.markdown(f'<style>{design.read()}</style>',unsafe_allow_html=True)



class MultiApp:
    def run():
        
                
        selected = option_menu(
            menu_title='Nepal Stock Solutions',
            menu_icon='graph-up-arrow',
            options=['Home','Team','Data','Visualize','Predict','Indicators'],
            default_index=0,
            orientation="horizontal",
            styles= {
                "container": {"padding": "10px", "background-color": "white"},
                "icon": {"color": "white", "font-size": "15px"},
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "green",
                }
            }
                
            )

        
        if selected == "Home":
            home.app() 
        if selected == "Team":
            Team.app() 
        if selected == "Contacts":
            contacts.app() 
        if selected == "Data":
            data.app()        
        if selected == "Visualize":
            visualization.app()  
        if selected == "Predict":
            predict.app()  
        if selected == "Indicators":
            indicators.app()    
             
    run()            
         
