from  src.pipeline import predict_pipeline
import streamlit as st
import sys
import string
from src.exception import CustomException
import pandas as pd
import altair as alt
from src.utils import load_object


html_temp = """
<div style ="background-color:#51E1ED;padding:13px">
<h1 style ="color:black;text-align:center;"> BBC News Classification !🍥</h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)




def main():
    try:
            raw_text = st.text_area("Type your Text here .....",key="text")
            no_punct = raw_text.translate(str.maketrans("", "", string.punctuation))
            col1, col2 = st.columns(2)

            def clear_text():
                st.session_state["text"] = ""

                
                
            if st.button("FIND"):

                with col1:
                     st.success("Original Text")
                     st.write(raw_text)
                     st.success("Prediction Probability")
                     if len(no_punct.split())>=5:
                          PredictPipeline=predict_pipeline.PredictPipeline(raw_text)
                          pred,probability=PredictPipeline.predict()

              
                        

                          st.write(probability) 


                    
                          st.button("Reset", on_click=clear_text)


                          with col2:
                                
                                st.success("Prediction")
                                st.write(pred) 
                                st.success("Probability Graph")
                                fig = alt.Chart(probability).mark_bar().encode(x='Category',y='Probability', color='Category')
                                st.altair_chart(fig,use_container_width=True)
                     else:
                            col2.write("Provide more information ")
                    
                            st.button("REST", on_click=clear_text)


                    
                
                     


    except Exception as e:
        raise CustomException(e,sys)



st.sidebar.subheader("About App")
st.sidebar.text("BBC News Classification App with Streamlit")


st.sidebar.subheader("By")
st.sidebar.text("Rakshit Khajuria - 19bec109")
st.sidebar.text("Prikshit Sharma - 19bec062")






if __name__ == '__main__':
	main()