from  src.pipeline import predict_pipeline
import streamlit as st
import sys
import string
from src.exception import CustomException

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
            col1, col2, col3 = st.columns(3)

            def clear_text():
                st.session_state["text"] = ""
                
                
            if st.button("FIND"):
                if len(no_punct.split())>=5:

                    PredictPipeline=predict_pipeline.PredictPipeline(raw_text)
                    pred,probability=PredictPipeline.predict()


                    col2.write(pred)
                    col2.write(probability)
                
                    st.button("REST", on_click=clear_text)
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