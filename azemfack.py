import pandas as pd 
import streamlit as st
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg

from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class

url= "https://github.com/241854/DATA-SCIENTIST_CON_R.git"

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

def main():
    st.title("APPLICAZIONE MACHINE LEARNING(AI) INDUSTRY 4.0")
    st.sidebar.write("[Author: Pages Azemfack Aime](%s)"% url)
    st.sidebar.markdown("OBBIETTIVO")
    st.sidebar.markdown("***Questa applicazione ti permette di monitorare e fare le predizione nell'ambito ingegneristica e manutenzione degli impianti. l utente avrà la possibilità***"
                        "***di vissualizzare  graficamente e stimare nel tempo il ciclo di vita di ogni  pezzo meccanico dell impianto dipendentamente dai suei parametri.***"
                        )
    file = st.sidebar.file_uploader(label = "carica i tuoi dati csv", type = ["csv"])
    if file is not None:
        data = pd.read_csv(file)
        st.dataframe(data.head(10))
        
        profile = st.button("Visualizza i tuoi dati")
        if profile:
            profile_df = pandas_profiling.ProfileReport(data)
            st_profile_report(profile_df)
            
        target = st.selectbox("selezionare il target", data.columns)
        task = st.selectbox("selezionare il tipo di Machine Learning", ["Regression","Classification"])
        
        if task == "Regression":
            if st.button("calcola il modello"):
                exo_reg = setup_reg(data, target = target)
                model_reg = compare_models_reg()
                save_model_reg(model_reg, "best_model_regression.pkl")
                st.success("Modello di Regressione generato con successo!!!")
                
                ## resultato
                st.write("Risidui")
                plot_model_reg(model_reg, plot = "residui", save = True)
                st.image("risidui.png")
                
                st.write("feature importance")
                plot_model_reg(model_reg, plot = "feature", save = True)
                st.image("feature importance")
                
                with open("best_reg_model.pkl", "rb") as f:
                    st.download_button("scaricare il modello(pipline)", f, file_name = "best_reg_model.pkl")
                    
                    
        if task == "Classification":
            if st.button("calcola il modello"):
                exp_reg = setup_class(data, target = target)
                model_class = compare_models_class()
                save_model_class(model_class, "best_model_class.pkl")
                st.success("Modello di classificazione generato con successo!!!")
                
                ## resultato
                col5,col6 =st.columns(2)
                with col5:
                    st.write("ROC curve")
                    plot_model_class(model_class, save = True)
                    st.image("AUC.png")
                
                with col6 :
                    st.write("Classification Report")
                    plot_model_class(model_class, plot = "class_report", save = True)
                    st.image("Class Report.png")
                
                col7,col8 =st.columns(2)
                with col7:
                    st.write("Confusion Matrix")
                    plot_model_class(model_class, plot = "confusion matrix", save = True)
                    st.image("Confusion Matrix")
                    
                with col8:
                    st.write("Features Importance")
                    plot_model_class(model_class, plot = "carateristiche", save = True)
                    st.image("Features Importance")
                
                with open("best_class_model.pkl", "rb") as f:
                    st.download_button("scaricare il modello Classificazione", f, file_name = "best_class_model.pkl")
    else:
        st.image("https://hooshio.com/wp-content/uploads/2022/08/b2.jpg")       
if __name__=="__main__":
    main()
