import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import altair as alt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





def plot_confusion_matrix(cm,
                          target_names,
                          title='',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    


Airways_df_train = pd.read_csv("C:/Users/dell/Desktop/cmse 830 final/train.csv")
Airways_df_test = pd.read_csv("C:/Users/dell/Desktop/cmse 830 final/test.csv")

# remove redundant and nulls
cols = Airways_df_train.columns
del Airways_df_train[cols[0]]
del Airways_df_train[cols[1]]
Airways_df_train = Airways_df_train.dropna()
del Airways_df_test[cols[0]]
del Airways_df_test[cols[1]]
Airways_df_test = Airways_df_test.dropna()

# A dummy instance
airways_df_train = Airways_df_train
airways_df_test = Airways_df_test

# Analyse each column
cols = Airways_df_train.columns
# categories histogram histogram
cat_cols = [0,1,3,4,22]
cat_cols_0 = Airways_df_train[cols[0]].unique()
cat_cols_1 = Airways_df_train[cols[1]].unique()
cat_cols_3 = Airways_df_train[cols[3]].unique()
cat_cols_4 = Airways_df_train[cols[4]].unique()
cat_cols_22 = Airways_df_train[cols[22]].unique()
cols_rankings = [6,7,8,9,10,11,12,13,14,15,16,17,18,19]


def plot_confusion_matrix(cm,
                          target_names,
                          title='',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    st.pyplot(fig)



siteHeader = st.container()
dataExploration = st.container()
newFeatures = st.container()

with siteHeader:
    st.title('Passenger Sentiment Analysis')

with dataExploration:
    st.header('Airline Passenger Review Data')
    



with st.sidebar:
    st.write("Which option do you want to select ?")
    
    
    option= st.radio(label='select one',options=['Data Summary','EDA','Models and Outputs','Explore'])
    
    
if(option=="Data Summary"):
    
    #image = Image.open("/Users/sandeepvemulapalli/Desktop/fitness.png")
    #st.image(image,caption="Image taken from Spanish Fan share- Google")
    
    st.subheader("Project Organization: ")
    st.write("1: Dataset Description")
    st.write("2: Exploring Data through visual means")
    st.write("3: Performance analysis of predicting customer satisfaction through ML Algorithms")
    st.write("4: Customizing user own data")
    
    st.write("-------------------------------------------------------------------------------------------------")
    st.header("Description")
    st.write("This dataset contains an airline passenger satisfaction survey. What factors are highly correlated to a satisfied (or dissatisfied) passenger? Can you predict passenger satisfaction?")
    st.write("the data set about survey of Passenger Satisfaction in US airline. \n"
             "Generally, classification method is when the output is a category. \n"
             "For example in this case, from the survey has been done, how do customers feel “satisfied” or “neutral or dissatisfied”. For more detail, let’s see what’s in this data set!")
    
    st.dataframe(Airways_df_train)
    st.markdown("1.Satisfaction : Airline satisfaction level(Satisfaction, neutral or dissatisfaction).")
    st.markdown("2.Age : The actual age of the passengers.")
    st.markdown("3.Gender : Gender of the passengers (Female, Male). ")
    st.markdown(" 4.Type of Travel : Purpose of the flight of the passengers (Personal Travel, Business Travel).")
    st.markdown("5.Class : Travel class in the plane of the passengers (Business, Eco, Eco Plus).")
    st.markdown(" 6.Customer Type : The customer type (Loyal customer, disloyal customer).")
    st.markdown("7.Flight Distance : The flight distance of this journey.")
    st.markdown("8.Inflight Wifi Service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5). ")
    st.markdown("9.Ease of Online Booking : Satisfaction level of online booking" )
    st.markdown("10.Inflight Service : Satisfaction level of inflight service.")
    st.markdown("11.Online Boarding : Satisfaction level of inflight service.")
    st.markdown("12.Inflight Entertainment: Satisfaction level of inflight entertainment.")
    st.markdown("13.Food and drink: Satisfaction level of Food and drink.")
    st.markdown("14.Seat comfort: Satisfaction level of Seat comfort..")
    st.markdown("15.On-board service: Satisfaction level of On-board service.")
    st.markdown("16.Leg room service: Satisfaction level of Leg room service.")
    st.markdown("17.Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient.")
    st.markdown("18.Baggage handling: Satisfaction level of baggage handling.")
    st.markdown("19.Gate location: Satisfaction level of Gate location.")
    st.markdown("20.Cleanliness: Satisfaction level of Cleanliness.")
    st.markdown("21.Check-in service: Satisfaction level of Check-in service.")
    st.markdown("22.Departure Delay in Minutes: Minutes delayed when departure.")
    st.markdown("23.Arrival Delay in Minutes: Minutes delayed when Arrival.")


    

    

elif (option =='EDA'):
    # Plot-1#
    
    
    st.subheader('Plot-1: Distribution plots of categorical flight specifications')
        
    selected_option_1= st.selectbox("Select an attribute for x",cols[cat_cols]) 
    title = "Satisfaction results by " + str(selected_option_1)
    fig = plt.figure(figsize = (8,5))
    sns.countplot(x = str(selected_option_1), data = Airways_df_train, hue ="satisfaction", palette ="husl" )
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")
    st.pyplot(fig)
    
    
    # plot-2: plots for user ratings
    st.subheader('Plot-2: Distribution plots of categorical user ratings between 0 to 5')
    selected_option_2= st.selectbox("Select an attribute for x",cols[cols_rankings])
    title = "Satisfaction results by " + str(selected_option_2)
    
    fig = plt.figure(figsize = (8,5))
    sns.countplot(x = str(selected_option_2), data = Airways_df_train, hue ="satisfaction",palette ="Pastel1" )
    # format graph
    plt.title("Satisfaction results by Ease of Online booking")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")
    st.pyplot(fig)
    
    
    # plot-3 combining plot 1 and 2
    st.subheader('Plot-3: Combing Plot-1 and Plot-2 for a certain user rating')
    st.text("Please zoom or click expand option it to see the details")
    st.write("Satisfaction results by " + str(selected_option_1) + " and " + str(selected_option_2))
    selected_option_3= st.selectbox("Select the user rating",[0,1,2,3,4,5])
    fig = plt.figure(figsize = (16,8))
    data_new = Airways_df_train.where(Airways_df_train[selected_option_2]==selected_option_3).dropna()
    
    sub_title = 'Satisfaction results by '+ str(selected_option_1) + " so that rating given for "  + str(selected_option_2) +' is 0,1,2,3,4,5'
    fig.suptitle(sub_title)
    sns.countplot(x = str(selected_option_1), data = data_new, hue ="satisfaction",palette ="PuBuGn")
    rat = "Rating "+ str(selected_option_3)
    plt.title("Rating 0")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")    
    st.pyplot(fig)
    
    # Plot-4: Scatter Plots
    st.subheader('Plot-4: Scatter Plots')
    cols_contineous = [2,5,20,21]
    st.text("Contineous Data Plots")
    sel_pot_4 = st.selectbox("X-Axis Variable:",cols[cols_contineous])
    sel_opt_5 = st.selectbox("Y-Axis variable:",cols[cols_contineous])
    fig = plt.figure(figsize = (8,5))
    sns.scatterplot(x = str(sel_pot_4), y = str(sel_opt_5), data = Airways_df_train, hue = "satisfaction")
    plt.legend()
    title = sel_pot_4+" Vs "+sel_opt_5+" with hue as Satisfaction"
    plt.title(title)
    st.pyplot(fig)
    st.subheader('Plot-5: Scatter plots when user rating is selected in between 0 to 5')
    wrt = "for "+selected_option_2+" select rating"
    sel_6 = st.selectbox(wrt,[0,1,2,3,4,5])
    new_data = Airways_df_train.where(Airways_df_train[selected_option_2]==sel_6).dropna()
    fig = plt.figure(figsize = (8,5))
    sns.scatterplot(x = str(sel_pot_4), y = str(sel_opt_5), data = new_data, hue = "satisfaction")
    plt.legend()
    title = sel_pot_4+" Vs "+sel_opt_5+" given rating "+ str(sel_6)+ " for "+selected_option_2
    plt.title(title)
    st.pyplot(fig)
    
       
elif (option =='Models and Outputs'):
    models = ["Logistic Regression", "SVM", "Decision Trees"]
    sel_mod = st.selectbox("Select model you want to explore",models)
    for i in cat_cols:
        le = LabelEncoder().fit(Airways_df_train[cols[i]])
        Airways_df_train[cols[i]] = le.transform(Airways_df_train[cols[i]])
        Airways_df_test[cols[i]] = le.transform(Airways_df_test[cols[i]])
        
    X_train = Airways_df_train[cols[0:-1]]
    y_train = Airways_df_train[cols[-1]]    
    
    X_test = Airways_df_test[cols[0:-1]]
    y_test = Airways_df_test[cols[-1]] 
    
    
    
    X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(
                            X_train , np.ravel(y_train),
                    test_size = 0.60, random_state = 101)
    
    
    X_train_mod_1, X_test_mod_1, y_train_mod_1, y_test_mod_1 = train_test_split(
                            X_test , np.ravel(y_test),
                    test_size = 0.80, random_state = 100)

    X_train_std = X_train_mod
    X_test_std = X_train_mod_1 
    # Feature Scaling
    cols_contineous = [2,5,20,21] 
    sc = StandardScaler()
    sc.fit(X_train_mod[cols[[2,5,20,21]]])
    X_train_std[cols[[2,5,20,21]]] = sc.transform(X_train_mod[cols[[2,5,20,21]]])
    X_test_std[cols[[2,5,20,21]]] = sc.transform(X_train_mod_1[cols[[2,5,20,21]]])
    
    y_train_std = y_train_mod
    y_test_std = y_train_mod_1
    labels = ['neutral or dissatisfied', 'satisfied']
    cols_contineous = [2,5,20,21]
    
    if sel_mod == "Logistic Regression":
        # logisticRegression
        st.subheader("Model after Scaled, removing Nulls, Encoding")
        st.dataframe(X_train_std)
        clf = LogisticRegression(random_state=0).fit(X_train_std, y_train_std)
        predictions_logistic = clf.predict(X_test_std)
        cm1 = confusion_matrix(y_test_std,predictions_logistic)
        plot_confusion_matrix(cm1,labels,"logistic Regression")
        txt = "Finetuned Hyper-Parameters are default values."
        st.write(txt)
        clf_report = classification_report(y_test_std,
                                   predictions_logistic,
                                   labels=[0,1],
                                   target_names=labels,
                                   output_dict=True)
        
        fig = plt.figure(figsize = (8,5))
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
        plt.legend()
        title = "Classification Report"
        plt.title(title)
        st.pyplot(fig)        
        
    if sel_mod == "SVM":
        st.subheader("Model after Scaled, removing Nulls, Encoding")
        st.dataframe(X_train_std)        
        svm = SVC(kernel= 'linear', random_state=1, C=0.1)
        svm.fit(X_train_std, y_train_std)
        predictions = svm.predict(X_test_std)
        cm = confusion_matrix(y_test_std,predictions)
        plot_confusion_matrix(cm,labels,"SVM")
        
        txt = "Finetuned Hyper-Parameters are: kernel= 'linear', random_state=1, C=0.1"
        st.write(txt)
        clf_report = classification_report(y_test_std,
                                   predictions,
                                   labels=[0,1],
                                   target_names=labels,
                                   output_dict=True)
        
        fig = plt.figure(figsize = (8,5))
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
        plt.legend()
        title = "Classification Report"
        plt.title(title)
        st.pyplot(fig)                
    if sel_mod == "Decision Trees":
        st.subheader("Model after Scaled, removing Nulls, Encoding")
        st.dataframe(X_train_std)
        DTclf = tree.DecisionTreeClassifier()
        DTclf = DTclf.fit(X_train_std, y_train_std)
        prediction_tree = DTclf.predict(X_test_std)
        cm2 = confusion_matrix(y_test_std,prediction_tree)
        plot_confusion_matrix(cm2,labels,"Decision Tress")
        labels = ['neutral or dissatisfied', 'satisfied']
        txt = "Finetuned Hyper-Parameters are: criterion = ”gini”, splitter  = ”best” and all other are default values."
        st.write(txt)
        clf_report = classification_report(y_test_std,
                                   prediction_tree,
                                   labels=[0,1],
                                   target_names=labels,
                                   output_dict=True)
        
        fig = plt.figure(figsize = (8,5))
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
        plt.legend()
        title = "Classification Report"
        plt.title(title)
        st.pyplot(fig)                
    
    


elif (option=='Explore'):
    
    st.subheader("Instructions to follow: ")
    st.write("1. Select your own data (it is not essential to select entire data as data. Data is finetuned for selected features).")
    st.write("2. Selection of data is divided into 3 categories:")
    st.write("------------a. Flight metrics -> Same for every customer.")
    st.write("------------b. Flight ratings -> Given by customers for services they experienced.")
    st.write("------------c. Flight travel metrics -> Contineous values")
    
    st.write("----------------------------------------------------------------------------------------------")
    cat_cols = [0,1,3,4]
    cols_rankings = [6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    cols_contineous = [2,5,20,21]
    
    new_cat_cols = st.multiselect(
    'Select your own customized input for flight characteristics. Flight characteristics are customer preferences.',
    cols[cat_cols])
    
    new_rank_cols = st.multiselect(
    'Select your own customized input for ranking metrics. Ranking metrics are in-flight customer service experiences.',
    cols[cols_rankings])
    
    new_cont_cols = st.multiselect(
    'What your own customized input for contineous metrics. These are pre-post departure metrics.',
    cols[cols_contineous])
    
    new_cols = new_cat_cols+new_rank_cols+new_cont_cols
    st.dataframe(Airways_df_train[new_cols].head())
    
    st.write("Please Provide Data for flight characteristics, ranking metrics, contineous metrics selected above")
    
    #cat_cols_0 = Airways_df_train[cols_rankings[0]].unique()
    
    
    new_cols1 = []
    for i in range(len(new_cols)):
        
        if new_cols[i] == cols[cat_cols[0]]:
            st.write("0: Male")
            st.write("1: Female")
            var1 = st.select_slider("Enter the gender",options=[0,1],value=0)
            new_cols1.append(var1)
            
        if new_cols[i] == cols[cat_cols[1]]:
            st.write("*0: Loyal Customer*")
            st.write("1: Not so Loyal Customer")
            var2 = st.select_slider("Enter Customer type",options=[0,1],value=0)
            new_cols1.append(var2)
            
        if new_cols[i] == cols[cat_cols[2]]:
            #var3 = st.slider("Enter the age",min_value=21, max_value=84,value=44)
            st.write("*0: Personal Travel*")
            st.write("1: Business Travel")
            var3 = st.select_slider("Enter the type of travel",options=[0,1],value=0)
            new_cols1.append(var3)
            
        
        if new_cols[i] == cols[cat_cols[3]]:
            st.write("0: Eco")
            st.write("1: Eco_Plus")
            st.write("2: Business")
            var4 = st.slider("Enter the class",min_value=0, max_value=2,value=0,step=1)
            new_cols1.append(var4)
            
        if new_cols[i] == cols[cols_rankings[0]]:
            var5 = st.slider("Enter the rating of inflight wifi services",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var5)
            
        if new_cols[i] == cols[cols_rankings[1]]:
            var6 = st.slider("Enter the rating of Departure/Arrival time convenient",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var6)
            
        if new_cols[i] == cols[cols_rankings[2]]:
            var7 = st.slider("Enter the rating of Ease of Online booking",min_value=0, max_value=5,value=0,step=1)        
            new_cols1.append(var7)
            
        if new_cols[i] == cols[cols_rankings[3]]:
            var8 = st.slider("Enter the rating of Gate location",min_value=0, max_value=5,value=0,step=1)  
            new_cols1.append(var8)
            
        if new_cols[i] == cols[cols_rankings[4]]:
            var9 = st.slider("Enter the rating of Food and drink",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var9)
            
        if new_cols[i] == cols[cols_rankings[5]]:
            var10 = st.slider("Enter the rating of Online boarding",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var10)
            
        if new_cols[i] == cols[cols_rankings[6]]:
            var11 = st.slider("Enter the rating of Seat comfort",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var11)
            
        if new_cols[i] == cols[cols_rankings[7]]:
            var12 = st.slider("Enter the rating of Inflight entertainment",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var12)
            
        if new_cols[i] == cols[cols_rankings[8]]:
            var13 = st.slider("Enter the rating of On-board service",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var13)
            
        if new_cols[i] == cols[cols_rankings[9]]:
            var14 = st.slider("Enter the rating of Leg room services",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var14)
        if new_cols[i] == cols[cols_rankings[10]]:
            var15 = st.slider("Enter the rating of Baggage handling",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var15)
            
        if new_cols[i] == cols[cols_rankings[11]]:
            var16 = st.slider("Enter the rating of Checkin service",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var16)
            
        if new_cols[i] == cols[cols_rankings[12]]:
            var17 = st.slider("Enter the rating of Inflight service",min_value=0, max_value=5,value=0,step=1)
            new_cols1.append(var17)
            
        if new_cols[i] == cols[cols_rankings[13]]:
            var18 = st.slider("Enter the rating of Cleanliness",min_value=0, max_value=5,value=0,step=1)  
            new_cols1.append(var18)
            
        if new_cols[i] == cols[cols_contineous[0]]:
            var19 = st.slider("Enter the age",min_value=21, max_value=84,value=44,step=2)
            new_cols1.append(var19)

        if new_cols[i] == cols[cols_contineous[1]]:
            var20 = st.number_input("Enter the Flight distance",min_value=31.0, max_value=4000.0,value=100.0,step=10.0)
            new_cols1.append(var20)
            
        if new_cols[i] == cols[cols_contineous[2]]:
            var21 = st.number_input("Enter the departure by delay",min_value=2.0, max_value=1592.0,value=100.0,step=10.0)
            new_cols1.append(var21)
        if new_cols[i] == cols[cols_contineous[3]]:
            var22 = st.number_input("Enter the Arrival Delay in Minutes",min_value=2.0, max_value=1584.0,value=100.0,step=10.0)
            new_cols1.append(var22)
            
    test_data = pd.DataFrame(new_cols1,index=new_cols)   
    test_data = test_data.transpose()
    
    if test_data.shape[1] <= 5:
        st.subheader("Data is Very Poor: Need More Data for better predictions")

    if  6 <= test_data.shape[1] <=10 :
        st.subheader("Data is Ok: More Data may work for better prediction")
        st.dataframe(test_data)
        for i in cat_cols:
            le = LabelEncoder().fit(Airways_df_train[cols[i]])
            Airways_df_train[cols[i]] = le.transform(Airways_df_train[cols[i]])
            
            
        new_airways_tr = Airways_df_train[new_cols]
        
        
        
        
        X_train = new_airways_tr[new_cols]
        y_train = Airways_df_train[cols[-1]]
        
        
        X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(
                                X_train , np.ravel(y_train),
                        test_size = 0.60, random_state = 101)
        
       
        
        # Feature Scaling
         
        sc = StandardScaler()
        sc.fit(X_train_mod)
        X_train_std = sc.transform(X_train_mod)
        X_test_std = sc.transform(test_data)
        y_train_std = y_train_mod
        
        labels = ['neutral or dissatisfied', 'satisfied']
    
        
        
        
        
        models = ["Logistic Regression"]
        sel_mod = st.selectbox("Select model you want to test on",models)   
        if sel_mod == "Logistic Regression":
            # logisticRegression
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=0).fit(X_train_std, y_train_std)
            predictions_logistic = clf.predict(X_test_std)
    
            if predictions_logistic == labels[1]:
                st.write("customer is satisfied")
            if predictions_logistic == labels[0]:
                st.write("customer is Neutral or Not-satisfied")    

        
    if test_data.shape[1] > 10 :
        st.subheader("Data is go to go")
        st.dataframe(test_data)
        for i in cat_cols:
            le = LabelEncoder().fit(Airways_df_train[cols[i]])
            Airways_df_train[cols[i]] = le.transform(Airways_df_train[cols[i]])
            
            
        new_airways_tr = Airways_df_train[new_cols]
        
        
        
        
        X_train = new_airways_tr[new_cols]
        y_train = Airways_df_train[cols[-1]]
        
        
        X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(
                                X_train , np.ravel(y_train),
                        test_size = 0.60, random_state = 101)
        
       
        
        # Feature Scaling
         
        sc = StandardScaler()
        sc.fit(X_train_mod)
        X_train_std = sc.transform(X_train_mod)
        X_test_std = sc.transform(test_data)
        y_train_std = y_train_mod
        
        labels = ['neutral or dissatisfied', 'satisfied']
        
        models = ["Logistic Regression"]
        sel_mod = st.selectbox("Select model you want to test on",models)   
        if sel_mod == "Logistic Regression":
            # logisticRegression
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=0).fit(X_train_std, y_train_std)
            predictions_logistic = clf.predict(X_test_std)
    
            if predictions_logistic == labels[1]:
                st.write("customer is satisfied")
            if predictions_logistic == labels[0]:
                st.write("customer is Neutral or Not-satisfied")    
