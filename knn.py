import streamlit as st 
import pandas as pd 
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

st.title("Top 50 song popularity")

st.sidebar.title('Tahapan Machine Learning')
menu_options = ['Data Collection','EDA', 'Modeling','Testing']
selected_menu = st.sidebar.selectbox('Pilih Menu:', menu_options,key=0)

if selected_menu == 'Data Collection':
    st.header('Accessing Data')
    st.text("data ini merupakan data 50 musik top yang di update setiap minggunya")
    data = pd.read_csv("Top-50-musicality-global.csv")
    df = pd.DataFrame(data)
    def addfitur():
        try:
            pop_scores = []

            for i in df["Popularity"]:
                if i >= 81 and i<=100:
                    pop_scores.append("Very popular")
                elif i >= 61 and i<=80:
                    pop_scores.append("Popular")
                elif i >= 40 and i<=60:
                    pop_scores.append("Middle popularity")
                else:
                    pop_scores.append("Unpopular")

            df["Popularity rank"] = pop_scores

            print(df[["Energy", "Liveness", "Popularity","Positiveness","Loudness","Popularity rank"]].tail(10))
            return True
        except:
            return False
            
    if addfitur():
        st.text("")
    st.dataframe(df)
    
    st.text("Jumlah baris dan kolom")
    st.write(df.shape)
    st.text("Informasi type data : ")
    st.write(df.dtypes)
    st.text("melihat fitur yang bersifat numerik")
    st.write(df.describe())
    st.text("menemukan missing value pada data")
    st.write(df.isnull().sum())

    #cek data
    st.subheader("Cleaning Data")
    st.text("melakukan pembersihan pada data")
    st.write(df.dropna(axis=0, inplace=True))

    st.text("check duplikasi data")
    st.write(df.duplicated().sum())

    st.text("menampilkan kembali data setelah dibersihkan")
    st.write(df.isnull().sum())

    st.subheader("menampilkan outlier")
    st.text("menggunakan boxplot untuk melihat outlier")
    columns_to_analyze = ['Energy', 'Liveness', 'Positiveness', 'Loudness']
    for column in columns_to_analyze:
        st.write(f"Boxplot untuk kolom '{column}':")
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df[[column]])
        st.pyplot(plt)
        plt.clf() 

    st.subheader("membersihkan outlier")
    def removeOutlier(col):
        sorted(col)
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        lowerRange = q1 - (1.5 * iqr)
        upperRange = q3 + (1.5 * iqr)
        return lowerRange, upperRange

    def main():
   
        df = pd.read_csv("Top-50-musicality-global.csv") 

    # Pembersihan outlier pada kolom Loudness
        lowScore, highScore = removeOutlier(df['Loudness'])
        df['Loudness'] = np.where(df['Loudness'] > highScore, highScore, df['Loudness'])
        df['Loudness'] = np.where(df['Loudness'] < lowScore, lowScore, df['Loudness'])

    # Pembersihan outlier pada kolom Energy
        lowScore, highScore = removeOutlier(df['Energy'])
        df['Energy'] = np.where(df['Energy'] > highScore, highScore, df['Energy'])
        df['Energy'] = np.where(df['Energy'] < lowScore, lowScore, df['Energy'])

        lowScore, highScore = removeOutlier(df['Liveness'])
        df['Liveness'] = np.where(df['Liveness'] > highScore, highScore, df['Liveness'])
        df['Liveness'] = np.where(df['Liveness'] < lowScore, lowScore, df['Liveness'])

    # Pembersihan outlier pada kolom Positiveness
        lowScore, highScore = removeOutlier(df['Positiveness'])
        df['Positiveness'] = np.where(df['Positiveness'] > highScore, highScore, df['Positiveness'])
        df['Positiveness'] = np.where(df['Positiveness'] < lowScore, lowScore, df['Positiveness'])

    # Menampilkan data setelah pembersihan outlier
        st.info("Data setelah pembersihan outlier")
        columns_to_analyze = ['Energy', 'Liveness', 'Positiveness', 'Loudness']
        for column in columns_to_analyze:
            st.write(f"kolom '{column}' setelah di bersihkan :")
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df[[column]])
            st.pyplot(plt)
            plt.clf()

    if __name__ == "__main__":
        main()
    

    st.subheader("melihat korelasi")
    st.text("pada diagram ini kita melihat korelasi tiap data")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_cols]
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='RdYlGn', ax=ax)
    st.pyplot(fig)
    st.text("Atribut Energy, Loudness, Liveness, dan Positiveness memiliki hubungan kuat")


    st.subheader("Preprocessing")
    st.text("disini kita akan menampilkan data kembali setelah di preprocessing")
    def del_fitur():
        try:
            del df['Danceability']
            del df['Acousticness']
            del df['duration']
            del df['Instrumentalness']
            del df['Key']
            del df['Mode']
            del df['Speechiness']
            del df['Tempo']
            del df['TSignature']
            del df['Artist Name']
            del df['Album Name']
            del df['Country']

            return True
        except:
            return False


    if del_fitur():
        st.info("Berhasil menghapus Fitur!")
        st.write(df.dtypes)
    else:
        st.warning("gagal menghapus!")

elif selected_menu == 'EDA':
    st.header('Exploratory Data Analysis')
    st.subheader("Relasi antar Data")
    data = pd.read_csv("Top-50-musicality-global.csv")
    df = pd.DataFrame(data)
    def del_fitur():
        try:
            del df['Danceability']
            del df['Acousticness']
            del df['duration']
            del df['Instrumentalness']
            del df['Key']
            del df['Mode']
            del df['Speechiness']
            del df['Tempo']
            del df['TSignature']
            del df['Artist Name']
            del df['Album Name']
            del df['Country']

            return True
        except:
            return False


    if del_fitur():
        st.text("")

    #visulisasi
    st.text("relasi data Loudness dan Energy")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x='Loudness', y='Energy', data=df, ax=ax)
    st.pyplot(fig)
    st.text("relasi data Liveness dan Positiveness")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x='Liveness', y='Positiveness', data=df, ax=ax)
    st.pyplot(fig)
    st.text("relasi data Energy dan Positiveness")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x='Energy', y='Positiveness', data=df, ax=ax)
    st.pyplot(fig)
    st.text("relasi data Loudness dan Positiveness")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x='Loudness', y='Positiveness', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribusi data")
    st.text("disini saya akan menampilkan distribusi 4 fitur tersebut")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(x='Energy', data=df, kde=True)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(x='Liveness', data=df, kde=True)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(x='Positiveness', data=df, kde=True)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(x='Loudness', data=df, kde=True)
    st.pyplot(fig)

elif selected_menu == 'Modeling':
    st.header('Modeling Data')
    data = pd.read_csv("Top-50-musicality-global.csv")
    df = pd.DataFrame(data)
    def del_fitur():
        try:
            del df['Danceability']
            del df['Acousticness']
            del df['duration']
            del df['Instrumentalness']
            del df['Key']
            del df['Mode']
            del df['Speechiness']
            del df['Tempo']
            del df['TSignature']
            del df['Artist Name']
            del df['Album Name']
            del df['Country']

            return True
        except:
            return False
        
    def addfitur():
        try:
            pop_scores = []

            for i in df["Popularity"]:
                if i >= 81 and i<=100:
                    pop_scores.append("Very popular")
                elif i >= 61 and i<=80:
                    pop_scores.append("Popular")
                elif i >= 40 and i<=60:
                    pop_scores.append("Middle popularity")
                else:
                    pop_scores.append("Unpopular")

            df["Popularity rank"] = pop_scores

            print(df[["Energy", "Liveness", "Popularity","Positiveness","Loudness","Popularity rank"]].tail(10))
            return True
        except:
            return False
        
    if addfitur() and del_fitur():
        st.text("menampilkan data & fitur kembali")
        st.write(df)
    
    def knn():
        # model
        X = df[["Energy", "Liveness","Positiveness","Loudness"]]
        y = df["Popularity rank"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17, stratify=y)

        n_range = np.arange(3, 20)

        train_accuracies = []
        test_accuracies = []

        for z in n_range:
            knn = KNeighborsClassifier(n_neighbors=z)
            knn.fit(X_train, y_train)
            train_accuracies.append(knn.score(X_train, y_train))
            test_accuracies.append(knn.score(X_test, y_test))

        best_n = n_range[np.argmax(test_accuracies)]
        best_knn = KNeighborsClassifier(n_neighbors=best_n)
        best_knn.fit(X_train, y_train)

        train_accuracy_percentage = best_knn.score(X_train, y_train) * 100
        test_accuracy_percentage = best_knn.score(X_test, y_test) * 100

        neigh = f"Akurasi model terbaik (n_neighbors = {best_n}):"
        train = f"Akurasi pada data latih: {train_accuracy_percentage:.2f}%"
        test = f"Akurasi pada data uji: {test_accuracy_percentage:.2f}%"

        return neigh,train,test
    
    def main():
        st.subheader("model KNN-Classifier")
        st.info("Berikut akurasi model KNN-classifier")
        neigh,train,test = knn()
        st.text(neigh)
        st.text(train)
        st.text(test)

    if __name__ == "__main__":
        main()

elif selected_menu == 'Testing':
    st.subheader("Testing")
    st.text("disini kami melakukan testing terhadap model yang kami buat")
    data = pd.read_csv("Top-50-musicality-global.csv")
    df = pd.DataFrame(data)
    def del_fitur():
        try:
            del df['Danceability']
            del df['Acousticness']
            del df['duration']
            del df['Instrumentalness']
            del df['Key']
            del df['Mode']
            del df['Speechiness']
            del df['Tempo']
            del df['TSignature']
            del df['Album Name']
            del df['Country']

            return True
        except:
            return False
        
    def addfitur():
        try:
            pop_scores = []

            for i in df["Popularity"]:
                if i > 80:
                    pop_scores.append("Very popular")
                elif i > 60:
                    pop_scores.append("Popular")
                elif i > 40:
                    pop_scores.append("Middle popularity")
                else:
                    pop_scores.append("Unpopular")

            df["Popularity rank"] = pop_scores

            print(df[["Energy", "Liveness", "Popularity","Positiveness","Loudness","Popularity rank"]].tail(10))
            return True
        except:
            return False
    
    def knn():
        # model
        X = df[["Energy", "Liveness","Positiveness","Loudness"]]
        y = df["Popularity rank"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17, stratify=y)

        n_range = np.arange(3, 20)

        train_accuracies = []
        test_accuracies = []

        for z in n_range:
            knn = KNeighborsClassifier(n_neighbors=z)
            knn.fit(X_train, y_train)
            train_accuracies.append(knn.score(X_train, y_train))
            test_accuracies.append(knn.score(X_test, y_test))

        best_n = n_range[np.argmax(test_accuracies)]
        best_knn = KNeighborsClassifier(n_neighbors=best_n)
        best_knn.fit(X_train, y_train)

        train_accuracy_percentage = best_knn.score(X_train, y_train) * 100
        test_accuracy_percentage = best_knn.score(X_test, y_test) * 100

        neigh = f"Akurasi model terbaik (n_neighbors = {best_n}):"
        train = f"Akurasi pada data latih: {train_accuracy_percentage:.2f}%"
        test = f"Akurasi pada data uji: {test_accuracy_percentage:.2f}%"

        return neigh,train,test
    

    def sample():
        X = df[["Energy", "Liveness", "Positiveness","Loudness"]]
        y = df["Popularity rank"]

        knn = KNeighborsClassifier(n_neighbors=5)  # contoh dengan 5 tetangga
        knn.fit(X, y)

    # Buat DataFrame sample untuk prediksi
        sample = pd.DataFrame({'Energy': [energy], 'Liveness': [liveness], 'Positiveness': [positiveness],'Loudness':[loudness]})

    # Lakukan prediksi dengan model yang sudah dilatih
        predicted_rank = knn.predict(sample)

        distances, indices = knn.kneighbors(sample, n_neighbors=5)

        print(f'Peringkat prediksi: {predicted_rank[0]}')

        print("5 Tetangga Terdekat:")
        neighbors_data = {'Rank': [], 'Distance': [], 'Track Name': [],'Artist Name':[], 'Energy': [], 'Liveness': [], 'Positiveness': [], 'Loudness': []}
        for i in range(len(distances[0])):
            neighbor_rank = y.iloc[indices[0][i]]
            neighbor_distance = distances[0][i]
            neighbor_song_data = df.iloc[indices[0][i]] 
            neighbors_data['Rank'].append(neighbor_rank)
            neighbors_data['Distance'].append(neighbor_distance)
            neighbors_data['Track Name'].append(neighbor_song_data['Track Name'])
            neighbors_data['Artist Name'].append(neighbor_song_data['Artist Name'])
            neighbors_data['Energy'].append(neighbor_song_data['Energy'])
            neighbors_data['Liveness'].append(neighbor_song_data['Liveness'])
            neighbors_data['Positiveness'].append(neighbor_song_data['Positiveness'])
            neighbors_data['Loudness'].append(neighbor_song_data['Loudness'])

        neighbors_df = pd.DataFrame(neighbors_data)
        st.text("musik yang serupa : ")
        st.write(neighbors_df)
        st.text("hasil klasifikasi nya menggunakan KNN : ")
        


        return predicted_rank[0]

    

    energy = st.number_input("Energy")
    liveness = st.number_input("Liveness")
    positiveness = st.number_input("Positiveness")
    loudness = st.number_input("Loudness")

    if st.button("Predict"):
        del_fitur()
        addfitur()
        knn()
        st.info(sample())
        

