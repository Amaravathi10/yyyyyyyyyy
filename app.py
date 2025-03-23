from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from googletrans import Translator
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
translator = Translator()

# Feature 1-3 Functions remain unchanged
# Feature 4 Functions (replaced with new code)
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    label_encoders = {}
    categorical_columns = ['category', 'Crop', 'District', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna('Unknown'))  # Handle NaN values
        label_encoders[col] = le

    return df, label_encoders

def prepare_data(df):
    X = df[['District', 'Season', 'Crop', 'category', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource']]
    y = df['ExpYield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X

def train_model(X_train, y_train):
    gb_regressor = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_regressor.fit(X_train, y_train)
    return gb_regressor

def translate_to_telugu(text):
    if text is None:
        return "గుర్తించబడలేదు"  # "Not identified" in Telugu
    try:
        translation = translator.translate(str(text), dest='te')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return str(text)

def get_top_crop_recommendations(model, district, season, df, label_encoders, scaler, top_n=3):
    try:
        district_encoded = label_encoders['District'].transform([district])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]
    except ValueError as e:
        print(f"Error: {e}. దయచేసి డేటాసెట్ నుండి చెల్లుబాటు అయ్యే జిల్లా మరియు సీజన్‌ను అందించండి.")
        return []

    district_season_crops = df[(df['District'] == district_encoded) & (df['Season'] == season_encoded)]['Crop'].unique()

    if len(district_season_crops) < top_n:
        season_crops = df[df['Season'] == season_encoded]['Crop'].unique()
        unique_crops = df[df['Crop'].isin(season_crops)][['Crop', 'category', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource']].drop_duplicates(subset=['Crop'])
    else:
        unique_crops = df[df['Crop'].isin(district_season_crops)][['Crop', 'category', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource']].drop_duplicates(subset=['Crop'])

    input_data = []
    for _, row in unique_crops.iterrows():
        input_row = [
            district_encoded, season_encoded, row['Crop'], row['category'],
            row['CNext'], row['CLast'], row['CTransp'], row['IrriType'], row['IrriSource']
        ]
        input_data.append(input_row)

    input_df = pd.DataFrame(input_data, columns=['District', 'Season', 'Crop', 'category', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource'])
    input_scaled = scaler.transform(input_df)

    predicted_yields = model.predict(input_scaled)
    input_df['ExpYield'] = predicted_yields
    top_crops = input_df.sort_values(by='ExpYield', ascending=False).drop_duplicates(subset=['Crop']).head(top_n)

    recommendations = []
    for _, row in top_crops.iterrows():
        recommendation = {
            'Crop': translate_to_telugu(label_encoders['Crop'].inverse_transform([int(row['Crop'])])[0]),
            'Category': translate_to_telugu(label_encoders['category'].inverse_transform([int(row['category'])])[0]),
            'CNext': translate_to_telugu(label_encoders['CNext'].inverse_transform([int(row['CNext'])])[0]),
            'CLast': translate_to_telugu(label_encoders['CLast'].inverse_transform([int(row['CLast'])])[0]),
            'CTransp': translate_to_telugu(label_encoders['CTransp'].inverse_transform([int(row['CTransp'])])[0]),
            'IrriType': translate_to_telugu(label_encoders['IrriType'].inverse_transform([int(row['IrriType'])])[0]),
            'IrriSource': translate_to_telugu(label_encoders['IrriSource'].inverse_transform([int(row['IrriSource'])])[0]),
            'ExpYield': f"{row['ExpYield']:.2f}"
        }
        recommendations.append(recommendation)

    return recommendations

# Load datasets
df_crop = pd.read_csv('Crop_recommendation.csv')
df_train = pd.read_csv('Train.csv')
df_new = pd.read_csv('new_dataset.csv')

# Feature 1 Model Setup (unchanged)
features_f1 = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X_f1 = df_crop[features_f1]
y_f1 = df_crop['label']
X_train_f1, _, y_train_f1, _ = train_test_split(X_f1, y_f1, test_size=0.2, random_state=42)
rf_model_f1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_f1.fit(X_train_f1, y_train_f1)
avg_temp = df_crop['temperature'].mean()
avg_humidity = df_crop['humidity'].mean()
avg_ph = df_crop['ph'].mean()
avg_rainfall = df_crop['rainfall'].mean()

# Feature 2 Model Setup (unchanged)
le_f2 = LabelEncoder()
df_crop['label_encoded'] = le_f2.fit_transform(df_crop['label'].str.lower())
X_f2 = df_crop[['label_encoded']]
y_f2 = df_crop[['N', 'P', 'K']]
X_train_f2, _, y_train_f2, _ = train_test_split(X_f2, y_f2, test_size=0.2, random_state=42)
rf_model_f2 = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
rf_model_f2.fit(X_train_f2, y_train_f2)

# Feature 3 Model Setup (unchanged)
df_crop_f3 = df_crop.rename(columns={'label': 'Crop'})
data_f3 = df_train.copy()
for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
    crop_values = {crop: df_crop_f3[df_crop_f3['Crop'] == crop][col].mean() for crop in df_crop_f3['Crop'].unique()}
    data_f3[col] = data_f3['Crop'].map(crop_values).fillna(df_crop_f3[col].mean())
data_f3 = data_f3.rename(columns={'rainfall': 'rainfall_required'})
label_encoders_f3 = {}
for col in ['Crop', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season', 'District']:
    le = LabelEncoder()
    data_f3[col] = le.fit_transform(data_f3[col].astype(str))
    label_encoders_f3[col] = le
features_f3 = ['District', 'Season', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall_required']
X_f3 = data_f3[features_f3]
y_f3 = data_f3['Crop']
X_train_f3, _, y_train_f3, _ = train_test_split(X_f3, y_f3, test_size=0.2, random_state=42)
model_f3 = GradientBoostingRegressor()
model_f3.fit(X_train_f3, y_train_f3)

# Feature 4 Model Setup (replaced)
df_f4, label_encoders_f4 = load_and_preprocess_data('Train.csv')
X_train_f4, _, y_train_f4, _, scaler_f4, _ = prepare_data(df_f4)
model_f4 = train_model(X_train_f4, y_train_f4)


# CUSTOM_TRANSLATIONS dictionary (unchanged)
CUSTOM_TRANSLATIONS = {
    'apple': 'ఆపిల్',
    'banana': 'అరటి',
    'blackgram': 'మినుములు',
    'chickpea': 'శనగలు',
    'chillies': 'మిరపకాయలు',
    'coconut': 'కొబ్బరి',
    'coffee': 'కాఫీ',
    'coriander': 'కొత్తిమీర',
    'cotton': 'పత్తి',
    'grapes': 'ద్రాక్ష',
    'groundnut': 'వేరుశనగ',
    'jute': 'జనపనార',
    'kidneybeans': 'రాజ్మా',
    'lentil': 'పప్పు',
    'maize': 'మొక్కజొన్న',
    'mango': 'మామిడి',
    'marigold': 'బంతి పుష్పం',
    'mothbeans': 'మోత్ బీన్స్',
    'mungbean': 'పెసర్లు',
    'muskmelon': 'ఖర్బుజ',
    'orange': 'నారింజ',
    'paddy': 'వరి',
    'papaya': 'బొప్పాయి',
    'pomegranate': 'దానిమ్మ',
    'redgram': 'కందులు',
    'soybean': 'సోయాబీన్',
    'sunflower': 'పొద్దతిరుగుడు',
    'tomato': 'టమాటో',
    'turmeric': 'పసుపు',
    'watermelon': 'పుచ్చకాయ'
}

TELUGU_TO_ENGLISH = {v: k for k, v in CUSTOM_TRANSLATIONS.items()}


DISTRICT_TRANSLATIONS = {
    'Adilabad': 'ఆదిలాబాద్',
    'Karimnagar': 'కరీంనగర్',
    'Warangal': 'వరంగల్',
    'Khammam': 'ఖమ్మం',
    'Nalgonda': 'నల్గొండ',
    'Mahabubnagar': 'మహబూబ్‌నగర్',
    'Rangareddy': 'రంగారెడ్డి',
    'Hyderabad': 'హైదరాబాద్',
    'Medak': 'మెదక్',
    'Nizamabad': 'నిజామాబాద్',
    'Jagtial': 'జగిత్యాల',
    'Peddapalli': 'పెద్దపల్లి',
    'Siddipet': 'సిద్దిపేట',
    'Kamareddy': 'కామారెడ్డి',
    'Sangareddy': 'సంగారెడ్డి',
    'Bhadradri Kothagudem': 'భద్రాద్రి కొత్తగూడెం',
    'Jayashankar Bhupalpally': 'జయశంకర్ భూపాలపల్లి',
    'Jogulamba Gadwal': 'జోగులాంబ గద్వాల్',
    'Wanaparthy': 'వనపర్తి',
    'Nagarkurnool': 'నాగర్‌కర్నూల్',
    'Vikarabad': 'వికారాబాద్',
    'Rajanna Sircilla': 'రాజన్న సిరిసిల్లా',
    'Mahabubabad': 'మహబూబాబాద్',
    'Suryapet': 'సూర్యాపేట',
    'Yadadri Bhuvanagiri': 'యాదాద్రి భువనగిరి',
    'Narayanpet': 'నారాయణపేట',
    'Nirmal': 'నిర్మల్'
}
TELUGU_TO_ENGLISH_DISTRICTS = {v: k for k, v in DISTRICT_TRANSLATIONS.items()}


unique_districts_english = df_train['District'].unique()
districts_telugu = [DISTRICT_TRANSLATIONS.get(district, district) for district in unique_districts_english]
districts_telugu.sort()

def t(text):
    try:
        text_lower = text.lower()
        if text_lower in CUSTOM_TRANSLATIONS:
            return CUSTOM_TRANSLATIONS[text_lower]
        return translator.translate(text, dest='te').text
    except:
        return text

# Feature 1 Functions (unchanged)
def predict_crop(N, P, K):
    input_data = np.array([[N, P, K, avg_temp, avg_humidity, avg_ph, avg_rainfall]])
    return t(rf_model_f1.predict(input_data)[0])

def find_similar_crops(N, P, K, main_recommendation):
    crop_avg = df_crop.groupby('label')[['N', 'P', 'K']].mean().reset_index()
    nbrs = NearestNeighbors(n_neighbors=6, metric='euclidean').fit(crop_avg[['N', 'P', 'K']])
    distances, indices = nbrs.kneighbors([[N, P, K]])
    similar_crops = []
    for idx, distance in zip(indices[0], distances[0]):
        crop_data = crop_avg.iloc[idx]
        if crop_data['label'] != main_recommendation:
            adjustment = {
                'crop': t(crop_data['label']),
                'N_diff': crop_data['N'] - N,
                'P_diff': crop_data['P'] - P,
                'K_diff': crop_data['K'] - K,
            }
            similar_crops.append(adjustment)
    return similar_crops[:5]

def print_adjustment(current_value, target_value, nutrient):
    diff = target_value - current_value
    if diff > 0:
        return t(f"పెంచుకోవాలి {abs(diff):.2f} టార్గెట్ {nutrient} కి చేరుకోవాలి {target_value:.2f}")
    elif diff < 0:
        return t(f"తగ్గించుకోవాలి {abs(diff):.2f} టార్గెట్ {nutrient} కి చేరుకోవాలి {target_value:.2f}")
    else:
        return t(f"మార్పు అవసరం లేదు (ప్రస్తుత మరియు టార్గెట్ {nutrient}: {current_value:.2f})")

# Feature 2 Functions (unchanged)
def predict_npk_values(crop_name):
    # Convert Telugu crop name back to English if it's in Telugu
    crop_name_lower = crop_name.lower()
    if crop_name_lower in TELUGU_TO_ENGLISH:
        crop_name_english = TELUGU_TO_ENGLISH[crop_name_lower]
    else:
        crop_name_english = crop_name_lower  # Assume it's already in English if not found
    crop_encoded = le_f2.transform([crop_name_english])[0]
    npk_pred = rf_model_f2.predict([[crop_encoded]])[0]
    crop_data = df_crop[df_crop['label'].str.lower() == crop_name_english]
    return {
        'N': {'predicted': npk_pred[0], 'min': crop_data['N'].min(), 'max': crop_data['N'].max(), 'mean': crop_data['N'].mean()},
        'P': {'predicted': npk_pred[1], 'min': crop_data['P'].min(), 'max': crop_data['P'].max(), 'mean': crop_data['P'].mean()},
        'K': {'predicted': npk_pred[2], 'min': crop_data['K'].min(), 'max': crop_data['K'].max(), 'mean': crop_data['K'].mean()}
    }

# Feature 3 Functions (unchanged)
def get_crop_details(crop_name):
    crop_data = df_crop_f3[df_crop_f3['Crop'] == crop_name]
    if crop_data.empty:
        print(f"No data found for crop: {crop_name}")
        return {'N': 0, 'P': 0, 'K': 0, 'rainfall_required': 0}
    return {
        'N': crop_data['N'].mean(),
        'P': crop_data['P'].mean(),
        'K': crop_data['K'].mean(),
        'rainfall_required': crop_data['rainfall_required'].mean()
    }

def predict_crops_with_details(district, total_area):
    try:
        print(f"డిస్ట్రిక్ట్ కోసం ప్రెడిక్ట్ చేస్తున్నాం: {district}, ప్రాంతం: {total_area}")
        
        if district not in label_encoders_f3['District'].classes_:
            print(f"డిస్ట్రిక్ట్ {district} లేబుల్_ఎన్కోడర్స్_f3 లో కనపడలేదు. మోడ్ విలువను ఉపయోగిస్తున్నాం.")
            district_encoded = data_f3['District'].mode()[0]
        else:
            district_encoded = label_encoders_f3['District'].transform([district])[0]

        input_data = {f: data_f3[f].mean() for f in features_f3}
        input_data['District'] = district_encoded
        input_df = pd.DataFrame([input_data])

        crop_pred = model_f3.predict(input_df)[0]
        primary_crop = label_encoders_f3['Crop'].inverse_transform([int(round(crop_pred))])[0]
        print(f"ప్రాథమిక పంట ప్రెడిక్ట్ చేయబడింది: {primary_crop}")

        district_crops = df_new[df_new['District'] == district] if not df_new[df_new['District'] == district].empty else df_new
        if district_crops.empty:
            print("డిస్ట్రిక్ట్-నిర్దిష్ట పంటలు కనపడలేదు. new_dataset నుండి అన్ని పంటలను ఉపయోగిస్తున్నాం.")
        
        required_columns = ['Crop', 'Market Demand', 'Yield (kg/ha)', 'Rainfall (mm)']
        for col in required_columns:
            if col not in district_crops.columns:
                print(f"హెచ్చరిక: కాలమ్ {col} new_dataset.csv లో మిస్సింగ్. డిఫాల్ట్ విలువలను ఉపయోగిస్తున్నాం.")
                if col == 'Market Demand':
                    district_crops[col] = 'మధ్యస్థ'
                elif col in ['Yield (kg/ha)', 'Rainfall (mm)']:
                    district_crops[col] = 0

        demand_rank = {'High': 1, 'Medium': 2, 'Low': 3}
        district_crops['Demand Rank'] = district_crops['Market Demand'].map(demand_rank).fillna(2)
        district_crops = district_crops.sort_values(by=['Demand Rank', 'Yield (kg/ha)'], ascending=[True, False])

        recommendations = [{
            'Name': t(primary_crop),
            'Area': total_area * 0.4,
            'Rainfall Needed': get_crop_details(primary_crop)['rainfall_required'] if get_crop_details(primary_crop)['rainfall_required'] else 0,
            'Nutrient Requirements': get_crop_details(primary_crop),
            'Market Demand': 'అధిక',
            'Source': 'ప్రాథమిక ప్రెడిక్షన్'
        }]
        
        remaining_area = total_area * 0.6
        num_additional_crops = min(len(district_crops), 3)
        if num_additional_crops > 0:
            area_per_additional_crop = remaining_area / num_additional_crops
            for _, crop in district_crops.head(num_additional_crops).iterrows():
                recommendations.append({
                    'Name': t(crop['Crop']),
                    'Area': area_per_additional_crop,
                    'Rainfall Needed': crop['Rainfall (mm)'] if pd.notna(crop['Rainfall (mm)']) else 0,
                    'Yield': crop['Yield (kg/ha)'] if pd.notna(crop['Yield (kg/ha)']) else 0,
                    'Market Demand': crop['Market Demand'] if pd.notna(crop['Market Demand']) else 'మధ్యస్థ',
                    'Source': 'మార్కెట్ విశ్లేషణ'
                })
        else:
            print("new_dataset.csv లో అదనపు పంటలు కనపడలేదు.")

        print(f"చివరి సిఫార్సులు: {recommendations}")
        return recommendations if recommendations else []
    
    except Exception as e:
        print(f"predict_crops_with_details లో ఎర్రర్: {e}")
        return []

# Routes (Feature 4 route updated)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feature1', methods=['GET', 'POST'])
def feature1():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        recommended_crop = predict_crop(N, P, K)
        similar_crops = find_similar_crops(N, P, K, recommended_crop)
        result = {'crop': recommended_crop, 'similar': [{'crop': c['crop'], 'N': print_adjustment(N, N + c['N_diff'], 'N'), 'P': print_adjustment(P, P + c['P_diff'], 'P'), 'K': print_adjustment(K, K + c['K_diff'], 'K')} for c in similar_crops]}
        return render_template('feature1.html', result=result)
    return render_template('feature1.html')

@app.route('/feature2', methods=['GET', 'POST'])
def feature2():
    crops = sorted([t(crop) for crop in df_crop['label'].str.lower().unique()])
    if request.method == 'POST':
        crop_name = request.form['crop']
        npk_stats = predict_npk_values(crop_name)
        return render_template('feature2.html', result=npk_stats, crops=crops)
    return render_template('feature2.html', crops=crops)

@app.route('/feature3', methods=['GET', 'POST'])
def feature3():
    districts = sorted([DISTRICT_TRANSLATIONS[district] for district in df_new['District'].unique()])
    if request.method == 'POST':
        try:
            district_telugu = request.form['district'].strip()
            district = TELUGU_TO_ENGLISH_DISTRICTS.get(district_telugu, district_telugu)  # Convert Telugu to English, fallback to input if not found
            area = float(request.form['area'])
            if area <= 0:
                raise ValueError("మొత్తం ప్రాంతం 0 కంటే ఎక్కువగా ఉండాలి.")
            
            recommendations = predict_crops_with_details(district, area)
            print(f"సిఫార్సులు: {recommendations}")
            
            if not recommendations:
                return render_template('feature3.html', error="సిఫార్సులను రూపొందించలేకపోయాం. దయచేసి డిస్ట్రిక్ట్ పేరు లేదా డేటాసెట్‌ను తనిఖీ చేయండి.", districts=districts)
            
            return render_template('feature3.html', result=recommendations, districts=districts)
        except ValueError as ve:
            print(f"విలువ ఎర్రర్ feature3 లో: {ve}")
            return render_template('feature3.html', error=f"చెల్లని ఇన్‌పుట్: {str(ve)}", districts=districts)
        except Exception as e:
            print(f"feature3 లో ఎర్రర్: {e}")
            return render_template('feature3.html', error=f"రిక్వెస్ట్ ప్రాసెసింగ్‌లో ఎర్రర్: {str(e)}", districts=districts)
    return render_template('feature3.html', districts=districts)

@app.route('/feature4', methods=['GET', 'POST'])
def feature4():
    if request.method == 'POST':
        district = request.form['district']  # This will now be in Telugu
        season = request.form['season']      # This will be "రబీ" or "ఖరీఫ్"
        # Convert Telugu district back to English for processing
        district_english = TELUGU_TO_ENGLISH_DISTRICTS.get(district, district)
        # Convert Telugu season back to English for processing
        season_english = "Rabi" if season == "రబీ" else "Kharif"
        recommendations = get_top_crop_recommendations(model_f4, district_english, season_english, df_f4, label_encoders_f4, scaler_f4)
        return render_template('feature4.html', result=recommendations, districts=districts_telugu)
    return render_template('feature4.html', districts=districts_telugu)

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    if not message:
        return jsonify({'reply': 'దయచేసి సందేశం పంపండి.'})
    reply = process_chat_message(message)
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)