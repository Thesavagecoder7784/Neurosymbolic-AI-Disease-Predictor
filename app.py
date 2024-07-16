from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import networkx as nx
import json

app = Flask(__name__)

# Load datasets
symptom_severity_df = pd.read_csv('Dataset/Symptom-severity.csv')
disease_symptom_df = pd.read_csv('Dataset/dataset.csv')
disease_description_df = pd.read_csv('Dataset/symptom_Description.csv')
disease_precaution_df = pd.read_csv('Dataset/symptom_precaution.csv')

# Initialize data structures
symptom_severity_dict = dict(zip(symptom_severity_df['Symptom'], symptom_severity_df['weight']))
symptom_to_index = {symptom: index for index, symptom in enumerate(symptom_severity_dict.keys())}
disease_symptom_dup = disease_symptom_df.drop_duplicates()
disease_symptom_dict = {}
symptom_disease_dict = {}

# Function to capitalize words and remove underscores
capitalize_and_remove_underscore = lambda x: x.title().replace('_', '')
disease_precaution_df = disease_precaution_df.fillna('')

symptom_dict = {
    'abdominal_pain': 'Pain or discomfort felt in the area between the chest and pelvis',
    'yellowing_of_eyes': 'Discoloration of the white part of the eyes due to the buildup of bilirubin',
    'weight_gain': 'Increase in body weight beyond normal fluctuations',
    'dehydration': 'Insufficient amount of water in the body',
    'painful_walking': 'Discomfort or pain experienced while walking',
    'blackheads': 'Small, dark-colored bumps that appear on the skin',
    'nausea': 'Feeling of unease or discomfort in the stomach',
    'swelling_of_stomach': 'Abnormal enlargement or bloating of the abdomen',
    'throat_irritation': 'Sensation of discomfort, itchiness, or scratchiness in the throat',
    'stiff_neck': 'Stiffness, soreness, or difficulty in moving the neck',
    'acidity': 'Excessive acid production in the stomach',
    'lack_of_concentration': 'Difficulty focusing, paying attention, or staying mentally engaged',
    'spotting_ urination': 'Occurrence of small amounts of blood in the urine',
    'belly_pain': 'Pain or discomfort in the abdominal area',
    'rusty_sputum': 'Presence of reddish-brown or rusty-colored sputum',
    'swollen_legs': 'Abnormal swelling of the legs',
    'chills': 'Sudden feeling of coldness accompanied by shivering',
    'pus_filled_pimples': 'Pimples filled with pus on the skin',
    'palpitations': 'Awareness of the heartbeat, often characterized by irregular or rapid pulsations',
    'unsteadiness': 'Lack of stability or balance',
    'scurring': 'Formation of scales or flakes on the skin',
    'fluid_overload': 'Excessive accumulation of fluid in the body',
    'malaise': 'General feeling of discomfort, illness, or unease',
    'runny_nose': 'Excessive production of mucus from the nose',
    'knee_pain': 'Pain or discomfort in the knee joint',
    'swelled_lymph_nodes': 'Enlargement of lymph nodes, often indicating an infection',
    'dizziness': 'Feeling of lightheadedness, unsteadiness, or vertigo',
    'continuous_feel_of_urine': 'Persistent sensation of needing to urinate',
    'stomach_pain': 'Pain or discomfort in the stomach',
    'dark_urine': 'Abnormally dark-colored urine',
    'weakness_of_one_body_side': 'Weakness or paralysis affecting one side of the body',
    'yellowish_skin': 'Abnormal yellow discoloration of the skin',
    'muscle_wasting': 'Loss or decrease in muscle mass',
    'mucoid_sputum': 'Thick, sticky mucus produced by the respiratory tract',
    'pain_behind_the_eyes': 'Pain felt behind or around the eyes',
    'blurred_and_distorted_vision': 'Loss of sharpness and clarity of vision',
    'irritation_in_anus': 'Discomfort, itching, or inflammation in the anus',
    'distention_of_abdomen': 'Abdominal swelling or bloating',
    'depression': 'Persistent feeling of sadness, hopelessness, or lack of interest',
    'watering_from_eyes': 'Excessive tear production from the eyes',
    'sunken_eyes': 'Abnormal inward appearance or hollowing of the eyes',
    'itching': 'Unpleasant sensation that triggers the desire to scratch',
    'drying_and_tingling_lips': 'Dryness and tingling sensation on the lips',
    'fatigue': 'Extreme tiredness, lack of energy, or exhaustion',
    'slurred_speech': 'Difficulty in pronouncing words clearly or coherently',
    'foul_smell_of urine': 'Unpleasant or strong odor of urine',
    'neck_pain': 'Pain or discomfort in the neck',
    'skin_rash': 'Abnormal changes in the skin, often characterized by redness, itchiness, or irritation',
    'hip_joint_pain': 'Pain or discomfort in the hip joint',
    'excessive_hunger': 'Abnormally increased or excessive appetite',
    'sinus_pressure': 'Feeling of pressure or pain in the sinuses',
    'swelling_joints': 'Abnormal swelling or enlargement of the joints',
    'history_of_alcohol_consumption': 'Previous or current habit of consuming alcohol',
    'abnormal_menstruation': 'Irregular, unusual, or abnormal menstrual periods',
    'bladder_discomfort': 'Discomfort or pain in the bladder',
    'high_fever': 'Elevated body temperature above the normal range',
    'diarrhoea': 'Frequent, loose, or watery bowel movements',
    'red_spots_over_body': 'Red spots or patches appearing on the skin',
    'nodal_skin_eruptions': 'Skin eruptions or lesions characterized by nodes or lumps',
    'skin_peeling': 'Shedding or loss of the outer layer of the skin',
    'movement_stiffness': 'Difficulty in initiating or controlling movements',
    'chest_pain': 'Pain or discomfort in the chest',
    'continuous_sneezing': 'Repetitive and frequent sneezing',
    'passage_of_gases': 'Release of gas through the anus',
    'shivering': 'Involuntary shaking or trembling of the body',
    'vomiting': 'Expelling the contents of the stomach through the mouth',
    'yellow_crust_ooze': 'Yellowish crust or discharge from a skin wound or lesion',
    'family_history': 'History of specific diseases or conditions within the family',
    'yellow_urine': 'Abnormally yellow-colored urine',
    'coma': 'Unresponsive state of deep unconsciousness',
    'muscle_pain': 'Pain or discomfort in the muscles',
    'constipation': 'Difficulty or infrequent bowel movements',
    'indigestion': 'Discomfort or pain in the upper abdomen, often associated with eating',
    'increased_appetite': 'Excessive or abnormally increased appetite',
    'puffy_face_and_eyes': 'Swelling or puffiness of the face and eyes',
    'polyuria': 'Excessive production of urine',
    'burning_micturition': 'Burning or painful sensation during urination',
    'bloody_stool': 'Presence of blood in the stool',
    'blister': 'Small fluid-filled raised area on the skin',
    'receiving_blood_transfusion': 'Process of receiving blood or blood products from a donor',
    'toxic_look_(typhos)': 'Appearance of toxicity or illness',
    'bruising': 'Appearance of bruise or discoloration of the skin',
    'cramps': 'Painful muscle contractions',
    'red_sore_around_nose': 'Red, irritated, or sore skin around the nose',
    'muscle_weakness': 'Decreased strength or power in the muscles',
    'weight_loss': 'Reduction in body weight, often unintentional',
    'dischromic_patches': 'Abnormal patches of skin color',
    'irritability': 'Tendency to be easily annoyed or agitated',
    'prominent_veins_on_calf': 'Clearly visible veins on the calf muscles',
    'pain_in_anal_region': 'Pain or discomfort in the anal area',
    'inflammatory_nails': 'Inflammation or swelling around the nails',
    'mild_fever': 'Slight elevation in body temperature',
    'internal_itching': 'Itching sensation inside the body or on internal organs',
    'sweating': 'Production of sweat, often due to heat, physical activity, or anxiety',
    'swollen_extremeties': 'Enlargement or swelling of the hands, feet, or limbs',
    'acute_liver_failure': 'Sudden loss of liver function',
    'joint_pain': 'Pain or discomfort in the joints',
    'spinning_movements': 'Sensation of the surrounding environment spinning or moving',
    'silver_like_dusting': 'Silver-like appearance or dusting on the skin',
    'extra_marital_contacts': 'Engaging in sexual activity outside of a committed relationship or marriage',
    'weakness_in_limbs': 'Weakness or lack of strength in the limbs',
    'lethargy': 'State of tiredness, lack of energy, or sluggishness',
    'blood_in_sputum': 'Presence of blood in the coughed up sputum',
    'back_pain': 'Pain or discomfort in the back',
    'headache': 'Pain or discomfort in the head or upper neck',
    'enlarged_thyroid': 'Abnormal enlargement of the thyroid gland in the neck',
    'restlessness': 'Inability to rest or relax, often associated with anxiety or agitation',
    'visual_disturbances': 'Abnormalities or changes in vision',
    'breathlessness': 'Difficulty in breathing or shortness of breath',
    'loss_of_smell': 'Decreased or complete loss of the sense of smell',
    'redness_of_eyes': 'Redness or bloodshot appearance of the eyes',
    'cold_hands_and_feets': 'Unusually cold hands and feet',
    'anxiety': 'Feeling of worry, fear, or unease',
    'altered_sensorium': 'Changes in perception, awareness, or consciousness',
    'loss_of_appetite': 'Lack of desire or reduced interest in eating',
    'patches_in_throat': 'Discolored or abnormal patches in the throat',
    'small_dents_in_nails': 'Tiny depressions or dents on the nails',
    'fast_heart_rate': 'Elevated or rapid heart rate',
    'stomach_bleeding': 'Bleeding from the stomach or gastrointestinal tract',
    'brittle_nails': 'Nails that are weak, fragile, or prone to breakage',
    'loss_of_balance': 'Lack of balance or equilibrium',
    'swollen_blood_vessels': 'Enlarged or distended blood vessels',
    'obesity': 'Excessive accumulation of body fat',
    'phlegm': 'Thick, sticky mucus produced by the respiratory tract',
    'ulcers_on_tongue': 'Painful sores or lesions on the tongue',
    'congestion': 'Excessive buildup of fluid or mucus in the respiratory tract',
    'mood_swings': 'Rapid or extreme changes in mood or emotional state',
    'cough': 'Expelling air from the lungs with a sudden, sharp sound',
    'receiving_unsterile_injections': 'Administration of injections without proper sterilization',
    'irregular_sugar_level': 'Abnormal or fluctuating blood sugar levels',
    'pain_during_bowel_movements': 'Pain or discomfort experienced during bowel movements'
}

knowledge_graph = nx.DiGraph()

# Normalize symptoms and create dictionaries
for _, row in disease_symptom_dup.iterrows():
    disease = row['Disease'].strip()
    symptoms = [symptom.strip().lower() for symptom in row.drop('Disease').dropna().tolist()]
    disease_symptom_dict.setdefault(disease, []).extend(symptoms)
    for symptom in symptoms:
        symptom_disease_dict.setdefault(symptom, []).extend(disease)
        knowledge_graph.add_edges_from([(symptom, disease)])

# Define custom layer for symbolic reasoning
class SymbolicReasoningLayer(layers.Layer):
    def __init__(self, knowledge_graph, num_diseases, **kwargs):
        super(SymbolicReasoningLayer, self).__init__(**kwargs)
        self.knowledge_graph = knowledge_graph
        self.num_diseases = num_diseases
        self.symbolic_reasoning_output = layers.Dense(self.num_diseases, activation='softmax')

    @tf.function
    def call(self, inputs):
        inferred_diseases = []
        for symptoms in inputs:
            inferred_disease = set()
            for symptom in symptoms:
                if symptom in self.knowledge_graph:
                    related_diseases = nx.descendants(self.knowledge_graph, symptom)
                    inferred_disease.update(related_diseases)
            inferred_diseases.append(list(inferred_disease))
        return self.symbolic_reasoning_output(tf.convert_to_tensor(inferred_diseases))

# Encode symptoms into a one-hot encoded vector
def encode_symptoms(symptoms):
    encoded_symptoms = np.zeros(len(symptom_severity_dict))
    for symptom in symptoms:
        if symptom in symptom_to_index:
            encoded_symptoms[symptom_to_index[symptom]] = 1
    return encoded_symptoms

# Build the hybrid model
symptom_input = layers.Input(shape=(len(symptom_severity_dict),), name='symptom_input')
neural_network_output = layers.Dense(128, activation='relu')(symptom_input)
neural_network_output = layers.Dropout(0.5)(neural_network_output)
neural_network_output = layers.Dense(64, activation='relu')(neural_network_output)
neural_network_output = layers.Dropout(0.5)(neural_network_output)
neural_network_output = layers.Dense(len(disease_symptom_dict), activation='softmax', name='neural_network_output')(neural_network_output)
symbolic_reasoning_output = SymbolicReasoningLayer(knowledge_graph, len(disease_symptom_dict))(symptom_input)
combined_output = layers.Multiply(name='combined_output')([neural_network_output, symbolic_reasoning_output])

# Create and compile the model
model = Model(inputs=[symptom_input], outputs=[combined_output])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare training data
X_train = []
y_train = []
for disease, symptoms in disease_symptom_dict.items():
    X_train.append(encode_symptoms(symptoms))
    y_train.append(np.array([1 if d == disease else 0 for d in disease_symptom_dict.keys()]))
X_train = np.array(X_train)
y_train = np.array(y_train)

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=32)

# Function to get feature importance based on model weights
def get_feature_importance(model, user_symptoms):
    feature_names = list(symptom_severity_dict.keys())
    user_symptom_indices = np.array([symptom_to_index[symptom] for symptom in user_symptoms if symptom in symptom_to_index])
    feature_importance = np.abs(model.get_weights()[0]).sum(axis=1)
    user_symptom_importance = feature_importance[user_symptom_indices]
    sorted_indices = np.argsort(user_symptom_importance)[::-1]
    sorted_features = [feature_names[i] for i in user_symptom_indices[sorted_indices]]
    sorted_importance = user_symptom_importance[sorted_indices]
    return sorted_features, sorted_importance


# Function to predict disease based on symptoms using the trained model
def predict_disease(symptoms):
    encoded_symptoms = encode_symptoms(symptoms)
    predicted = model.predict(np.array([encoded_symptoms]))
    disease_index = np.argmax(predicted)
    predicted_disease = list(disease_symptom_dict.keys())[disease_index]

    # Add explanation within the prediction process
    print("The model predicted the disease based on the following symptoms:")
    for symptom in symptoms:
        print("- " + symptom)
    print("The predicted disease is: " + predicted_disease)

    return predicted_disease


def explain_decision(symptoms, user_symptoms):
    encoded_symptoms = encode_symptoms(symptoms)
    predicted = model.predict(np.array([encoded_symptoms]))
    disease_index = np.argmax(predicted)
    predicted_disease = list(disease_symptom_dict.keys())[disease_index]
    top_diseases_indices = np.argsort(predicted)[0][-3:][::-1]
    top_diseases = [list(disease_symptom_dict.keys())[index] for index in top_diseases_indices]
    
    # Get disease description and precautions
    disease_description = get_disease_description(predicted_disease)
    disease_precautions = get_disease_precautions(predicted_disease)
    
    # Check for other diseases with the same symptoms
    potential_diseases = []
    for disease, symptoms in disease_symptom_dict.items():
        if set(user_symptoms).issubset(set(symptoms)):
            potential_diseases.append(disease)
    potential_diseases = set(potential_diseases)
    potential_diseases = list(potential_diseases)
    print("POTENTIAL", potential_diseases)
    
    diseases_same = []
    for disease in potential_diseases:
        dsymptoms = disease_symptom_dict[disease]
        check = all(item in dsymptoms for item in user_symptoms)
        if check is True:
            print()
            diseases_same.append(disease)
            print()
    
    print("TOP", top_diseases)
    for i in top_diseases:
        diseases_same.append(i)
    
    diseases_same = set(diseases_same)
    diseases_same = list(diseases_same)
    count = len(diseases_same)
    
    print()
    print("The predicted disease according to the model is ", predicted_disease)
        
    # Get disease description and precautions
    disease_description = get_disease_description(predicted_disease)
    disease_precautions = get_disease_precautions(predicted_disease)
    
    # Generate detailed explanation
    explanation = f"Based on the given symptoms, the model predicts the disease as '{predicted_disease}'.\n\n"

    # Symptom Analysis
    explanation += "Symptom Analysis:\n"
    explanation += "The model considers the following symptoms and their severities:\n"
    for symptom, severity in symptom_severity_dict.items():
        if symptom in user_symptoms:
            symptom_description = symptom_dict.get(symptom, "Description not available")
            explanation += f"- {symptom}: {symptom_description}. Severity: {severity}\n"
    explanation += "\n"

    # Disease Description
    explanation += "Description:\n"
    explanation += f"{disease_description}\n\n"

    # Precautions
    explanation += "Precautions:\n"
    for i, precaution in enumerate(disease_precautions):
        if (precaution!=""):
            explanation += f"{i + 1}. {precaution}\n"

    return explanation

def get_disease_description(disease):
    description = disease_description_df.loc[disease_description_df['Disease'] == disease, 'Description'].values[0]
    return description

# Function to get disease precautions
def get_disease_precautions(disease):
    precautions = disease_precaution_df.loc[disease_precaution_df['Disease'] == disease].drop('Disease', axis=1).values[0]
    # Apply the function to the numpy array
    precautions = np.vectorize(capitalize_and_remove_underscore)(precautions)
    
    return precautions

# Function to provide rule-based explanations
def get_explanations(symptoms):
    explanations = []
    for disease, symptoms_list in disease_symptom_dict.items():
        symptoms_list = set(symptoms_list)
        if set(symptoms).issubset(symptoms_list):
            explanations.append(f"If {', '.join(symptoms_list)}, then {disease}")
    return explanations

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        input_message = request.form.get('message')

        if not input_message:
            return jsonify({'error': 'No message provided.'}), 400

        # Split the input message into symptoms based on a delimiter (e.g., comma or space)
        patient_symptoms = [symptom.strip().replace(' ', '_').lower() for symptom in input_message.split(',')]
        predicted_disease = predict_disease(patient_symptoms)
        explanation = explain_decision(patient_symptoms, patient_symptoms)
        explanations = get_explanations(patient_symptoms)

        # Print the explanations
        if explanations:
            print("Rule based explanations based on entered symptoms:")
            for expln in explanations:
                print(f"- {expln}")
        else:
            print("No rule based explanations found for the given symptoms.\n")

        print('\nPatient Symptoms:')
        for symptom in patient_symptoms:
            print(symptom)
        print('\nPredicted Disease:\n', predicted_disease)

        print('\nExplanation:\n', explanation)

        return render_template('chat.html', input_message=input_message, result=explanation)
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
