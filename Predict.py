import pickle
from preprocessing import chuanhoa
import numpy as np

print("Các thông số cần nhập : Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age ")
input_string = input("Hãy nhập thông số của bạn theo thứ tự : " )
array1 = np.array(input_string)
array2 = array1.reshape(1, -1)
std_data = chuanhoa(array2)

loaded_model=pickle.load(open("diabetesmodel.sav",'rb'))

prediction = loaded_model.predict(std_data)
if(prediction[0] == 0):
    print("This person is Non-Diabetic")
else:
    print("This person is Diabetic")