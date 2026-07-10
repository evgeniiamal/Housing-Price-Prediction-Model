import joblib
import pandas as pd

print("The model is loading...")
# Загрузка обученной модели
model = joblib.load("model/best_model.pkl")

# Ввод характеристик пользователем
print("Please, fill it out in English")
type = input("Apartment type (Secondary/New building): ")
metro_st = input("Metro station: ")
min_to_metro = float(input("Minutes to metro: "))
region = input("Region: ")
rooms = float(input("Number of rooms: "))
area = float(input("Total area (m²): "))
l_area = float(input("Living area (m²): "))
k_area = float(input("Kitchen area (m²): "))
floor = float(input("Floor: "))
num_of_floors = int(input("Number of floors: "))
renovation = input("Renovation: ")

# Создание DataFrame
new_flat = pd.DataFrame([{
    "Apartment type": type,
    "Metro station": metro_st,
    "Minutes to metro": min_to_metro,
    "Region": region,
    "Number of rooms": rooms,
    "Area": area,
    "Living area": l_area,
    "Kitchen area": k_area,
    "Floor": floor,
    "Number of floors": num_of_floors,
    "Renovation": renovation}])


predicted_price = model.predict(new_flat)[0]

print(f"Ожидаемая цена: {predicted_price:,.0f} руб")