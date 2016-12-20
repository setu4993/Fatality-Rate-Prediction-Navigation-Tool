# Fatality Rate Prediction Navigation Tool
A neural network algorithm developed to predict the fatality rate of various possible route options.

This tool was created to help drivers selecting routes that are safe and not just fast. Developed an algorithm that uses the current time of the day, date, weather conditions, light conditions and location information to predict the fatality rate, and then compare each of the 3 possible routes recommended by the Google Maps API (https://developers.google.com/maps/).

The training is done by using the back propagation neural network and the National Highway Transportation and Safety Administration's Fatality Analysis Reporting System database (https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars).

The weather information is obtained from a free license of OpenWeatherMap API (https://openweathermap.org/). Used the cURL (https://curl.haxx.se/) library to connect the tool to the APIs by an HTTP connection. Used the tinyXML-2 (http://www.grinninglizard.com/tinyxml2/) library to parse the XML response from the APIs.
