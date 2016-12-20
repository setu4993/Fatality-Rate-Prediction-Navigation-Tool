#define _CRT_SECURE_NO_WARNINGS

#include "tinyxml2.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <time.h>
#include <vector>
#include "curl\curl.h"

#define input_nodes 14
#define layers 3
#define output_nodes 2
#define hidden_layer_nodes 5

double inp[input_nodes + 1];	//input layer node values + bias
double hid[hidden_layer_nodes + 1];	//hidden layer node values + bias
double wih[input_nodes + 1][hidden_layer_nodes];	//weight matrix for (input + bias) -> hidden layer
double who[hidden_layer_nodes + 1][output_nodes];	//weight matrix for (hidden layer + bias) -> output
double op[output_nodes]; //output values

std::ifstream weights("trained_weights.txt");

double start_lat, start_lng, end_lat, end_lng;
int route_weight[3];
int route_total_calls[3];
double route_safety_factor[3];
int rtc = 0; //route count
std::vector<std::string> route_name;
int route_time[3];
int w = 0;

double hid_min = 10, hid_max = -10, op_min = 0, op_max = 0;

char *weather_xml = "weather_output.xml";

using namespace std;
using namespace tinyxml2;

void map_read(char *);
int run_bpnn_step(double, double);
void set_lat_long_inputs(double, double);
void get_weather_xml(double, double);
void set_weather_inputs();
void set_input_random(int);
void set_inputs();
double run_network();
void print_output();
void set_weights();
void printweights();
void printinputs();

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp);

int main()
{
	char src[50], dst[50];
	set_weights();
	//printweights();
	cout << "Enter source: ";
	cin >> src;

	cout << "Enter destination: ";
	cin >> dst;

	cout << "\nYour journey is from " << src << " to " << dst << ".\n";

	string map_url("https://maps.googleapis.com/maps/api/directions/xml?alternatives=true&key=YOUR_GOOGLE_MAPS_KEY&origin=");
	map_url.append(src);
	map_url.append("&destination=");
	map_url.append(dst);

	//cout << "\n" << map_url << std::endl;

	CURL *curl;
	CURLcode res;

	fstream fs;
	string readBuffer;

	char *map_xml = "map_output.xml";

	fs.open(map_xml, std::fstream::out | std::fstream::trunc);

	curl = curl_easy_init();
	if (curl)
	{
		curl_easy_setopt(curl, CURLOPT_URL, map_url);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);

		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);

		fs << readBuffer;
		fs.close();
		curl_easy_cleanup(curl);
	}

	map_read(map_xml);

	print_output();
	getchar(); getchar();
	return 0;
}

void map_read(char *map_xml)
{
	tinyxml2::XMLDocument doc;
	doc.LoadFile(map_xml);
	const char * tmp;
	char * temp;
	char str[30] = { "" };

	const char* title;

	XMLElement *routeElement = doc.FirstChildElement("DirectionsResponse");
	for (XMLElement* child = routeElement->FirstChildElement("route"); child != NULL; child = child->NextSiblingElement("route"))
	{
		title = child->FirstChildElement("summary")->GetText();
		//printf("Summary: %s\n", title);
		route_name.push_back(title);
		//route_opt[i] = title;

		XMLElement *stepElement = child->FirstChildElement("leg");

		for (XMLElement* stepChild = stepElement->FirstChildElement("step"); stepChild != NULL; stepChild = stepChild->NextSiblingElement("step"))
		{
			tmp = stepChild->FirstChildElement("start_location")->FirstChildElement("lat")->GetText();
			start_lat = atof(tmp);
			tmp = stepChild->FirstChildElement("start_location")->FirstChildElement("lng")->GetText();
			start_lng = atof(tmp);
			
			tmp = stepChild->FirstChildElement("end_location")->FirstChildElement("lat")->GetText();
			end_lat = atof(tmp);
			tmp = stepChild->FirstChildElement("end_location")->FirstChildElement("lng")->GetText();
			end_lng = atof(tmp);

			route_weight[rtc] += run_bpnn_step(start_lat, start_lng);

			if ((abs(end_lat - start_lat) > 0.1) || (abs(end_lng - start_lng) > 0.1))
			{
				int n = (abs(start_lat - end_lat) + abs(start_lng - end_lng)) * 10;
				
				double diff_lat = (end_lat - start_lat) / n;
				double diff_lng = (end_lng - start_lng) / n;

				for (int l = 1; l < n; l++)
				{
					route_weight[rtc] += run_bpnn_step((start_lat + (l * diff_lat)), (start_lng + (l * diff_lng)));
				}
				//cout << "Route " << (rtc + 1) << " weight so far: " << route_weight[rtc];
			}

			

			//route_weight[rtc] += run_bpnn_step(start_lat, start_lng);

			tmp = stepChild->FirstChildElement("duration")->FirstChildElement("text")->GetText();
			//cout << "\n\n  " << tmp;

			strcpy(str, tmp);
			//cout << "\n\n  " << str;
			temp = strtok(str, " ");
			int step_time = 0, time_tmp = 0;
			while (temp != NULL)
			{
				//printf("%s\n", temp);
				if (atoi(temp) > 0)
				{
					time_tmp = atoi(temp);
				}
				//cout << "\n\n" << time_tmp;

				if ((strcmp(temp, "min") == 0) || (strcmp(temp, "mins") == 0))
				{
					step_time = step_time + time_tmp;
				}
				else if ((strcmp(temp, "hour") == 0) || (strcmp(temp, "hours") == 0))
				{
					step_time = step_time + (time_tmp * 60);
				}
				
				temp = strtok(NULL, " ");
			}
			route_time[rtc] += step_time;
		}
		
		rtc++;
	}
}

void set_lat_long_inputs(double lat, double lng)
{
	inp[5] = lat;
	inp[6] = lng;
}

int run_bpnn_step(double lat, double lng)
{
	int local_wt = 0;
	
	get_weather_xml(lat, lng);
	set_lat_long_inputs(lat, lng);
	set_weather_inputs();
	set_inputs();
	//printinputs();
	local_wt = run_network();

	return local_wt;
}

void get_weather_xml(double lat, double lng)
{
	ostringstream strs;
	string weather_url("http://api.openweathermap.org/data/2.5/weather?mode=xml&lat=");
	
	strs << lat;
	string str = strs.str();
	weather_url.append(str);
	weather_url.append("&lon=");
	
	strs.str("");
	strs.clear();
	strs << lng;
	str = strs.str();
	weather_url.append(str);

	if(w == 0)
	{
		weather_url.append("&appid=YOUR_OPENWEATHERMAP_API_KEY");
		w = 1;
	}
	else if (w == 1)
	{
		weather_url.append("&appid=YOUR_OPENWEATHERMAP_API_KEY");
		w = 2;
	}
	else if (w == 2)
	{
		weather_url.append("&appid=YOUR_OPENWEATHERMAP_API_KEY");
		w = 3;
	}
	else if (w == 3)
	{
		weather_url.append("&appid=YOUR_OPENWEATHERMAP_API_KEY");
		w = 4;
	}
	else if (w == 4)
	{
		weather_url.append("&appid=YOUR_OPENWEATHERMAP_API_KEY");
		w = 5;
	}
	else
	{
		weather_url.append("&appid=YOUR_OPENWEATHERMAP_API_KEY");
		w = 0;
	}

	CURL *curl;
	CURLcode res;

	fstream fs;
	string writeBuffer;

	fs.open(weather_xml, std::fstream::out | std::fstream::trunc);

	curl = curl_easy_init();
	if (curl)
	{
		curl_easy_setopt(curl, CURLOPT_URL, weather_url);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);

		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &writeBuffer);
		res = curl_easy_perform(curl);

		fs << writeBuffer;
		fs.close();
		curl_easy_cleanup(curl);
	}	
}


void set_weather_inputs()
{
	char str[20] = { "" };

	char * tmp;

	int srse_time[3], sset_time[3], curr_time[5], local_time[5];
	int i = 0;
	int sunrise, sunset, current;

	//get and store current GMT time
	time_t rawtime;
	struct tm * ptm;
	struct tm * ctm;

	time(&rawtime);
	ptm = gmtime(&rawtime);

	ctm = localtime(&rawtime);

	curr_time[0] = ptm->tm_mday; //date 1-31
	curr_time[1] = 1 + ptm->tm_mon; //month 0-11
	curr_time[2] = 1 + ptm->tm_wday; //day of the week 0-6
	curr_time[3] = ptm->tm_hour; //hours 0-23
	curr_time[4] = ptm->tm_min; //mins 0-59

	local_time[0] = ctm->tm_mday; //date 1-31
	local_time[1] = 1 + ctm->tm_mon; //month 0-11
	local_time[2] = 1 + ctm->tm_wday; //day of the week 0-6
	local_time[3] = ctm->tm_hour; //hours 0-23
	local_time[4] = ctm->tm_min; //mins 0-59

	//set date, time inputs
	for (int j = 0; j < 5; j++)
	{
		inp[i] = local_time[i];
	}

	current = (curr_time[3] * 60) + curr_time[4]; //current time in mins

	//cout << "Current: " << curr_time[0] << "  " << curr_time[1] << "  " << curr_time[2] << "\n";
	
	//load XML file
	tinyxml2::XMLDocument doc;
	doc.LoadFile(weather_xml);


	//get sunset time
	const char* title = doc.FirstChildElement("current")->FirstChildElement("city")->FirstChildElement("sun")->Attribute("rise");
	//cout << "Name of play (1): \n" << title1;

	strcpy(str, title);

	tmp = strtok(str, "T");
	strcpy(str, "");
	tmp = strtok(NULL, ":");
	while (tmp != NULL)
	{
		//printf("%s\n", tmp);
		srse_time[i++] = atoi(tmp);
		tmp = strtok(NULL, ":");
	}

	sunrise = (srse_time[0] * 60) + srse_time[1]; //sunrise time in mins


	//cout << "Start: " << strt_time[0] << "  " << strt_time[1] << "  " << strt_time[2] << "\n";

	//get sunset time
	title = doc.FirstChildElement("current")->FirstChildElement("city")->FirstChildElement("sun")->Attribute("set");
	
	//cout << "Name of play (1): \n" << title;

	strcpy(str, title);

	tmp = strtok(str, "T");
	strcpy(str, "");
	tmp = strtok(NULL, ":");
	while (tmp != NULL)
	{
		//printf("%s\n", tmp);
		sset_time[i++] = atoi(tmp);
		tmp = strtok(NULL, ":");
	}
	sunset = (sset_time[0] * 60) + sset_time[1]; //sunset time in mins


												  //get weather conditions
	title = doc.FirstChildElement("current")->FirstChildElement("weather")->Attribute("number");
	//printf("Name of play (1): %s\n", num);
	//cout << "Name of play (1): " << title;

	int id = atoi(title);

	//set light conditions depending on time
	if (abs(sunset - current) < 60)
	{
		inp[7] = (rand() % 25) / (double)100;
		inp[8] = (rand() % 25 + 50) / (double)100;	//dusk
	}
	else if (abs(sunrise - current) < 60)
	{
		inp[7] = (rand() % 25 + 25) / (double)100;
		inp[8] = (rand() % 25 + 25) / (double)100;
		//inp[7] = 0.5;
		//inp[8] = 0.5;	//dawn
	}
	else if ((current > sunset) || (current < sunrise))
	{
		set_input_random(7);
		//inp[7] = (rand() % 100 - 100) / (double)100;
		inp[8] = (rand() % 25 + 75) / (double)100;
		//inp[8] = 1;	//dark
	}
	else if ((current < sunset) || (current > sunrise))
	{
		inp[7] = (rand() % 25 + 75) / (double)100;	//daylight
		set_input_random(8);
	}

	//set weather inputs according to weather conditions
	switch (id)
	{
	case 800:
	case 801:
	case 904:
	case 951:
	case 952:
	case 953:
	case 954:
	case 955:
		//inp[9] = 1;
		inp[9] = (rand() % 25 + 75) / (double)100;	//clear sky
		set_input_random(10);
		set_input_random(11);
		set_input_random(12);
		set_input_random(13);
		break;

	case 500:
	case 501:
	case 300:
	case 301:
	case 310:
	case 311:
	case 313:
	case 321:
	case 520:
		//inp[10] = 0.5;
		//inp[11] = 0.5;	//drizzle = rain 0.5
		inp[10] = (rand() % 25 + 25) / (double)100;
		inp[11] = (rand() % 25 + 25) / (double)100;
		set_input_random(9);
		set_input_random(12);
		set_input_random(13);
		break;
	case 511:
	case 615:
	case 616:
		inp[10] = (rand() % 25 + 25) / (double)100;
		inp[11] = (rand() % 25 + 25) / (double)100;
		//inp[10] = 0.5;
		//inp[11] = 0.5;	//freezing rain
		set_input_random(9);
		set_input_random(12);
		set_input_random(13);
		break;
	case 502:
	case 503:
	case 504:
	case 521:
	case 522:
	case 531:
	case 302:
	case 312:
	case 314:
		inp[10] = (rand() % 25 + 75) / (double)100;
		//inp[10] = 1;	//rain 1.0
		set_input_random(9);
		set_input_random(11);
		set_input_random(12);
		set_input_random(13);
		break;
	case 906:
	case 611:
	case 612:
		inp[11] = (rand() % 25 + 75) / (double)100;
		//inp[11] = 1;	//sleet hail
		set_input_random(9);
		set_input_random(10);
		set_input_random(12);
		set_input_random(13);
		break;
	case 600:
	case 620:
	case 903:
		inp[11] = (rand() % 25 + 25) / (double)100;
		//inp[11] = 0.5;	//snow 0.5
		set_input_random(9);
		set_input_random(10);
		set_input_random(12);
		set_input_random(13);
		break;
	case 601:
	case 602:
	case 621:
	case 622:
		inp[11] = (rand() % 25 + 75) / (double)100;
		//inp[11] = 1;	//snow 1.0
		set_input_random(9);
		set_input_random(10);
		set_input_random(12);
		set_input_random(13);
		break;
	case 701:
	case 711:
	case 721:
	case 741:
		inp[12] = (rand() % 25 + 25) / (double)100;
		//inp[12] = 1;	//fog, smog, smoke
		set_input_random(9);
		set_input_random(11);
		set_input_random(10);
		set_input_random(13);
		break;
	case 802:
	case 803:
	case 804:
		inp[12] = (rand() % 25 + 25) / (double)100;
		//inp[12] = 0.5;	//cloudy
		set_input_random(9);
		set_input_random(11);
		set_input_random(10);
		set_input_random(13);
		break;
	case 200:
	case 201:
	case 202:
	case 210:
	case 211:
	case 212:
	case 221:
	case 230:
	case 231:
	case 232:
	case 731:
	case 751:
	case 761:
	case 762:
	case 771:
	case 781:
	case 900:
	case 901:
	case 902:
	case 905:
	case 956:
	case 957:
	case 958:
	case 959:
	case 960:
	case 961:
	case 962:
		inp[13] = (rand() % 25 + 75) / (double)100;
		//inp[13] = 1;	//Severe Crosswinds / Blowing sand, soil, dirt / Blowing snow
		set_input_random(9);
		set_input_random(11);
		set_input_random(12);
		set_input_random(10);
		break;
	default:
		break;
	}

}

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
}

void set_input_random(int i)
{
	inp[i] = (rand() % 100 - 100) / (double)100;
}

void set_inputs()
{
	int i;
	for (i = 0; i<input_nodes; i++)
	{
		switch (i)
		{
		case 0:
			inp[i] = inp[i] - 15; // 1 / (double)(1 + exp(-0.18 * (tmp[i] - 15)));	//date
								  //inp[i] = tmp[i] / (double)(1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 8)))) - 1;
			break;
		case 1:
			inp[i] = inp[i] - 6; // / (1 + exp(-0.18 * (tmp[i] - 15)));	//month
								 //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 3)))) - 1;
			break;
		case 2:
			inp[i] = inp[i] - 3.5; // / (1 + exp(-0.18 * (tmp[i] - 15)));	//day of the week
								   //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 2)))) - 1;
			break;
		case 3:
			inp[i] = inp[i] - 12; // / (1 + exp(-0.18 * (tmp[i] - 15)));	//hour
								  //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 6)))) - 1;
			break;
		case 4:
			inp[i] = inp[i] - 30;// / (1 + exp(-0.18 * (tmp[i] - 15)));	//minutes
								 //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 15)))) - 1;
			break;
		case 5:
			inp[i] = inp[i] - 44.00; //latitude
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 13)))) - 1;
			break;
		case 6:
			inp[i] = inp[i] + 115.00; //longitude
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 24)))) - 1;
			break;
		}
	}

	inp[input_nodes] = 1;

	for (int j = 0; j < hidden_layer_nodes; j++)
	{
		hid[j] = 0;
	}

	hid[hidden_layer_nodes] = 1;

	for (int j = 0; j < output_nodes; j++)
	{
		op[j] = 0;
	}
	
}

double run_network()
{
	int i, j;

	//calculation of hidden layer

	for (j = 0; j < hidden_layer_nodes; j++)
	{
		for (i = 0; i <= input_nodes; i++)
		{
			hid[j] = hid[j] + (wih[i][j] * inp[i]);
		}

	}
	for (j = 0; j < hidden_layer_nodes; j++)
	{
		//cout << "\n hid " << hid[j];
		//hid[j] = 1.0 / (1.0 + exp(-1 * hid[j]));	//sigmoid
		//hid[j] = hid[j] / (1.0 + abs(hid[j]));	//softsign
												//cout << "\n hid " << hid[j];
		//hid[j] = (2.0000 / (1.0000 + exp(-2 * hid[j]))) - 1; //tanh
		hid[j] = (2.0000 / (1.0000 + exp(-2 * hid[j] / 10))) - 1; //tanh x/10
	}


	//calculation of output layer
	for (j = 0; j <= hidden_layer_nodes; j++)
	{
		for (i = 0; i < output_nodes; i++)
		{
			op[i] = op[i] + (who[j][i] * hid[j]);
		}
	}

	for (j = 0; j < output_nodes; j++)
	{
		//cout << "\nCheck " << op[j];
		//op[j] = op[j] / (1.0 + abs(op[j]));	//softsign at the output
		//op[j] = 1.0 / (1.0 + exp(-1 * op[j]));	//sigmoid at the output
		//op[j] = op[j] * 10;
		//op[j] = (2.0000 / (1.0000 + exp(-2 * op[j]))) - 1; //tanh
		
		op[j] = (2.0000 / (1.0000 + exp(-2 * ((3.5 * op[j]) - 1)))) - 1;  //tanh (3.5x - 1)
		//cout << "\t" << op[j];
		//hid_min = fmin(op[1], hid_min);
		//hid_max = fmax(op[1], hid_max);

		if (op[j] <= 0.5)
		{
			op[j] = 0;
		}
		else
		{
			//cout << "\n" << op[j];
			op[j] = 1;
			
		}
	}

	//cout << "\nReturning " << op[1];

	route_total_calls[rtc]++;
	return op[1];
}

void print_output()
{

	for (int j = 0; j < rtc; j++)
	{
		route_safety_factor[j] = (double)route_weight[j] / route_total_calls[j];
	}
	//sort the outputs
	for (int j = 0; j < rtc; j++)
	{
		for (int k = (j + 1); k < rtc; k++)
		{
			if (route_safety_factor[j] > route_safety_factor[k])
			{
				int tmp = route_weight[j];
				route_weight[j] = route_weight[k];
				route_weight[k] = tmp;

				string rt_tmp = route_name[j];
				route_name[j] = route_name[k];
				route_name[k] = rt_tmp;

				tmp = route_time[j];
				route_time[j] = route_time[k];
				route_time[k] = tmp;

				double swp = route_safety_factor[j];
				route_safety_factor[j] = route_safety_factor[k];
				route_safety_factor[k] = swp;
			}
		}
	}

	for (int j = 0; j < rtc; j++)
	{
		cout << "\n\nThe route with the summary '" << route_name[j] << "' ";

		if (j == 0)
		{
			cout << "is the safest route ";
		}
		else if (j == 1)
		{
			cout << "is a safer route ";
		}
		else
		{
			cout << "is a safe route ";
		}
		
		cout << "with a fatality factor of " << route_safety_factor[j] << " and will take " << (int)(route_time[j] / 60) << " hour(s) and " << (route_time[j] % 60) << " minute(s) to the destination."; //<< route_weight[j]  << " / " << route_total_calls[j] << " = " 

		//cout << "\nHid Min " << hid_min << "\tHid Max " << hid_max;
	}
}

void set_weights()
{
	int i = 0, j = 0, k = 0;
	while (!weights.eof())
	{
		string value;
		getline(weights, value);
		//float tmp;
		char seps[] = ",";
		char *token;

		token = strtok(&value[0], seps);
		//std::cout << "\n" << token << "\n";
		while (token != NULL)
		{
			if (i < (input_nodes + 1))
			{
				if (j < hidden_layer_nodes)
				{
					wih[i][j++] = atof(token);
					if (j == hidden_layer_nodes)
					{
						i++;
						j = 0;
					}
				}
			}
			else if (k < (hidden_layer_nodes + 1))
			{
				if (j < output_nodes)
				{
					who[k][j++] = atof(token);
					if (j == output_nodes)
					{
						k++;
						j = 0;
					}
				}
			}
			token = strtok(NULL, ",");
		}
	}
}

void printweights()
{
	int i, j;

	cout << "\nI -> H\n";
	for (i = 0; i <= input_nodes; i++)
	{
		for (j = 0; j < hidden_layer_nodes; j++)
		{
			cout << wih[i][j];
			if (j != (hidden_layer_nodes - 1))
			{
				cout << ",";
			}
		}
		cout << "\n";
	}
	cout << "\n\nH -> O\n";
	for (i = 0; i <= hidden_layer_nodes; i++)
	{
		for (j = 0; j < output_nodes; j++)
		{
			cout << who[i][j];
			if (j != (output_nodes - 1))
			{
				cout << ",";
			}
		}
		cout << "\n";
	}

}

void printinputs()
{
	int i;
	cout << "\nInput Nodes:\n";
	for (i = 0; i <= input_nodes; i++)
	{
		cout << "\t" << inp[i];
	}
	cout << "\nHidden Nodes:\n";
	for (i = 0; i <= hidden_layer_nodes; i++)
	{
		cout << "\t" << hid[i];
	}
}