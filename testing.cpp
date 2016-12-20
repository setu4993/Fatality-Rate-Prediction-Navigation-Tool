#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <stdlib.h>

using namespace std;

#define input_nodes 14
#define layers 3
#define output_nodes 2
#define hidden_layer_nodes 5

double wih[input_nodes + 1][hidden_layer_nodes];	//weight matrix for (input + bias) -> hidden layer
double who[hidden_layer_nodes + 1][output_nodes];	//weight matrix for (hidden layer + bias) -> output
double inp[input_nodes + 1];	//input layer node values + bias
int tmp[input_nodes];
double hid[hidden_layer_nodes + 1];	//hidden layer node values + bias
double tv[output_nodes];	// target value
double op[output_nodes]; //output values
double alpha = 0.9;	//learning rate
double deltaho[output_nodes];
double deltah[hidden_layer_nodes];
double delWho[hidden_layer_nodes + 1][output_nodes];
double delin[hidden_layer_nodes];
double delWih[input_nodes + 1][hidden_layer_nodes];
long maxEpoch = 1;
int tst;
double crr = 0, err = 0;

double mse = 0;
double initial_weight = 1;
double hid_min = 10, hid_max = -10, op_min = 0, op_max = 0;

ifstream file("Testing9.csv");
ifstream out("trained_weights.txt");
string value;

void create_network();
void clear_values();
void run_network();
void recal_weights();
void next_iter();
void printweights();
void printinputs();
void set_weights();
void test_network();

void main()
{
	int j = 0;
	long epoch = 0;
	//create_network();
	//printweights();
	set_weights();
	printweights();
	//clear_values();
	//printinputs();
	//file.open("test4.csv", std::ifstream::in);

	while (!file.eof())
	{
		getline(file, value);
		next_iter();
		run_network();
		test_network();
		clear_values();
	}
	file.clear();
	file.close();

	cout << "\n\nCorrect: " << crr << "\n\nError: " << err;

	getchar();
}

void create_network()
{
	int i, j, tmp;

	//initializing all weights randomly

	for (i = 0; i <= input_nodes; i++)
	{
		for (j = 0; j < hidden_layer_nodes; j++)
		{
			tmp = rand() % 200 - 100;
			wih[i][j] = (double)tmp / 100;
			/*
			if ((i + j) % 2)
			{
			wih[i][j] = (double)(i + j)*(-1) / (input_nodes + hidden_layer_nodes);
			}
			else
			{
			wih[i][j] = (double)(i + j) / (input_nodes + hidden_layer_nodes);
			}
			*/
		}
	}

	for (i = 0; i <= hidden_layer_nodes; i++)
	{
		for (j = 0; j < output_nodes; j++)
		{
			tmp = rand() % 200 - 100;
			who[i][j] = (double)tmp / 100;
		}
	}

	//clear();
}

void clear_values()
{
	//clear all input and hidden node values (biases as 1)
	int i = 0, j = 0;
	for (i = 0; i < input_nodes; i++)
	{
		inp[i] = 0;
	}
	inp[i] = 1;	//bias

	for (i = 0; i < hidden_layer_nodes; i++)
	{
		hid[i] = 0;
		deltah[j] = 0;
		delin[i] = 0;
	}
	hid[i] = 1;	//bias

				//clear all output and target values
	for (i = 0; i < output_nodes; i++)
	{
		op[i] = 0;
		tv[i] = 0;
		deltaho[i] = 0;
	}
}

void next_iter()
{
	int i = 0;
	int temp = 0;
	char seps[] = ",";
	char *token;

	token = strtok(&value[0], seps);
	//std::cout << "\n" << token << "\n";
	while (token != NULL)
	{
		tmp[i] = atof(token);
		switch (i)
		{
		case 0:
			inp[i] = tmp[i] - 15; // 1 / (double)(1 + exp(-0.18 * (tmp[i] - 15)));	//date
								  //inp[i] = tmp[i] / (double)(1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 8)))) - 1;
			break;
		case 1:
			inp[i] = tmp[i] - 6; // / (1 + exp(-0.18 * (tmp[i] - 15)));	//month
								 //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 3)))) - 1;
			break;
		case 2:
			inp[i] = tmp[i] - 3.5; // / (1 + exp(-0.18 * (tmp[i] - 15)));	//day of the week
								   //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 2)))) - 1;
			break;
		case 3:
			inp[i] = tmp[i] - 12; // / (1 + exp(-0.18 * (tmp[i] - 15)));	//hour
								  //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 6)))) - 1;
			break;
		case 4:
			inp[i] = tmp[i] - 30;// / (1 + exp(-0.18 * (tmp[i] - 15)));	//minutes
								 //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 15)))) - 1;
			break;
		case 5:
			inp[i] = atof(token) - 44.00; //latitude
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 13)))) - 1;
			break;
		case 6:
			inp[i] = atof(token) + 115.00; //longitude
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 24)))) - 1;
			break;
		case 8:
			switch (tmp[i])
			{
			case 1:
				//inp[7] = 1; //daylight
				inp[7] = (rand() % 25 + 75) / (double)100;
				inp[8] = (rand() % 100 - 100) / (double)100;
				break;
			case 2:
			case 6:
				inp[7] = (rand() % 100 - 100) / (double)100;
				//inp[8] = 1; //dark
				inp[8] = (rand() % 25 + 75) / (double)100;
				break;
			case 3:
				inp[7] = (rand() % 25) / (double)100;
				inp[8] = (rand() % 25 + 50) / (double)100;
				//inp[7] = 0.25;
				//inp[8] = 0.75;
				break;
			case 4:
			case 5:
				inp[7] = (rand() % 25 + 25) / (double)100;
				inp[8] = (rand() % 25 + 25) / (double)100;
				//inp[7] = 0.5;
				//inp[8] = 0.5;
				break;
			default:
				break;
			}
			break;
		case 9: switch (tmp[i])
		{
		case 1:
			//inp[9] = 1;	//clear sky
			inp[9] = (rand() % 25 + 75) / (double)100;
			inp[10] = (rand() % 100 - 100) / (double)100;
			inp[11] = (rand() % 100 - 100) / (double)100;
			inp[12] = (rand() % 100 - 100) / (double)100;
			inp[13] = (rand() % 100 - 100) / (double)100;
			break;
		case 2:
			//inp[10] = 1;	//rain
			inp[10] = (rand() % 25 + 75) / (double)100;
			inp[9] = (rand() % 100 - 100) / (double)100;
			inp[11] = (rand() % 100 - 100) / (double)100;
			inp[12] = (rand() % 100 - 100) / (double)100;
			inp[13] = (rand() % 100 - 100) / (double)100;
			break;
		case 3:
			//inp[11] = 1;	//Sleet / Hail
			inp[11] = (rand() % 25 + 75) / (double)100;
			inp[10] = (rand() % 100 - 100) / (double)100;
			inp[9] = (rand() % 100 - 100) / (double)100;
			inp[12] = (rand() % 100 - 100) / (double)100;
			inp[13] = (rand() % 100 - 100) / (double)100;
			break;
		case 12:
			//inp[10] = 0.5;
			//inp[11] = 0.5;	//freezing rain or drizzle
			inp[10] = (rand() % 25 + 25) / (double)100;
			inp[11] = (rand() % 25 + 25) / (double)100;
			inp[9] = (rand() % 100 - 100) / (double)100;
			inp[12] = (rand() % 100 - 100) / (double)100;
			inp[13] = (rand() % 100 - 100) / (double)100;
			break;
		case 4:
			//inp[11] = 0.5;	//snow
			inp[11] = (rand() % 25 + 25) / (double)100;
			inp[10] = (rand() % 100 - 100) / (double)100;
			inp[9] = (rand() % 100 - 100) / (double)100;
			inp[12] = (rand() % 100 - 100) / (double)100;
			inp[13] = (rand() % 100 - 100) / (double)100;
			break;
		case 5:
			//inp[12] = 1;	//fog, smog, smoke
			inp[12] = (rand() % 25 + 75) / (double)100;
			inp[10] = (rand() % 100 - 100) / (double)100;
			inp[11] = (rand() % 100 - 100) / (double)100;
			inp[9] = (rand() % 100 - 100) / (double)100;
			inp[13] = (rand() % 100 - 100) / (double)100;
			break;
		case 10:
			//inp[12] = 0.5;	//cloudy
			inp[12] = (rand() % 25 + 25) / (double)100;
			inp[10] = (rand() % 100 - 100) / (double)100;
			inp[11] = (rand() % 100 - 100) / (double)100;
			inp[9] = (rand() % 100 - 100) / (double)100;
			inp[13] = (rand() % 100 - 100) / (double)100;
			break;
		case 6:
		case 7:
		case 11:
			//inp[13] = 1;	//Severe Crosswinds / Blowing sand, soil, dirt / Blowing snow
			inp[13] = (rand() % 25 + 75) / (double)100;
			inp[10] = (rand() % 100 - 100) / (double)100;
			inp[11] = (rand() % 100 - 100) / (double)100;
			inp[12] = (rand() % 100 - 100) / (double)100;
			inp[9] = (rand() % 100 - 100) / (double)100;
			break;
		}
				break;

		case 10:
			if (atof(token) < 0.95)
			{
				tv[0] = 1;	//low
			}
			else
			{
				tv[1] = 1; //high
			}
			//tv = atof(token);
			break;
		default: break;
		}

		token = strtok(NULL, ",");
		i++;
	}
	/*
	for (i = 0; i < input_nodes; i++)
	{
	if (inp[i] == 0)
	{
	temp = rand() % 100 - 100;
	inp[i] = (double)temp / 100;
	}
	else if (inp[i] == 1)
	{
	temp = rand() % 100;
	inp[i] = (double)temp / 100;
	}
	}
	*/
	for (i = 0; i < hidden_layer_nodes; i++)
	{
		hid[i] = 0;
	}
	hid[hidden_layer_nodes] = 1;	//bias	
}

void run_network()
{
	int i, j;

	//calculation of hidden layer

	//cout << "\n";

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
		//cout << "\nHidden " << hid[j];

		//hid_min = fmin(hid[j], hid_min);
		//hid_max = fmax(hid[j], hid_max);
		//hid[j] = (2.0000 / (1.0000 + exp(-2 * hid[j]))) - 1; //tanh
		hid[j] = (2.0000 / (1.0000 + exp(-2 * hid[j] / 10))) - 1; //tanh x/10
																  //cout << "\t" << hid[j];
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
		//hid_min = fmin(op[j], hid_min);
		//hid_max = fmax(op[j], hid_max);
		//cout << "\n" << op[j];
		//op[j] = op[j] / (1.0 + abs(op[j]));	//softsign at the output
		//op[j] = 1.0 / (1.0 + exp(-1 * op[j]));	//sigmoid at the output
		//op[j] = (2.0000 / (1.0000 + exp(-2 * op[j]))) - 1; //tanh
		op[j] = (2.0000 / (1.0000 + exp(-2 * ((3.5 * op[j]) - 1)))) - 1;  //tanh (3.5x - 1)
																 //hid_min = fmin(op[j], hid_min);
																 //hid_max = fmax(op[j], hid_max);
																 //cout << "\t" << op[j];
		if (op[j] <= 0.95)
		{
			op[j] = 0;
		}
		else
		{
			op[j] = 1;
		}
	}
	//cout << "\nHid Min " << hid_min << "\tHid Max " << hid_max;
}

void printweights()
{
	int i, j;

	for (i = 0; i <= input_nodes; i++)
	{
		cout << "\nI -> H\n";
		for (j = 0; j < hidden_layer_nodes; j++)
		{
			cout << "\t" << wih[i][j];
		}
	}

	for (i = 0; i <= hidden_layer_nodes; i++)
	{
		cout << "\n\nH -> O\n";
		for (j = 0; j < output_nodes; j++)
		{
			cout << "\t" << who[i][j];
		}
	}

}

void printinputs()
{
	
	int i;
	/*
	cout << "\nInput Nodes:\n";
	for (i = 0; i <= input_nodes; i++)
	{
		cout << inp[i] << "\t";
	}
	cout << "\nHidden Nodes:\n";
	for (i = 0; i <= hidden_layer_nodes; i++)
	{
		cout << hid[i] << "\t";
	}
	*/

	cout << "\n" << value;
	cout << "\nTarget Value:\n";
	for (i = 0; i < output_nodes; i++)
	{
		cout << tv[i] << "\t";
	}
	
	cout << "\nOutput Value:\n";
	for (i = 0; i < output_nodes; i++)
	{
		cout << op[i] << "\t";
	}
}

void set_weights()
{
	//cout<< "Hello";
	int i = 0, j = 0, k = 0;
	while (!out.eof())
	{
		getline(out, value);
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
		/*
		for (i = 0; i < input_nodes; i++)
		{
		//for (j = 0; j < hidden_layer_nodes; j++)
		//{
		while (token != NULL)
		{
		wih[i][j] = atof(token);
		j++;
		token = strtok(NULL, ",");
		}
		//}
		}
		*/
	}
	out.clear();
}

void test_network()
{
	//printinputs();
	int i = 0;
	tst = 1;
	/*
	for (i = 0; i < output_nodes; i++)
	{
	*/
		if ((op[1] == tv[1]) && (tst == 1))
		{
			tst = 1;
		}
		else
		{
			tst = 0;
		}
	//}

	if (tst == 1)
	{
		crr++;
	}
	else
	{
		err++;
		//printinputs();
	}
}