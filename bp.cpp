#include "bp.h"
#include <fstream>

using namespace std;

BpNet::BpNet()
{
	srand((unsigned)time(NULL));        // 随机数种子    
	error = 100.f;                      // error初始值，极大值即可

	// 初始化输入层
	for (int i = 0; i < innode; i++)
	{
		inputLayer[i] = new inputNode();
		for (int j = 0; j < hidenode; j++)
		{
			inputLayer[i]->weight.push_back(get_11Random());//输入层权值初始化
			inputLayer[i]->wDeltaSum.push_back(0.f);//返回时较差初始化
		}
	}

	// 初始化隐藏层
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == hidelayer - 1)//判断隐含层是否为1层 1层的话赋值简单
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j] = new hiddenNode();//申请一个数组结构体，因为是左右的所以是二维数组
				hiddenLayer[i][j]->bias = get_11Random();//后期计算得出，所以可以赋值任意
				for (int k = 0; k < outnode; k++)
				{
					hiddenLayer[i][j]->weight.push_back(get_11Random());//后期计算得出，所以可以赋值任意
					hiddenLayer[i][j]->wDeltaSum.push_back(0.f);//如果一层就可以直接计算
				}
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j] = new hiddenNode();
				hiddenLayer[i][j]->bias = get_11Random();
				for (int k = 0; k < hidenode; k++) { hiddenLayer[i][j]->weight.push_back(get_11Random()); }
			}
		}
	}

	// 初始化输出层
	for (int i = 0; i < outnode; i++)
	{
		outputLayer[i] = new outputNode();
		outputLayer[i]->bias = get_11Random();
	}
}

void BpNet::forwardPropagationEpoc()
{
	// forward propagation on hidden layer
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == 0)
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < innode; k++)
				{
					sum += inputLayer[k]->value * inputLayer[k]->weight[j];
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = sigmoid(sum);
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < hidenode; k++)
				{
					sum += hiddenLayer[i - 1][k]->value * hiddenLayer[i - 1][k]->weight[j];
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = sigmoid(sum);
			}
		}
	}

	// forward propagation on output layer
	for (int i = 0; i < outnode; i++)
	{
		double sum = 0.f;
		for (int j = 0; j < hidenode; j++)
		{
			sum += hiddenLayer[hidelayer - 1][j]->value * hiddenLayer[hidelayer - 1][j]->weight[i];
		}
		sum += outputLayer[i]->bias;
		outputLayer[i]->value = sigmoid(sum);
	}
}

void BpNet::backPropagationEpoc()
{
	// backward propagation on output layer
	// -- compute delta
	for (int i = 0; i < outnode; i++)
	{
		double tmpe = fabs(outputLayer[i]->value - outputLayer[i]->rightout);
		error += tmpe * tmpe / 2;

		outputLayer[i]->delta
			= (outputLayer[i]->value - outputLayer[i]->rightout)*(1 - outputLayer[i]->value)*outputLayer[i]->value;
	}

	// backward propagation on hidden layer
	// -- compute delta
	for (int i = hidelayer - 1; i >= 0; i--)    // 反向计算
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < outnode; k++){ sum += outputLayer[k]->delta * hiddenLayer[i][j]->weight[k]; }
				hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value) * hiddenLayer[i][j]->value;
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < hidenode; k++){ sum += hiddenLayer[i + 1][k]->delta * hiddenLayer[i][j]->weight[k]; }
				hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value) * hiddenLayer[i][j]->value;
			}
		}
	}
	// backward propagation on input layer
	// -- update weight delta sum
	for (int i = 0; i < innode; i++)
	{
		for (int j = 0; j < hidenode; j++)
		{
			inputLayer[i]->wDeltaSum[j] += inputLayer[i]->value * hiddenLayer[0][j]->delta;
		}
	}

	// backward propagation on hidden layer
	// -- update weight delta sum & bias delta sum
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				for (int k = 0; k < outnode; k++)
				{
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * outputLayer[k]->delta;
				}
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				for (int k = 0; k < hidenode; k++)
				{
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * hiddenLayer[i + 1][k]->delta;
				}
			}
		}
	}

	// backward propagation on output layer
	// -- update bias delta sum
	for (int i = 0; i < outnode; i++) outputLayer[i]->bDeltaSum += outputLayer[i]->delta;
}

void BpNet::training(static vector<sample> sampleGroup, double threshold)
{
	int sampleNum = sampleGroup.size();//定义样本数量等于容器的大小

	std::ofstream ofs("E:/log-error.txt");  // 定义输出误差文件

	int numCount = 1;

	while (error > threshold) // 当误差大于阈值时报错
	{
		// cout << "training error: " << error << endl; // 输出误差
		ofs << error << std::endl;

		error = 0.f;
		// initialize delta sum 将区间（数据范围,数据类型)元素赋值到当前的vector容器中，更新wDeltaSum值
		for (int i = 0; i < innode; i++)
		{
			inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(), 0.f);//范围和数据类型
		}
		for (int i = 0; i < hidelayer; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->wDeltaSum.assign(hiddenLayer[i][j]->wDeltaSum.size(), 0.f);
				hiddenLayer[i][j]->bDeltaSum = 0.f;
			}
		}
		for (int i = 0; i < outnode; i++) outputLayer[i]->bDeltaSum = 0.f;

		for (int iter = 0; iter < sampleNum; iter++)
		{
			setInput(sampleGroup[iter].in); //调用函数 把sampleGroup[iter].in赋值value方便处理
			setOutput(sampleGroup[iter].out);//调用函数 把sampleGroup[iter].out赋值value方便处理

			forwardPropagationEpoc();//调用函数
			backPropagationEpoc();//调用函数
		}

		// 向后传播输入层更新权重
		for (int i = 0; i < innode; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				inputLayer[i]->weight[j] -= learningRate * inputLayer[i]->wDeltaSum[j] / sampleNum;
			}
		}

		// 向后传播的隐含层更新权重和隐含层 update weight & bias
		for (int i = 0; i < hidelayer; i++)
		{
			if (i == hidelayer - 1)
			{
				for (int j = 0; j < hidenode; j++)
				{
					// bias
					hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;

					// weight
					for (int k = 0; k < outnode; k++)
					{
						hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
					}
				}
			}
			else
			{
				for (int j = 0; j < hidenode; j++)
				{
					// bias
					hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;

					// weight
					for (int k = 0; k < hidenode; k++)
					{
						hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
					}
				}
			}
		}

		// 向后传播输出层 更新 bias-- update bias
		for (int i = 0; i < outnode; i++)
		{
			outputLayer[i]->bias -= learningRate * outputLayer[i]->bDeltaSum / sampleNum;
		}
	}

	ofs.close();
}
//该函数内部调用了forwardPropagationEpoc()backPropagationEpoc()setInput setOutput

void BpNet::predict(vector<sample>& testGroup)
{
	int testNum = testGroup.size();

	for (int iter = 0; iter < testNum; iter++)
	{
		testGroup[iter].out.clear();
		setInput(testGroup[iter].in);

		//向前传播输入层到隐含层计算
		for (int i = 0; i < hidelayer; i++)
		{
			if (i == 0)
			{
				for (int j = 0; j < hidenode; j++)
				{
					double sum = 0.f;
					for (int k = 0; k < innode; k++)
					{
						sum += inputLayer[k]->value * inputLayer[k]->weight[j];
					}
					sum += hiddenLayer[i][j]->bias;
					hiddenLayer[i][j]->value = sigmoid(sum);
				}
			}
			else
			{
				for (int j = 0; j < hidenode; j++)
				{
					double sum = 0.f;
					for (int k = 0; k < hidenode; k++)
					{
						sum += hiddenLayer[i - 1][k]->value * hiddenLayer[i - 1][k]->weight[j];
					}
					sum += hiddenLayer[i][j]->bias;
					hiddenLayer[i][j]->value = sigmoid(sum);
				}
			}
		}

		// 向前传播隐含层到输出层计算
		for (int i = 0; i < outnode; i++)
		{
			double sum = 0.f;
			for (int j = 0; j < hidenode; j++)
			{
				sum += hiddenLayer[hidelayer - 1][j]->value * hiddenLayer[hidelayer - 1][j]->weight[i];
			}
			sum += outputLayer[i]->bias;
			outputLayer[i]->value = sigmoid(sum);
			testGroup[iter].out.push_back(outputLayer[i]->value);
		}
	}
}
//把容器里的输入值，一个一个赋值
void BpNet::setInput(static vector<double> sampleIn)
{
	for (int i = 0; i < innode; i++) inputLayer[i]->value = sampleIn[i];
}

void BpNet::setOutput(static vector<double> sampleOut)
{
	for (int i = 0; i < outnode; i++) outputLayer[i]->rightout = sampleOut[i];
}

