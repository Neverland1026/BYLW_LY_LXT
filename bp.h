#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define innode 7 // ÊäÈë½ÚµãÊý
#define hidenode 5 //Òþº¬½ÚµãÊý
#define hidelayer 1 // Òþº¬²ãÊý
#define outnode 5 //Êä³ö½ÚµãÊý
#define learningRate 0.1 // Ñ§Ï°ËÙÂÊ

// --- -1~1 Ëæ»úÊý²úÉúÆ÷ --- 
inline double get_11Random()    // -1 ~ 1
{
	return ((2.0*(double)rand() / RAND_MAX) - 1);
}


/**************sigmoidº¯Êý*****************/

inline double sigmoid(double x)
{
	double ans = 1 / (1 + exp(-x));
	return ans;
}

//---------ÊäÈë²ã½Úµã¡£°üº¬ÒÔÏÂ·ÖÁ¿£º---
//1£¬value£º ¹Ì¶¨ÊäÈëÖµ£»
//2£¬weight£ºÃæ¶ÔµÚÒ»²ãÒþº¬²ãÃ¿¸ö½Úµã¶¼ÓÐÈ¨Öµ
//3£¬wDeltaSum£º·µ»ØÊ±µÚÒ»²ãÒþº¬²ãÃ¿¸ö½ÚµãÈ¨ÖµµÄdeltaÖµµÄÀÛ»ý

typedef struct inputNode
{
	double value;
	vector<double>weight, wDeltaSum;

}inputNode;

//---------Êä³ö²ã½Úµã¡£°üº¬ÒÔÏÂÊýÖµ£º---
//1£¬value: ½Úµãµ±Ç°Öµ£»
//2,delta : ÓëÕýÈ·Êä³öÖµÖ®¼äµÄdeltaÖµ
//3,rightout ÕýÈ·Êä³öÖµ
//4£¬bias Æ«ÒÆÁ¿
//5£¬bDelaSum: biasµÄdeltaÖµµÄÀÛ»ý£¬Ã¿¸ö½ÚµãÒ»¸ö

typedef struct outputNode
{
	double value, delta, rightout, bias, bDeltaSum;

}outputNode;

// --- Òþº¬²ã½Úµã¡£°üº¬ÒÔÏÂÊýÖµ£º--- 
// 1.value:     ½Úµãµ±Ç°Öµ£» 
// 2.delta:     BPÍÆµ¼³öµÄdeltaÖµ£»
// 3.bias:      Æ«ÒÆÁ¿
// 4.bDeltaSum: biasµÄdeltaÖµµÄÀÛ»ý£¬Ã¿¸ö½ÚµãÒ»¸ö
// 5.weight:    Ãæ¶ÔÏÂÒ»²ã£¨Òþº¬²ã/Êä³ö²ã£©Ã¿¸ö½Úµã¶¼ÓÐÈ¨Öµ£» 
// 6.wDeltaSum£º weightµÄdeltaÖµµÄÀÛ»ý£¬Ãæ¶ÔÏÂÒ»²ã£¨Òþº¬²ã/Êä³ö²ã£©Ã¿¸ö½Úµã¸÷×Ô»ýÀÛ

typedef struct hiddenNode   // Òþº¬²ã½Úµã
{
	double value, delta, bias, bDeltaSum;
	vector<double> weight, wDeltaSum;
}hiddenNode;

// --- µ¥¸öÑù±¾´æ´¢·½Ê½ --- 
typedef struct sample
{
	vector<double> in, out;
}sample;

// --- BPÉñ¾­ÍøÂç --- 

class BpNet
{
public:
	BpNet();    //¹¹Ôìº¯Êý
	void forwardPropagationEpoc();  // µ¥¸öÑù±¾Ç°Ïò´«²¥
	void backPropagationEpoc();     // µ¥¸öÑù±¾ºóÏò´«²¥

	void training(static vector<sample> sampleGroup, double threshold);// ¸üÐÂ weight, bias
	void predict(vector<sample>& testGroup);                          // Éñ¾­ÍøÂçÔ¤²â

	void setInput(static vector<double> sampleIn);     // ÉèÖÃÑ§Ï°Ñù±¾ÊäÈë
	void setOutput(static vector<double> sampleOut);    // ÉèÖÃÑ§Ï°Ñù±¾Êä³ö

public:
	double error;
	//´´½¨Ò»¸öÊý×éÖ¸Õë£¬¸ÃÊý×éµÄÖ¸ÏòÒ»¸ö½á¹¹Ìå±äÁ¿£¬ÕâÑù×öµÄºÃ´¦¾ÍÊÇÊ¹Êý¾Ýµ÷ÓÃÇå³þ
	inputNode* inputLayer[innode];                      // ÊäÈë²ã£¨½öÒ»²ã£©
	outputNode* outputLayer[outnode];                   // Êä³ö²ã£¨½öÒ»²ã£©
	hiddenNode* hiddenLayer[hidelayer][hidenode];       // Òþº¬²ã£¨¿ÉÄÜÓÐ¶à²ã£©
};
