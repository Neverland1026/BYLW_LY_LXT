// namespace std
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ctime>
#include <functional>
#include <locale>
#include "bp.h"

// boost
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <utility>

// namespace pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/segmentation/sac_segmentation.h>

// ���� IO ����
namespace IO{

	/*
	 * \function: �� txt �ļ�תΪ std::vector
	 * \brief   :
	 * \param   : path txt�ļ���·��
	 * \param   : vec �������
	 * \output  : �ж�ֵ
	 * \author  : Neverland_LY
	 */
	int loadTxt(std::string path, std::vector<std::vector<std::string>> & vec){
		std::ifstream ifs(path.c_str(), std::ios::in);
		if (!ifs) { std::cout << "Can not open the file!\n"; return -1; }
		std::string lineStr;
		while (getline(ifs, lineStr)) {
			std::stringstream ss(lineStr);
			std::string str;
			std::vector<std::string> lineArray;
			while (getline(ss, str, ' '))
				lineArray.push_back(str);
			vec.push_back(lineArray);
		}
		vec.shrink_to_fit();
		return 1;
	}

	/*
	* \brief ����һ���ļ����µ����е����ļ�
	* \param path �ļ�·��(����·���������·��)
	* \param file_abs_path ·��
	* \param file_name �ļ���
	*/
	void getFileAbsolutePath(std::string root_path,
		std::vector<std::string> &file_abs_path,
		std::vector<std::string> &file_name) {

		boost::filesystem::recursive_directory_iterator itor(root_path.c_str());
		boost::filesystem::recursive_directory_iterator itEnd;
		for (; itor != itEnd; ++itor) {
			boost::filesystem::path file_path = itor->path();

			if (boost::filesystem::is_regular_file(file_path)) {
				file_abs_path.push_back(file_path.string());
				file_name.push_back(file_path.filename().string());
			}

			if (boost::filesystem::is_directory(file_path)) {
				getFileAbsolutePath(file_path.string(), file_abs_path, file_name);
			}
		}
	}

}

// ������Ĵ���ϼ�
std::vector<std::vector<std::string>> VEC;  // ȫ�ֱ���
namespace LXT{

	/*
	* \function: sortCloud
	* \brief   : �Ե��ƽ�����������
	* \param   : vecA Ҫ�����ĵ���
	* \param   : vecB ��׼˳�����
	* \output  : �жϱ�־λ
	* \author  : Neverland_LY
	*/
	int sortCloud(const std::vector<std::vector<std::string>> & vecA,
		const std::vector<std::vector<std::string>> & vecB){

		std::cout << " > ��ʼ���е������� ...\n";

		// ������ A תΪ pcl::cloudXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubA : vecA){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubA[0]);
			point.y = std::stod(vecSubA[1]);
			point.z = std::stod(vecSubA[2]);
			cloudA->points.push_back(point);
		}

		// ������ B תΪ pcl::cloudXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubB : vecB){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubB[0]);
			point.y = std::stod(vecSubB[1]);
			point.z = std::stod(vecSubB[2]);
			cloudB->points.push_back(point);
		}

		// ����kdtree
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud(cloudA); // ע��Ҫ��A��������������ӵ���A����
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);//����1���������
		std::vector<float> pointNKNSquaredDistance(K);

		// ��ʼ����
		if (cloudA->points.size() != cloudB->points.size()){
			std::cout << "���Ƹ�������ȣ���ȷ�ϣ�\n";
			return -1;
		}

		std::string path("E:/sorted.txt");
		std::ofstream ofs(path);
		for (int i = 0; i < cloudB->points.size(); ++i){
			pcl::PointXYZ p;
			p.x = std::stod(vecB[i][0]);
			p.y = std::stod(vecB[i][1]);
			p.z = std::stod(vecB[i][2]);
			if (kdtree.nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){
				// �ҵ����ˣ��������
				int index = pointIdxNKNSearch[0];
				for (int k = 0; k < vecA[0].size(); ++k){
					ofs << vecA[index][k] << " ";
				}
				ofs << '\n';
			}
		}
		ofs.close();

		std::cout << " > ����������� ...		" << path << std::endl;

		return 1;
	}

	/*
	* \function: isInRange
	* \brief   : �ж�һ��ֵ�Ƿ�����һ������
	* \param   : num Ҫ�жϵ���
	* \param   : range ���伯��
	* \output  : �жϱ�־λ
	* \author  : Neverland_LY
	*/
	int isInRange(float num, const std::vector<std::pair<float, float>> & range){
		for (auto pa : range){
			if (num >= pa.first && num <= pa.second)
				return 1; // �ҵ���
		}
		return -1; // �Ҳ���
	}

	/*
	* \function: generateBinarySignal
	* \brief   : ���ݹ������ɶ������ź�
	* \param   : vec �������ݣ���������Ϊ XYZIRoughReturnsRGBH
	* \output  : �жϱ�־λ
	* \author  : Neverland_LY
	*/
	int generateBinarySignal(const std::vector<std::vector<std::string>> & vec){
		// ����
		// 012  3    4      5      6  7  8  9
		// XYZ  I  Rough  Returns  R  G  B  H
		std::cout << " > ��ʼ���ж������ź����� ...\n";
		std::string path("E:/BinarySignal.txt");
		std::ofstream ofs_bin(path);

		// �˹����� 11
		std::vector<std::pair<float, float>> vecBuildings;
		std::ifstream ifs_buildings("E:/range_buildings_11.txt");
		{
			std::string lineStr;
			while (getline(ifs_buildings, lineStr)) {
				float num = std::stod(lineStr.substr(0, 5));
				vecBuildings.push_back(std::pair<float, float>(num - 0.5f, num + 0.5f));
			}
		}
		ifs_buildings.close();

		// ��· 01
		std::vector<std::pair<float, float>> vecRoads;
		std::ifstream ifs_roads("E:/range_road_01.txt");
		{
			std::string lineStr;
			while (getline(ifs_roads, lineStr)) {
				float num = std::stod(lineStr.substr(0, 5));
				vecRoads.push_back(std::pair<float, float>(num - 0.5f, num + 0.5f));
			}
		}
		ifs_roads.close();

		// ��� 10
		std::vector<std::pair<float, float>> vecEarths;
		std::ifstream ifs_earths("E:/range_earth_10.txt");
		{
			std::string lineStr;
			while (getline(ifs_earths, lineStr)) {
				float num = std::stod(lineStr.substr(0, 5));
				vecEarths.push_back(std::pair<float, float>(num - 0.5f, num + 0.5f));
			}
		}
		ifs_earths.close();

		// ����
		// 012  3    4      5      6  7  8  9
		// XYZ  I  Rough  Returns  R  G  B  H
		for (auto vecSub : vec){
			// nan ���ó� 0.00f
			for (auto & value : vecSub){
				if (value == "nan") value = "0.00";
			}

			std::string str("");
			// �����ж� I
			// str += " ";
			float intensity = std::stod(vecSub[3]);
			if (isInRange(intensity, vecBuildings) > 0) str += "1 1";
			else if (isInRange(intensity, vecRoads) > 0) str += "0 1";
			else if (isInRange(intensity, vecEarths) > 0) str += "1 0";
			else str += "0 0";

			// �ж�Rough
			str += " ";
			float rough = std::stod(vecSub[4]);
			if (rough >= 0.4f) str += "1 1";
			else if (rough >= 0.15f) str += "1 0";
			else if (rough <= 0.045f & rough >= 0.015f) str += "0 1";
			else str += "0 0";

			// �жϻز�����
			str += " ";
			int ret = (int)(std::stod(vecSub[5]));
			if (ret == 4) str += "1 1";
			else if (ret == 3) str += "1 0";
			else if (ret == 2) str += "0 1";
			else str += "0 0";

			// �ж���ɫ
			str += " ";
			int r = std::stod(vecSub[6]); int g = std::stod(vecSub[7]); int b = std::stod(vecSub[8]);
			if (r == 255 && g == 0) str += "1 1";
			else if (r == 46) str += "0 1";
			else if (r == 255 && g == 255) str += "1 0";
			else str += "0 0";

			// �жϸ߶�(abs)
			str += " ";
			float deltaH = std::fabs(std::stod(vecSub[9]));
			if (deltaH >= 9.0f) str += "1 1";
			else if (deltaH >= 4.0f) str += "1 0";
			else if (deltaH >= 0.1f && deltaH <= 3.0f) str += "0 1";
			else str += "0 0";

			// ������
			ofs_bin << vecSub[0] << " " << vecSub[1] << " " << vecSub[2] << " " << str << '\n';
		}
		ofs_bin.close();

		std::cout << " > �������ź�������� ...	" << path << std::endl;

		return 1;
	}

	int AI(std::vector<std::vector<double>> const & pTrainData,
		std::vector<std::vector<double>> const & pTrainTag,
		std::vector<std::vector<double>> const & pTestData,
		std::vector<std::vector<double>> & pTestTag){

		// ������������
		int numTrain = pTrainData.size();
		std::vector<sample> sampleInput(numTrain);
		for (int i = 0; i < numTrain; i++)
		{
			sampleInput[i].in = pTrainData[i];
			sampleInput[i].out = pTrainTag[i];
		}

		// ѵ��ģ��
		std::vector<sample> sampleGroupInput(std::begin(sampleInput), std::end(sampleInput));
		BpNet testNet;
		testNet.training(sampleGroupInput, 0.001);

		// �����������
		int numTest = pTestData.size();
		std::vector<sample> sampleOutput(numTest);
		for (int i = 0; i < numTest; i++){
			sampleOutput[i].in = pTestData[i];
			// sampleOutput[i].out = pTestTag[i]; // ��ʱΪ��
		}

		// ����ģ��
		std::vector<sample> sampleGroupOutput(std::begin(sampleOutput), std::end(sampleOutput));
		testNet.predict(sampleGroupOutput);

		for (int i = 0; i < sampleGroupOutput.size(); i++)
			pTestTag[i] = std::vector<double>(std::begin(sampleGroupOutput[i].out),
			std::end(sampleGroupOutput[i].out));

		return 1;
	}

	int readData(std::vector<std::vector<double>> & pVec, std::string path){
		std::ifstream ifs(path.c_str());
		if (!ifs){
			std::cout << "Can not find file!\n";
			return -1;
		}

		std::string lineStr;
		int indexVec = 0;
		while (getline(ifs, lineStr)) {
			std::stringstream ss(lineStr); // getline �÷�1
			std::string str;
			std::vector<std::string> lineArray;
			while (getline(ss, str, ' ')) // getline �÷�2
				lineArray.push_back(str);

			for (auto v : lineArray)
				pVec[indexVec].push_back(std::stod(v));

			indexVec++;
		}

		ifs.close();
		return 1;

	}

	int readData2(std::vector<std::vector<double>> & pVec, std::string path){
		std::ifstream ifs(path.c_str());
		if (!ifs){
			std::cout << "Can not find file!\n";
			return -1;
		}

		std::string lineStr;
		int indexVec = 0;
		while (getline(ifs, lineStr)) {
			std::stringstream ss(lineStr); // getline �÷�1
			std::string str;
			std::vector<std::string> lineArray;
			while (getline(ss, str, ' ')) // getline �÷�2
				lineArray.push_back(str);

			// ����ԭʼ���ݾ���
			std::vector<std::string> vv;
			vv.push_back(lineArray[0]);
			vv.push_back(lineArray[1]);
			vv.push_back(lineArray[2]);
			VEC.push_back(vv);

			for (int tt = 3; tt <= 12; ++tt){
				pVec[indexVec].push_back(std::stod(lineArray[tt]));
			}
			pVec[indexVec].shrink_to_fit();

			indexVec++;
			/*
						if (indexVec % 100000 == 0)
						std::cout << indexVec << '\n';*/
		}

		VEC.shrink_to_fit();

		ifs.close();
		return 1;

	}

	int BPTrain(){

		std::cout << " > ��ʼ��ȡѵ������ ...\n";

		// ���� train_num | test_num
		int train_num = 5; // ѵ�����ݸ��� ***
		int test_num = 1394229;  // �������ݸ��� ***

		// ���� train_data | train_tag | test_data | test_tag
		std::vector<std::vector<double>> train_data(train_num);
		std::vector<std::vector<double>> train_tag(train_num);
		std::vector<std::vector<double>> test_data(test_num);
		std::vector<std::vector<double>> test_tag(test_num);

		// ��������
		if (readData(train_data, "E:/train_data.txt") < 0 ||
			readData(train_tag, "E:/train_tag.txt") < 0 ||
			readData2(test_data, "E:/BinarySignal.txt") < 0){
			return -1;
		}

		std::cout << " > ��ʼѵ��ģ�� ...\n";

		// ѵ��
		if (AI(train_data, train_tag, test_data, test_tag) < 0){
			std::cout << " > ģ��ѵ������δ֪����\n";
			return -1;
		}

		// ���Ԥ����
		std::cout << " > ��ʼ���ɷ����� ...\n";
		std::ofstream ofs("E:/test_tag.txt");
		for (int i = 0; i < test_tag.size(); i++)
		{
			std::string str = "";
			str += VEC[i][0] + " ";
			str += VEC[i][1] + " ";
			str += VEC[i][2] + " ";
			for (int j = 0; j < test_tag[i].size(); j++){ // �����������
				str += std::to_string((int)(std::round(test_tag[i][j]))) + " ";
			}
			ofs << str.substr(0, str.length() - 1) << '\n';
		}
		ofs.close();

		std::cout << " > ѵ��ģ����� ...		" << "E:/test_tag.txt\n";

		return 1;
	}

	// �����ݼӱ�ǩ
	int addLabel6(const std::vector<std::vector<std::string>> & vec){

		std::cout << " > ��ʼ�����ݼӱ�ǩ( 6 �� 0 & 1) ...\n";
		// �ж�
		std::string path("E:/classfication-6.txt");
		std::ofstream ofs(path);
		for (auto vv : vec) {
			std::string str = vv[3] + vv[4] + vv[5] + vv[6] + vv[7] + vv[8];
			// ��־λ
			int index = 0;
			if (str == "011011") index = 1; //��
			else if (str == "100111") index = 2; // ����
			else if (str == "011001") index = 3; // ֲ��
			else if (str == "110100") index = 4; // ��·
			else if (str == "101100") index = 5; // ���
			else if (str == "000000") index = 6; // ˮ

			// ���������
			ofs << vv[0] << " " << vv[1] << " " << vv[2] << " " << std::to_string(index) << '\n';
		}
		ofs.close();

		std::cout << " > ���ݼӱ�ǩ��� ...		" << path << std::endl;

		return 1;
	}

	// �����ݼӱ�ǩ
	int addLabel10(const std::vector<std::vector<std::string>> & vec){

		std::cout << " > ��ʼ�����ݼӱ�ǩ( 10 �� 0 & 1) ...\n";
		// �ж�
		std::string path("E:/classfication-10.txt");
		std::ofstream ofs(path);
		for (auto vv : vec) {
			std::string str = vv[3] + vv[4] + vv[5] + vv[6] + vv[7] + vv[8] + vv[9] + vv[10] + vv[11] + vv[12];
			// ��־λ
			int index = 0;
			if (str == "0011110011") index = 1; //��
			else if (str == "1100101101") index = 2; // ����
			else if (str == "0001010010") index = 3; // ֲ��
			else if (str == "0110000100") index = 4; // ��·
			else if (str == "1000001000") index = 5; // ���
			else if (str == "0000000000") index = 6; // ˮ

			// ���������
			ofs << vv[0] << " " << vv[1] << " " << vv[2] << " " << std::to_string(index) << '\n';
		}
		ofs.close();

		std::cout << " > ���ݼӱ�ǩ��� ...		" << path << std::endl;

		return 1;
	}

	// �ж���ͬ���ֵ�λ���м�λ
	int getSameNumByBit(const std::string str, const std::string std_str){
		int num = 0;
		for (int i = 0; i < str.size(); ++i){
			if (str[i] == std_str[i]) num++;
		}
		return num;
	}

	// ���Ȩ��
	int getWeight(std::string str){
		// С��ָ���ı�׼�źţ�train-tag��
		std::vector<std::string> vecStr{ "11011", "11110", "10010", "00101", "01101" };
		std::vector<int> idx(vecStr.size());
		std::iota(idx.begin(), idx.end(), 0);
		std::vector<int> vecValue(vecStr.size());
		for (int i = 0; i<vecStr.size(); ++i){
			if (str == vecStr[i]) return i;
			vecValue[i] = getSameNumByBit(str, vecStr[i]);
		}
		std::sort(idx.begin(), idx.end(), [&vecValue](int i, int j) {return vecValue[i] > vecValue[j]; });
		if (vecValue[idx[0]] == vecValue[idx[1]]) return 0;
		return idx[0];
	}

	// ��5�� 0 �� 1 ��涨�����ֽ��бȽϣ�������ͬ�ĸ�����ΪȨ��ֵ
	int generateSignalByWeight(const std::vector<std::vector<std::string>> & vec){
		std::cout << " > ���������� ...\n";
		std::string path("E:/SignalByBit.txt");
		std::ofstream ofs(path);
		for (auto v : vec){
			std::string str = v[3] + v[4] + v[5] + v[6] + v[7];
			ofs << v[0] << " " << v[1] << " " << v[2] << " " << std::to_string(getWeight(str) + 1) << std::endl;
		}
		ofs.close();
		std::cout << " > ������� ...		" << path << std::endl;

		return 1;
	}
}

// Neverland_LY �Ĵ���ϼ�
namespace LY{

	/*
	 * \function: setZEqual2Zero
	 * \brief   : �����Ƶ� Z ��Ϊ 0����Ҫ���ڽ�����ֿ�ͶӰ��
	 * \param   : vecCloud �������
	 * \param   : path �������·��
	 * \output  : ״̬��־λ
	 * \author  : Neverland_LY
	 */
	int setZEqual2Zero(const std::vector<std::vector<std::string>> & vecCloud){

		std::cout << " > ��ʼ���е���ͶӰ ...\n";

		std::string path("E:/zero.txt");
		std::ofstream ofs(path);
		for (auto vecSub : vecCloud){
			for (int i = 0; i < vecSub.size(); ++i){
				if (i == 2){ // ��ʱ���� Z �ֶ�
					ofs << "0.00 ";
					continue;
				}
				ofs << vecSub[i] << " ";
			}
			ofs << "\n";
		}
		ofs.close();
		std::cout << " > ����ͶӰ��� ...		" << path << std::endl;
		return 1;
	}

	/*
	 * \function: splitCloud9Blocks
	 * \brief   : ��Ҫ���ڽ����ƽ��оŹ��񻮷֣��ֲ�ʾ�⣩
	 * \param   :
	 * \output  :
	 * \author  : Neverland_LY
	 */
	int splitCloud9Blocks(const std::vector<std::vector<std::string>> & vecCloud, float deltaMeshWidth = 0.0f){

		std::cout << " > ��ʼ���е��Ʒֿ� ...\n";

		// �ȼ��㳤��
		float maxX = -1 * INFINITY;
		float maxY = -1 * INFINITY;
		float maxZ = -1 * INFINITY;
		float minX = INFINITY;
		float minY = INFINITY;
		float minZ = INFINITY;
		for (auto vecSub : vecCloud){
			for (int i = 0; i < vecSub.size(); ++i){
				// ���� X
				if (std::stod(vecSub[0]) > maxX) maxX = std::stod(vecSub[0]);
				else if (std::stod(vecSub[0]) < minX) minX = std::stod(vecSub[0]);

				// ���� Y
				if (std::stod(vecSub[1]) > maxY) maxY = std::stod(vecSub[1]);
				else if (std::stod(vecSub[1]) < minY) minY = std::stod(vecSub[1]);

				// ���� Z
				if (std::stod(vecSub[2]) > maxZ) maxZ = std::stod(vecSub[2]);
				else if (std::stod(vecSub[2]) < minZ) minZ = std::stod(vecSub[2]);
			}
		}

		// ȡ����Ϊ����������С
		float meshWidth = ((maxX - minX) > (maxY - minY)) ? (maxX - minX) : (maxY - minY);

		// �������� X �ڵ�
		float x1 = minX + meshWidth / 3;
		float x2 = minX + meshWidth / 3 * 2;

		// �������� Y �ڵ�
		float y1 = minY + meshWidth / 3;
		float y2 = minY + meshWidth / 3 * 2;

		// ����Ź����϶���
		// float deltaMeshWidth = 30.0f;

		// �ж������Ǹ�������
		// 1  2  3
		// 4  5  6
		// 7  8  9
		// ��ȡ X �Ӵ�С��Y �Ӵ�С
		float x; float y; float z;
		std::ofstream ofs1("E:/Block-1.txt"); std::ofstream ofs2("E:/Block-2.txt");
		std::ofstream ofs3("E:/Block-3.txt"); std::ofstream ofs4("E:/Block-4.txt");
		std::ofstream ofs5("E:/Block-5.txt"); std::ofstream ofs6("E:/Block-6.txt");
		std::ofstream ofs7("E:/Block-7.txt"); std::ofstream ofs8("E:/Block-8.txt");
		std::ofstream ofs9("E:/Block-9.txt");
		for (auto vecSub : vecCloud){
			x = std::stod(vecSub[0]);
			y = std::stod(vecSub[1]);
			z = std::stod(vecSub[2]);

			if (x > x2){
				if (y > y2)
					ofs3 << std::to_string(x + 2 * deltaMeshWidth) << " "
					<< std::to_string(y + 2 * deltaMeshWidth) << " " << z << "\n";
				else if (y > y1)
					ofs6 << std::to_string(x + 2 * deltaMeshWidth) << " "
					<< std::to_string(y + 1 * deltaMeshWidth) << " " << z << "\n";
				else
					ofs9 << std::to_string(x + 2 * deltaMeshWidth) << " "
					<< std::to_string(y + 0 * deltaMeshWidth) << " " << z << "\n";
			}
			else if (x > x1){
				if (y > y2)
					ofs2 << std::to_string(x + 1 * deltaMeshWidth) << " "
					<< std::to_string(y + 2 * deltaMeshWidth) << " "
					<< z << "\n";
				else if (y > y1)
					ofs5 << std::to_string(x + 1 * deltaMeshWidth) << " "
					<< std::to_string(y + 1 * deltaMeshWidth) << " "
					<< z << "\n";
				else
					ofs8 << std::to_string(x + 1 * deltaMeshWidth) << " "
					<< std::to_string(y + 0 * deltaMeshWidth) << " " << z << "\n";
			}
			else{
				if (y > y2)
					ofs1 << std::to_string(x + 0 * deltaMeshWidth) << " "
					<< std::to_string(y + 2 * deltaMeshWidth) << " " << z << "\n";
				else if (y > y1)
					ofs4 << std::to_string(x + 0 * deltaMeshWidth) << " "
					<< std::to_string(y + 1 * deltaMeshWidth) << " " << z << "\n";
				else
					ofs7 << std::to_string(x + 0 * deltaMeshWidth) << " "
					<< std::to_string(y + 0 * deltaMeshWidth) << " " << z << "\n";
			}
		}

		ofs1.close(); ofs2.close(); ofs3.close();
		ofs4.close(); ofs5.close(); ofs6.close();
		ofs7.close(); ofs8.close(); ofs9.close();

		std::cout << " > ���Ʒֿ���� ...		" << "E:/Block-X.txt\n";

		return 1;
	}

	/*
	* \brief ����ռ�����ľ���
	* \param  p1 ��1
	* \param  p2 ��2
	* \return �����Ŀռ����
	*/
	inline float calcDistanceOf2Points(const Eigen::Vector3f & p1, const Eigen::Vector3f & p2) {
		Eigen::Vector3f diff = p1 - p2;
		return std::sqrt(diff.dot(diff));
	}

	struct Point {
		float x;
		float y;

		Point(float _x, float _y) {
			x = _x;
			y = _y;
		}
	};

	/**
	* ��С���˷�ֱ����ϣ����ǳ�����һԪ���Իع��㷨��
	* ����ɢ�����Ϊ  a x + b y + c = 0 ��ֱ��
	* ����ÿ����� X Y ��������Ƿ��� 0 ��ֵ����̬�ֲ��ġ�
	* ��һԪ���Իع��㷨������һԪ���Իع��㷨�ٶ� X �������ģ�ֻ�� Y ����
	*/
	bool lineFit(const std::vector<Point> &points, float &a, float &b, float &c) {

		// С�� 2 ����
		int size = points.size();
		if (size < 2) {
			a = 0;
			b = 0;
			c = 0;
			return false;
		}

		// ���� 3 ����
		float x_mean = 0.0f;
		float y_mean = 0.0f;
		for (int i = 0; i < size; i++) {
			x_mean += points[i].x;
			y_mean += points[i].y;
		}

		x_mean /= size; y_mean /= size; //���ˣ�������� x y �ľ�ֵ

		float Dxx = 0.0f, Dxy = 0.0f, Dyy = 0.0f;

		for (int i = 0; i < size; i++) {
			Dxx += (points[i].x - x_mean) * (points[i].x - x_mean);
			Dxy += (points[i].x - x_mean) * (points[i].y - y_mean);
			Dyy += (points[i].y - y_mean) * (points[i].y - y_mean);
		}

		float lambda = ((Dxx + Dyy) - std::sqrt(std::pow(Dxx - Dyy, 2) + 4 * std::pow(Dxy, 2))) / 2.0;
		float den = std::sqrt(Dxy * Dxy + std::pow(lambda - Dxx, 2));
		a = Dxy / den;
		b = (lambda - Dxx) / den;
		c = -a * x_mean - b * y_mean;
		return true;
	}

	/*
	 * \function: extractBoundary
	 * \brief   : ��ȡ��·�߽�
	 * \param   : vecCloud �������
	 * \param   : pathOutput ���·��
	 * \output  :
	 * \author  : Neverland_LY
	 */
	int extractBoundary(const std::vector<std::vector<std::string>> & vecCloud){

		std::cout << " > ��ʼ��ȡ���Ʊ߽� ...\n";

		// ����ת���� pcl::PointXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSub : vecCloud){
			pcl::PointXYZ point;
			point.x = std::stod(vecSub[0]);
			point.y = std::stod(vecSub[1]);
			point.z = std::stod(vecSub[2]);
			cloud->points.push_back(point);
		}

		// ����KdTree
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; // KdTree
		kdtree.setInputCloud(cloud);

		// ��׼���� 4r/3PI ���뾶ȡ 1.0f
		float radius;
		std::cout << " > �����������뾶��";
		std::cin >> radius;
		float stdDistance = 4 * radius / (3 * std::acos(-1));

		// ��ʼ�������е�
		std::string path("E:/border.txt");
		std::ofstream ofs(path);
		for (size_t i = 0; i < cloud->points.size(); ++i){

			// ���������
			std::vector<int> vecIdx;
			std::vector<float> vecDis;

			// ����㵽
			Eigen::Vector4f xyz_vectroid; // ��������
			if ((kdtree.radiusSearch(cloud->points[i], radius, vecIdx, vecDis)) > 0){
				pcl::compute3DCentroid(*cloud, vecIdx, xyz_vectroid);
			}


			// �������ĺ�Բ�ĵĿռ����
			Eigen::Vector3f p1 = Eigen::Vector3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
			Eigen::Vector3f p2 = Eigen::Vector3f(xyz_vectroid[0], xyz_vectroid[1], xyz_vectroid[2]);
			float distance = calcDistanceOf2Points(p1, p2);

			// �жϾ����Ƿ�������ֵ
			if ((distance > stdDistance * 0.7) && (distance < stdDistance * 1.3)){
				ofs << vecCloud[i][0] << " " << vecCloud[i][1] << " " << vecCloud[i][2] << " 255 255 0\n";
			}
		}
		ofs.close();

		std::cout << " > ���Ʊ߽���ȡ��� ...		" << path << std::endl;

		return 1;
	}

	/*
	 * \function: density
	 * \brief   : ���ݵ��ƽ����ܶȶ�ֵ��
	 * \param   :
	 * \output  :
	 * \author  : Neverland_LY
	 */
	int density(const std::vector<std::vector<std::string>> & vecCloud){

		std::cout << " > ��ʼ��������ܶ� ...\n";

		// ����ת���� pcl::PointXYZRGB
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSub : vecCloud){
			pcl::PointXYZ point;
			point.x = std::stod(vecSub[0]);
			point.y = std::stod(vecSub[1]);
			point.z = std::stod(vecSub[2]);
			cloud->points.push_back(point);
		}

		// ����KdTree
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; // KdTree
		kdtree.setInputCloud(cloud);

		// �������е�ĵ����ܶ�
		std::vector<int> vecIdx;
		std::vector<float> vecDis;
		float radius;
		std::cout << " > �����������뾶��";
		std::cin >> radius;
		std::vector<int> vecDensity(cloud->points.size());
		for (size_t i = 0; i < cloud->points.size(); ++i){
			// ��������һ���뾶��Χ�ڵĵ�����������ܶ�
			if ((kdtree.radiusSearch(cloud->points[i], radius, vecIdx, vecDis)) > 0){
				vecDensity[i] = vecIdx.size();
			}
		}

		// ���ܶȾ�ֵ
		int minDensity = *min_element(vecDensity.begin(), vecDensity.end());
		int maxDensity = *max_element(vecDensity.begin(), vecDensity.end());
		float sumDensity = std::accumulate(vecDensity.begin(), vecDensity.end(), 0.0f);
		sumDensity /= vecDensity.size();

		std::cout << " > �����ܶ���Сֵ�� " << minDensity << std::endl;
		std::cout << " > �����ܶ����ֵ�� " << maxDensity << std::endl;
		std::cout << " > �����ܶ�ƽ��ֵ�� " << maxDensity << std::endl;

		// �������
		std::string path = "E:/density.txt";
		std::ofstream ofs(path);
		int sizeFiled = vecCloud[0].size();
		for (int i = 0; i < vecCloud.size(); ++i){
			for (int j = 0; j < sizeFiled; ++j){
				ofs << vecCloud[i][j] << " ";
			}
			ofs << std::to_string(vecDensity[i]) << "\n";
		}
		ofs.close();
		std::cout << " > �����ܶȼ������ ...		" << path << std::endl;
		return 1;
	}

	/*
	 * \function: ������Ƶ����ԣ����Ƶ��ܶȣ�ÿƽ���ף������߲����㡢�ǵ����
	 * \brief   : �����������
	 * \param   : vecCloud ��������
	 * \output  : �������Զ�ȡ״̬
	 * \author  : Neverland_LY
	 */
	int computePointCloudProps(const std::vector<std::vector<std::string>> & vecCloud){

		std::cout << " > ��ʼ����������� ...\n";

		// ����ת���� pcl::PointXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSub : vecCloud){
			pcl::PointXYZ point;
			point.x = std::stod(vecSub[0]);
			point.y = std::stod(vecSub[1]);
			point.z = std::stod(vecSub[2]);
			cloud->points.push_back(point);
		}

		// ����KdTree
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; // KdTree
		kdtree.setInputCloud(cloud);

		// ��ʼ������
		float heightMax = -INFINITY; // ��ʼ���߳���ߵ�
		float heightMin = INFINITY; // ��ʼ���߳���͵�
		float density = 0.0f; // ��ʼ���ܶ�
		int NLoop = 0;
		int sumNeighbor = 0;
		float radius;
		std::cout << " > �����������뾶��";
		std::cin >> radius;
		for (auto point : cloud->points){

			// ����߲���
			heightMax = (heightMax > point.z) ? heightMax : point.z;
			heightMin = (heightMin < point.z) ? heightMin : point.z;

			// �����ܶ���
			std::vector<int> vecIdx;
			std::vector<float> vecDis;
			if ((kdtree.radiusSearch(point, radius, vecIdx, vecDis)) > 0){
				sumNeighbor += vecIdx.size();
			}
			NLoop++;
		}

		std::cout << "		�߳����ֵ�� " << heightMax << std::endl;
		std::cout << "		�߳���Сֵ�� " << heightMin << std::endl;
		std::cout << "		���̲߳ " << heightMax - heightMin << std::endl;
		std::cout << "		����ƽ���ܶȣ�" << (float)sumNeighbor / NLoop << std::endl;

		std::cout << " > �������Լ������ ...\n";

		return 1;
	}

	/*
	* \function: �ӵ���A���޳�����B�����ڴ�ԭʼ������ȥ������B
	* \brief   : ע����A-B
	* \param   : cloudA ����A
	* \param   : cloudA ����B
	* \output  : �������Զ�ȡ״̬
	* \author  : Neverland_LY
	*/
	int deleteBFromAXYZ(const std::vector<std::vector<std::string>> & vecA, const std::vector<std::vector<std::string>> & vecB){

		std::cout << " > ��ʼ�������� ...\n";

		// vecA ת���� pcl::PointXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubA : vecA){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubA[0]);
			point.y = std::stod(vecSubA[1]);
			point.z = std::stod(vecSubA[2]);
			cloudA->points.push_back(point);
		}

		// vecB ת���� pcl::PointXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubB : vecB){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubB[0]);
			point.y = std::stod(vecSubB[1]);
			point.z = std::stod(vecSubB[2]);
			cloudB->points.push_back(point);
		}

		// �� B ������ӵ� KdTree ��
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud(cloudB);
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);//����1���������
		std::vector<float> pointNKNSquaredDistance(K);

		//���ò���
		int sizeA = (int)cloudA->points.size(); int sizeB = (int)cloudB->points.size();

		// ���λ������0��ʾ���غϵ㣬1��ʾ���غϵ�
		std::vector<int> flagA(sizeA, 0); std::vector<int> flagB(sizeB, 0);

		// ���A������B�ĵ�
		std::vector<int> LA;
		LA.clear();  //���A������B�ĵ㣬��ʼΪ��, �����峤�ȣ���˺���Ҫ��push_back��ѹ������

		// ��ʼ���ñ��λ
		for (int i = 0; i < sizeA; ++i){
			if (kdtree.nearestKSearch(cloudA->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){
				if (pointNKNSquaredDistance[0] < 0.001)
					flagA[i] = 1; flagB[pointIdxNKNSearch[0]] = 1;
			}
		}

		// ��¼���
		for (int i = 0; i < sizeA; ++i){
			if (flagA[i] == 0)
				LA.push_back(i);
		}

		// �洢����  A - B = C
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudC(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*cloudA, LA, *cloudC);//�����������Ƶ���

		// �������
		std::cout << " > ���ڱ������ ...\n";
		std::string path("E:/AminusB.txt");
		std::ofstream ofs(path);
		for (auto p : cloudC->points){
			ofs << std::to_string(p.x) << " " << std::to_string(p.y) << " " << std::to_string(p.z) << "\n";
		}
		ofs.close();

		std::cout << " > ����������� ...		" << path << std::endl;

		return 1;
	}

	/*
	* \function: �Ե���AΪ��׼���Ե���B������׼
	* \brief   : ע����A-B
	* \param   : cloudA ����A  XYZRGB
	* \param   : cloudA ����B  XYZ
	* \output  : �������Զ�ȡ״̬
	* \author  : Neverland_LY
	*/
	int resampleColor(const std::vector<std::vector<std::string>> & vecA, const std::vector<std::vector<std::string>> & vecB){

		std::cout << " > ��ʼΪ���ƽ�����ɫ ...\n";

		// ������ A תΪ pcl::cloudXYZRGB
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (auto vecSubA : vecA){
			pcl::PointXYZRGB point;
			point.x = std::stod(vecSubA[0]);
			point.y = std::stod(vecSubA[1]);
			point.z = std::stod(vecSubA[2]);
			uint8_t r = int(std::stod(vecSubA[3]));
			uint8_t g = int(std::stod(vecSubA[4]));
			uint8_t b = int(std::stod(vecSubA[5]));

			uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
			point.rgb = *reinterpret_cast<float *>(&rgb);
			cloudA->points.push_back(point);
		}

		// ���� KdTree
		// �� A ������ӵ� KdTree ��
		pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
		kdtree.setInputCloud(cloudA);
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);//����1���������
		std::vector<float> pointNKNSquaredDistance(K);

		// �������� B ����
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZRGB>);
		int sizeCloudB = vecB.size();
		std::string path("E:/color.txt");
		std::ofstream ofs(path);
		for (int i = 0; i < sizeCloudB; ++i){
			pcl::PointXYZRGB p;
			p.x = std::stod(vecB[i][0]);
			p.y = std::stod(vecB[i][1]);
			p.z = std::stod(vecB[i][2]);
			if (kdtree.nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){
				// ֱ��������ƣ���Ҫ����ӵ���
				pcl::PointXYZRGB p = cloudA->points[pointIdxNKNSearch[0]];
				uint32_t rgb = *reinterpret_cast<int*>(&p.rgb);
				uint8_t r = (rgb >> 16) & 0x0000ff;
				uint8_t g = (rgb >> 8) & 0x0000ff;
				uint8_t b = (rgb)& 0x0000ff;

				ofs << vecB[i][0] << " " << vecB[i][1] << " " << vecB[i][2] << " "
					<< (int)r << " " << (int)g << " " << (int)b << "\n";
			}
		}
		ofs.close();

		std::cout << " > ������ɫ��� ...		" << path << std::endl;

		return 1;
	}

	/*
	* \function: ��ԭʼ������Ѱ�����յĽ������
	* \brief   : ��ȡ���ս�����
	* \param   : vecA ԭʼ�Ľ��������
	* \param   : vecB ƽ�����
	* \output  : �������Զ�ȡ״̬
	* \author  : Neverland_LY
	*/
	int findFinalBuilding(const std::vector<std::vector<std::string>> & vecA, const std::vector<std::vector<std::string>> & vecB){

		std::cout << " > ��ʼ�Ե��ƽ��з��� ...\n";

		// ��ȡƽ����ƹ���KdTree
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubB : vecB){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubB[0]);
			point.y = std::stod(vecSubB[1]);
			point.z = std::stod(vecSubB[2]);
			cloudB->points.push_back(point);
		}

		// ����KdTree
		// �� B ������ӵ� KdTree ��
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud(cloudB);
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);//����1���������
		std::vector<float> pointNKNSquaredDistance(K);

		std::string path1("E:/current-one.txt"), path2("E:/current-another.txt");
		std::ofstream ofs_building(path1);
		std::ofstream ofs_vege(path2);
		for (int i = 0; i < vecA.size(); ++i){
			pcl::PointXYZ point;
			point.x = std::stod(vecA[i][0]);
			point.y = std::stod(vecA[i][1]);
			point.z = 0.0f;
			if (kdtree.nearestKSearch(point, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){
				if (pointNKNSquaredDistance[0] < 1.0f){
					ofs_building << vecA[i][0] << " " << vecA[i][1] << " " << vecA[i][2] << "\n";
				}
				else{
					ofs_vege << vecA[i][0] << " " << vecA[i][1] << " " << vecA[i][2] << "\n";
				}
			}
		}
		ofs_building.close();
		ofs_vege.close();

		std::cout << " > ���Ʒ������ ...		" << path1 << "	" << path2 << std::endl;

		return 1;
	}
}

/*
 * \function printUsage
 * \brief ��ӡ����ʾ������Ϣ
 * \param ��
 * \output ������ر�־
 **/
int printUsage(){

	// ��ʾ��
	std::cout << "****************************************************\n" << std::endl;
	std::cout << "	���¹��ܽ����ڱ�ҵ�����ʱ���ݴ�������\n" << std::endl;
	std::cout << "****************************************************\n" << std::endl;

	// ����չʾ��ǰ��ʲô����
	std::cout << "�����б�\n" << std::endl;
	std::vector<std::string> vecFunc{
		"�ϲ������ֶΣ��ڵ��� A ��ѡ���ֶκ�ֱ��׷�� B ��ָ���ֶΣ�", // 1
		"����˳��������Ե��� B ������Ϊ�������������� A ��˳��",
		"���Ʒֿ飨�� X �� Y ����ȡ���Ƴ��Ϳ����Сֵ��Ȼ�� 9 �ȷ֣�",
		"����ͶӰ��ˮƽ�棨Ĭ��Ϊ 0.0m��",
		"�����ܶȶ�ֵ�������õ����ܶ���ֵ�������ƶ��ȷ֣�", // 5
		"��ȡ���Ʊ߽磨���Ʊ�����ͶӰ��ƽ�棩",
		"����������ԣ����Ƶĸ������ܶȣ�",
		"�������죨����� A �в��ڵ��� B �еĵ㣬�� A - B��",
		"������ɫ���Ե��� A Ϊ�ο��������õ����� B ����ɫ��",
		"����ͶӰλ�ý����Ʒֳ����ࣨ���� A Ϊԭʼ���ƣ����� B Ϊƽ����� ��", // 10
		"���ɶ������źţ������ж��������� XYZIcrRGBh ���� BinarySignal((XYZ0101010101)) ��",
		"ѵ�������磨�� BinarySignal(XYZ0101010101) ��Ϊ test_tag(XYZ010101��",
		"��ʼ�����ݼӱ�ǩ���� test_tag(XYZ010101) ��Ϊ classfication-6(XYZN)��", // 13
		"δ����BPѵ����ֱ�ӽ��з��ࣨ�� BinarySignal((XYZ0101010101)) ���� classfication-10(XYZN)��",
		"���ݵ�������(XYZ01010)����XYZ1-5����������ֵͬ�ĸ���������" };
	int index = 1;
	for (auto v : vecFunc){
		std::cout << "	" << std::to_string(index++) << "��" << v << std::endl;
	}

	// �жϹ���ѡ��cincin
	while (1){
		std::cout << "\n > ��������ѡ��Ĺ��ܣ� ";
		std::string numFunc;
		std::cin >> numFunc;
		std::cout << std::endl;
		if (numFunc == "EXIT" || numFunc == "exit"){
			std::cout << "Exit program ...\n";
			exit(0); break;
		}
		if (numFunc.length() > 2 || numFunc.length() < 1){
			std::cout << "�����������������룡\n"; continue;
		}

		return std::stoi(numFunc);
	}

	return -1;
}

/*
 * \function excuteFunc
 * \brief ����ָ��˳��ִ�г���
 * \param numFunc ˳���
 * \output ���״̬
 **/
int excuteFunc(int numFunc){

	// ��ʼ��һЩ�����������ò���
	std::vector<int> vecFunc(50);
	std::iota(vecFunc.begin(), vecFunc.end(), 0);

	// �����������
	std::vector<std::vector<std::string>> vecCloudA, vecCloudB;
	std::string pathA, pathB;
	if (numFunc == 12){ // �ض�������
		LXT::BPTrain();
	}
	else if (numFunc == 1 || numFunc == 2 || numFunc == 8 || numFunc == 9 || numFunc == 10){
		// ��Ҫ������������
		std::cout << " > ��������� A ��·����";
		std::cin >> pathA;
		std::cout << " > ��������� B ��·����";
		std::cin >> pathB;
		std::cout << " > ���ڶ�ȡ�������� ...\n";
		std::cout << " > ���ڶ�ȡ���� A ...";
		if (IO::loadTxt(pathA.c_str(), vecCloudA) < 0){
			std::cout << "���� A ·������ȷ��\n";
			return -1;
		}
		std::cout << "		������ " << vecCloudA.size() << std::endl;
		std::cout << " > ���ڶ�ȡ���� B ...";
		if (IO::loadTxt(pathB.c_str(), vecCloudB) < 0){
			std::cout << "���� B ·������ȷ��\n";
			return -1;
		}
		std::cout << "		������ " << vecCloudB.size() << std::endl;
		std::cout << " > �������ݶ�ȡ��� ...\n";
	}
	else{
		// ��Ҫһ����������
		std::cout << " > ��������Ƶ�·����";
		std::cin >> pathA; std::cout << std::endl;
		std::cout << " > ���ڶ�ȡ�������� ...";
		if (IO::loadTxt(pathA.c_str(), vecCloudA) < 0){
			std::cout << "����·������ȷ��\n";
			return -1;
		}
		std::cout << "		������ " << vecCloudA.size() << std::endl;
		std::cout << " > �������ݶ�ȡ��� ...\n";
	}


	// ���չ���ִ��  std::cout << " > ��ʼ���е������� ...\n";
	if (numFunc == 1) std::cout << " > �˹�����δ���� ...\n";
	else if (numFunc == 2) return LXT::sortCloud(vecCloudA, vecCloudB);
	else if (numFunc == 3) return LY::splitCloud9Blocks(vecCloudA, 0.0f); // �ڶ�������Ϊ����
	else if (numFunc == 4) return LY::setZEqual2Zero(vecCloudA);
	else if (numFunc == 5) return LY::density(vecCloudA);
	else if (numFunc == 6) return LY::extractBoundary(vecCloudA);
	else if (numFunc == 7) return LY::computePointCloudProps(vecCloudA);
	else if (numFunc == 8) return LY::deleteBFromAXYZ(vecCloudA, vecCloudB);
	else if (numFunc == 9) return LY::resampleColor(vecCloudA, vecCloudB);
	else if (numFunc == 10) return LY::findFinalBuilding(vecCloudA, vecCloudB);
	else if (numFunc == 11) return LXT::generateBinarySignal(vecCloudA);
	// else if (numFunc == 12) return LXT::BPTrain();
	else if (numFunc == 13) return LXT::addLabel6(vecCloudA);
	else if (numFunc == 14) return LXT::addLabel10(vecCloudA);
	else if (numFunc == 15) return LXT::generateSignalByWeight(vecCloudA);
}

int main(){

	// �������Ļ���
	setlocale(LC_ALL, "Chinese-simplified");

	// ��ӡ������Ϣ��ѡ����
	int numFunc = printUsage();
	if (numFunc < 0){
		std::cout << "������Ч!\n"; exit(-1);
	}

	// ������ִ�г���
	if (excuteFunc(numFunc) < 0){
		std::cout << " > ����ִ��ʧ�ܣ�\n"; return -1;
	}

	// ִ�����
	std::cout << "\n > ����ִ����ϣ������Ѿ��˳���\n\n"; Beep(500, 700);

	return 0;
}
