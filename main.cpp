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

// 定义 IO 操作
namespace IO{

	/*
	 * \function: 将 txt 文件转为 std::vector
	 * \brief   :
	 * \param   : path txt文件的路径
	 * \param   : vec 输出变量
	 * \output  : 判断值
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
	* \brief 遍历一个文件夹下的所有点云文件
	* \param path 文件路径(绝对路径或者相对路径)
	* \param file_abs_path 路径
	* \param file_name 文件名
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

// 李晓天的代码合集
std::vector<std::vector<std::string>> VEC;  // 全局变量
namespace LXT{

	/*
	* \function: sortCloud
	* \brief   : 对点云进行重新排序
	* \param   : vecA 要纠正的点云
	* \param   : vecB 标准顺序点云
	* \output  : 判断标志位
	* \author  : Neverland_LY
	*/
	int sortCloud(const std::vector<std::vector<std::string>> & vecA,
		const std::vector<std::vector<std::string>> & vecB){

		std::cout << " > 开始进行点云排序 ...\n";

		// 将点云 A 转为 pcl::cloudXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubA : vecA){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubA[0]);
			point.y = std::stod(vecSubA[1]);
			point.z = std::stod(vecSubA[2]);
			cloudA->points.push_back(point);
		}

		// 将点云 B 转为 pcl::cloudXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubB : vecB){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubB[0]);
			point.y = std::stod(vecSubB[1]);
			point.z = std::stod(vecSubB[2]);
			cloudB->points.push_back(point);
		}

		// 构建kdtree
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud(cloudA); // 注意要对A进行排序，这里添加的是A点云
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);//进行1邻域点搜索
		std::vector<float> pointNKNSquaredDistance(K);

		// 开始检索
		if (cloudA->points.size() != cloudB->points.size()){
			std::cout << "点云个数不相等，请确认！\n";
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
				// 找到点了，重新输出
				int index = pointIdxNKNSearch[0];
				for (int k = 0; k < vecA[0].size(); ++k){
					ofs << vecA[index][k] << " ";
				}
				ofs << '\n';
			}
		}
		ofs.close();

		std::cout << " > 点云排序完成 ...		" << path << std::endl;

		return 1;
	}

	/*
	* \function: isInRange
	* \brief   : 判断一个值是否落入一定区间
	* \param   : num 要判断的数
	* \param   : range 区间集合
	* \output  : 判断标志位
	* \author  : Neverland_LY
	*/
	int isInRange(float num, const std::vector<std::pair<float, float>> & range){
		for (auto pa : range){
			if (num >= pa.first && num <= pa.second)
				return 1; // 找到了
		}
		return -1; // 找不到
	}

	/*
	* \function: generateBinarySignal
	* \brief   : 根据规则生成二进制信号
	* \param   : vec 点云数据，数据依次为 XYZIRoughReturnsRGBH
	* \output  : 判断标志位
	* \author  : Neverland_LY
	*/
	int generateBinarySignal(const std::vector<std::vector<std::string>> & vec){
		// 对照
		// 012  3    4      5      6  7  8  9
		// XYZ  I  Rough  Returns  R  G  B  H
		std::cout << " > 开始进行二进制信号生成 ...\n";
		std::string path("E:/BinarySignal.txt");
		std::ofstream ofs_bin(path);

		// 人工建筑 11
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

		// 道路 01
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

		// 裸地 10
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

		// 对照
		// 012  3    4      5      6  7  8  9
		// XYZ  I  Rough  Returns  R  G  B  H
		for (auto vecSub : vec){
			// nan 设置成 0.00f
			for (auto & value : vecSub){
				if (value == "nan") value = "0.00";
			}

			std::string str("");
			// 首先判断 I
			// str += " ";
			float intensity = std::stod(vecSub[3]);
			if (isInRange(intensity, vecBuildings) > 0) str += "1 1";
			else if (isInRange(intensity, vecRoads) > 0) str += "0 1";
			else if (isInRange(intensity, vecEarths) > 0) str += "1 0";
			else str += "0 0";

			// 判断Rough
			str += " ";
			float rough = std::stod(vecSub[4]);
			if (rough >= 0.4f) str += "1 1";
			else if (rough >= 0.15f) str += "1 0";
			else if (rough <= 0.045f & rough >= 0.015f) str += "0 1";
			else str += "0 0";

			// 判断回波次数
			str += " ";
			int ret = (int)(std::stod(vecSub[5]));
			if (ret == 4) str += "1 1";
			else if (ret == 3) str += "1 0";
			else if (ret == 2) str += "0 1";
			else str += "0 0";

			// 判断颜色
			str += " ";
			int r = std::stod(vecSub[6]); int g = std::stod(vecSub[7]); int b = std::stod(vecSub[8]);
			if (r == 255 && g == 0) str += "1 1";
			else if (r == 46) str += "0 1";
			else if (r == 255 && g == 255) str += "1 0";
			else str += "0 0";

			// 判断高度(abs)
			str += " ";
			float deltaH = std::fabs(std::stod(vecSub[9]));
			if (deltaH >= 9.0f) str += "1 1";
			else if (deltaH >= 4.0f) str += "1 0";
			else if (deltaH >= 0.1f && deltaH <= 3.0f) str += "0 1";
			else str += "0 0";

			// 输出结果
			ofs_bin << vecSub[0] << " " << vecSub[1] << " " << vecSub[2] << " " << str << '\n';
		}
		ofs_bin.close();

		std::cout << " > 二进制信号生成完毕 ...	" << path << std::endl;

		return 1;
	}

	int AI(std::vector<std::vector<double>> const & pTrainData,
		std::vector<std::vector<double>> const & pTrainTag,
		std::vector<std::vector<double>> const & pTestData,
		std::vector<std::vector<double>> & pTestTag){

		// 整合输入数据
		int numTrain = pTrainData.size();
		std::vector<sample> sampleInput(numTrain);
		for (int i = 0; i < numTrain; i++)
		{
			sampleInput[i].in = pTrainData[i];
			sampleInput[i].out = pTrainTag[i];
		}

		// 训练模型
		std::vector<sample> sampleGroupInput(std::begin(sampleInput), std::end(sampleInput));
		BpNet testNet;
		testNet.training(sampleGroupInput, 0.001);

		// 整合输出数据
		int numTest = pTestData.size();
		std::vector<sample> sampleOutput(numTest);
		for (int i = 0; i < numTest; i++){
			sampleOutput[i].in = pTestData[i];
			// sampleOutput[i].out = pTestTag[i]; // 此时为空
		}

		// 测试模型
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
			std::stringstream ss(lineStr); // getline 用法1
			std::string str;
			std::vector<std::string> lineArray;
			while (getline(ss, str, ' ')) // getline 用法2
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
			std::stringstream ss(lineStr); // getline 用法1
			std::string str;
			std::vector<std::string> lineArray;
			while (getline(ss, str, ' ')) // getline 用法2
				lineArray.push_back(str);

			// 保留原始数据精度
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

		std::cout << " > 开始读取训练数据 ...\n";

		// 定义 train_num | test_num
		int train_num = 5; // 训练数据个数 ***
		int test_num = 1394229;  // 测试数据个数 ***

		// 定义 train_data | train_tag | test_data | test_tag
		std::vector<std::vector<double>> train_data(train_num);
		std::vector<std::vector<double>> train_tag(train_num);
		std::vector<std::vector<double>> test_data(test_num);
		std::vector<std::vector<double>> test_tag(test_num);

		// 读入数据
		if (readData(train_data, "E:/train_data.txt") < 0 ||
			readData(train_tag, "E:/train_tag.txt") < 0 ||
			readData2(test_data, "E:/BinarySignal.txt") < 0){
			return -1;
		}

		std::cout << " > 开始训练模型 ...\n";

		// 训练
		if (AI(train_data, train_tag, test_data, test_tag) < 0){
			std::cout << " > 模型训练发生未知错误！\n";
			return -1;
		}

		// 输出预测结果
		std::cout << " > 开始生成分类结果 ...\n";
		std::ofstream ofs("E:/test_tag.txt");
		for (int i = 0; i < test_tag.size(); i++)
		{
			std::string str = "";
			str += VEC[i][0] + " ";
			str += VEC[i][1] + " ";
			str += VEC[i][2] + " ";
			for (int j = 0; j < test_tag[i].size(); j++){ // 结果四舍五入
				str += std::to_string((int)(std::round(test_tag[i][j]))) + " ";
			}
			ofs << str.substr(0, str.length() - 1) << '\n';
		}
		ofs.close();

		std::cout << " > 训练模型完毕 ...		" << "E:/test_tag.txt\n";

		return 1;
	}

	// 给数据加标签
	int addLabel6(const std::vector<std::vector<std::string>> & vec){

		std::cout << " > 开始给数据加标签( 6 个 0 & 1) ...\n";
		// 判断
		std::string path("E:/classfication-6.txt");
		std::ofstream ofs(path);
		for (auto vv : vec) {
			std::string str = vv[3] + vv[4] + vv[5] + vv[6] + vv[7] + vv[8];
			// 标志位
			int index = 0;
			if (str == "011011") index = 1; //树
			else if (str == "100111") index = 2; // 房子
			else if (str == "011001") index = 3; // 植被
			else if (str == "110100") index = 4; // 道路
			else if (str == "101100") index = 5; // 裸地
			else if (str == "000000") index = 6; // 水

			// 输出到本地
			ofs << vv[0] << " " << vv[1] << " " << vv[2] << " " << std::to_string(index) << '\n';
		}
		ofs.close();

		std::cout << " > 数据加标签完毕 ...		" << path << std::endl;

		return 1;
	}

	// 给数据加标签
	int addLabel10(const std::vector<std::vector<std::string>> & vec){

		std::cout << " > 开始给数据加标签( 10 个 0 & 1) ...\n";
		// 判断
		std::string path("E:/classfication-10.txt");
		std::ofstream ofs(path);
		for (auto vv : vec) {
			std::string str = vv[3] + vv[4] + vv[5] + vv[6] + vv[7] + vv[8] + vv[9] + vv[10] + vv[11] + vv[12];
			// 标志位
			int index = 0;
			if (str == "0011110011") index = 1; //树
			else if (str == "1100101101") index = 2; // 房子
			else if (str == "0001010010") index = 3; // 植被
			else if (str == "0110000100") index = 4; // 道路
			else if (str == "1000001000") index = 5; // 裸地
			else if (str == "0000000000") index = 6; // 水

			// 输出到本地
			ofs << vv[0] << " " << vv[1] << " " << vv[2] << " " << std::to_string(index) << '\n';
		}
		ofs.close();

		std::cout << " > 数据加标签完毕 ...		" << path << std::endl;

		return 1;
	}

	// 判断相同数字的位数有几位
	int getSameNumByBit(const std::string str, const std::string std_str){
		int num = 0;
		for (int i = 0; i < str.size(); ++i){
			if (str[i] == std_str[i]) num++;
		}
		return num;
	}

	// 获得权重
	int getWeight(std::string str){
		// 小天指定的标准信号（train-tag）
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

	// 将5个 0 和 1 与规定的数字进行比较，根据相同的个数作为权重值
	int generateSignalByWeight(const std::vector<std::vector<std::string>> & vec){
		std::cout << " > 正在输出结果 ...\n";
		std::string path("E:/SignalByBit.txt");
		std::ofstream ofs(path);
		for (auto v : vec){
			std::string str = v[3] + v[4] + v[5] + v[6] + v[7];
			ofs << v[0] << " " << v[1] << " " << v[2] << " " << std::to_string(getWeight(str) + 1) << std::endl;
		}
		ofs.close();
		std::cout << " > 运行完毕 ...		" << path << std::endl;

		return 1;
	}
}

// Neverland_LY 的代码合集
namespace LY{

	/*
	 * \function: setZEqual2Zero
	 * \brief   : 将点云的 Z 变为 0（主要用于建筑物分块投影）
	 * \param   : vecCloud 输入点云
	 * \param   : path 点云输出路径
	 * \output  : 状态标志位
	 * \author  : Neverland_LY
	 */
	int setZEqual2Zero(const std::vector<std::vector<std::string>> & vecCloud){

		std::cout << " > 开始进行点云投影 ...\n";

		std::string path("E:/zero.txt");
		std::ofstream ofs(path);
		for (auto vecSub : vecCloud){
			for (int i = 0; i < vecSub.size(); ++i){
				if (i == 2){ // 此时候是 Z 字段
					ofs << "0.00 ";
					continue;
				}
				ofs << vecSub[i] << " ";
			}
			ofs << "\n";
		}
		ofs.close();
		std::cout << " > 点云投影完成 ...		" << path << std::endl;
		return 1;
	}

	/*
	 * \function: splitCloud9Blocks
	 * \brief   : 主要用于将点云进行九宫格划分（局部示意）
	 * \param   :
	 * \output  :
	 * \author  : Neverland_LY
	 */
	int splitCloud9Blocks(const std::vector<std::vector<std::string>> & vecCloud, float deltaMeshWidth = 0.0f){

		std::cout << " > 开始进行点云分块 ...\n";

		// 先计算长宽
		float maxX = -1 * INFINITY;
		float maxY = -1 * INFINITY;
		float maxZ = -1 * INFINITY;
		float minX = INFINITY;
		float minY = INFINITY;
		float minZ = INFINITY;
		for (auto vecSub : vecCloud){
			for (int i = 0; i < vecSub.size(); ++i){
				// 重置 X
				if (std::stod(vecSub[0]) > maxX) maxX = std::stod(vecSub[0]);
				else if (std::stod(vecSub[0]) < minX) minX = std::stod(vecSub[0]);

				// 重置 Y
				if (std::stod(vecSub[1]) > maxY) maxY = std::stod(vecSub[1]);
				else if (std::stod(vecSub[1]) < minY) minY = std::stod(vecSub[1]);

				// 重置 Z
				if (std::stod(vecSub[2]) > maxZ) maxZ = std::stod(vecSub[2]);
				else if (std::stod(vecSub[2]) < minZ) minZ = std::stod(vecSub[2]);
			}
		}

		// 取宽作为格网格网大小
		float meshWidth = ((maxX - minX) > (maxY - minY)) ? (maxX - minX) : (maxY - minY);

		// 换算两个 X 节点
		float x1 = minX + meshWidth / 3;
		float x2 = minX + meshWidth / 3 * 2;

		// 换算两个 Y 节点
		float y1 = minY + meshWidth / 3;
		float y2 = minY + meshWidth / 3 * 2;

		// 定义九宫格缝隙间距
		// float deltaMeshWidth = 30.0f;

		// 判断落入那个方块内
		// 1  2  3
		// 4  5  6
		// 7  8  9
		// 采取 X 从大到小，Y 从大到小
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

		std::cout << " > 点云分块完成 ...		" << "E:/Block-X.txt\n";

		return 1;
	}

	/*
	* \brief 计算空间两点的距离
	* \param  p1 点1
	* \param  p2 点2
	* \return 两点间的空间距离
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
	* 最小二乘法直线拟合（不是常见的一元线性回归算法）
	* 将离散点拟合为  a x + b y + c = 0 型直线
	* 假设每个点的 X Y 坐标的误差都是符合 0 均值的正态分布的。
	* 与一元线性回归算法的区别：一元线性回归算法假定 X 是无误差的，只有 Y 有误差。
	*/
	bool lineFit(const std::vector<Point> &points, float &a, float &b, float &c) {

		// 小于 2 个点
		int size = points.size();
		if (size < 2) {
			a = 0;
			b = 0;
			c = 0;
			return false;
		}

		// 大于 3 个点
		float x_mean = 0.0f;
		float y_mean = 0.0f;
		for (int i = 0; i < size; i++) {
			x_mean += points[i].x;
			y_mean += points[i].y;
		}

		x_mean /= size; y_mean /= size; //至此，计算出了 x y 的均值

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
	 * \brief   : 提取道路边界
	 * \param   : vecCloud 输入点云
	 * \param   : pathOutput 输出路径
	 * \output  :
	 * \author  : Neverland_LY
	 */
	int extractBoundary(const std::vector<std::vector<std::string>> & vecCloud){

		std::cout << " > 开始提取点云边界 ...\n";

		// 点云转换到 pcl::PointXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSub : vecCloud){
			pcl::PointXYZ point;
			point.x = std::stod(vecSub[0]);
			point.y = std::stod(vecSub[1]);
			point.z = std::stod(vecSub[2]);
			cloud->points.push_back(point);
		}

		// 构建KdTree
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; // KdTree
		kdtree.setInputCloud(cloud);

		// 标准距离 4r/3PI ，半径取 1.0f
		float radius;
		std::cout << " > 请输入搜索半径：";
		std::cin >> radius;
		float stdDistance = 4 * radius / (3 * std::acos(-1));

		// 开始遍历所有点
		std::string path("E:/border.txt");
		std::ofstream ofs(path);
		for (size_t i = 0; i < cloud->points.size(); ++i){

			// 保存邻域点
			std::vector<int> vecIdx;
			std::vector<float> vecDis;

			// 计算点到
			Eigen::Vector4f xyz_vectroid; // 定义质心
			if ((kdtree.radiusSearch(cloud->points[i], radius, vecIdx, vecDis)) > 0){
				pcl::compute3DCentroid(*cloud, vecIdx, xyz_vectroid);
			}


			// 计算质心和圆心的空间距离
			Eigen::Vector3f p1 = Eigen::Vector3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
			Eigen::Vector3f p2 = Eigen::Vector3f(xyz_vectroid[0], xyz_vectroid[1], xyz_vectroid[2]);
			float distance = calcDistanceOf2Points(p1, p2);

			// 判断距离是否满足阈值
			if ((distance > stdDistance * 0.7) && (distance < stdDistance * 1.3)){
				ofs << vecCloud[i][0] << " " << vecCloud[i][1] << " " << vecCloud[i][2] << " 255 255 0\n";
			}
		}
		ofs.close();

		std::cout << " > 点云边界提取完毕 ...		" << path << std::endl;

		return 1;
	}

	/*
	 * \function: density
	 * \brief   : 根据点云进行密度二值化
	 * \param   :
	 * \output  :
	 * \author  : Neverland_LY
	 */
	int density(const std::vector<std::vector<std::string>> & vecCloud){

		std::cout << " > 开始计算点云密度 ...\n";

		// 点云转换到 pcl::PointXYZRGB
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSub : vecCloud){
			pcl::PointXYZ point;
			point.x = std::stod(vecSub[0]);
			point.y = std::stod(vecSub[1]);
			point.z = std::stod(vecSub[2]);
			cloud->points.push_back(point);
		}

		// 构建KdTree
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; // KdTree
		kdtree.setInputCloud(cloud);

		// 计算所有点的点云密度
		std::vector<int> vecIdx;
		std::vector<float> vecDis;
		float radius;
		std::cout << " > 请输入搜索半径：";
		std::cin >> radius;
		std::vector<int> vecDensity(cloud->points.size());
		for (size_t i = 0; i < cloud->points.size(); ++i){
			// 这里是用一定半径范围内的点数来表达点的密度
			if ((kdtree.radiusSearch(cloud->points[i], radius, vecIdx, vecDis)) > 0){
				vecDensity[i] = vecIdx.size();
			}
		}

		// 求密度均值
		int minDensity = *min_element(vecDensity.begin(), vecDensity.end());
		int maxDensity = *max_element(vecDensity.begin(), vecDensity.end());
		float sumDensity = std::accumulate(vecDensity.begin(), vecDensity.end(), 0.0f);
		sumDensity /= vecDensity.size();

		std::cout << " > 点云密度最小值： " << minDensity << std::endl;
		std::cout << " > 点云密度最大值： " << maxDensity << std::endl;
		std::cout << " > 点云密度平均值： " << maxDensity << std::endl;

		// 输出点云
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
		std::cout << " > 点云密度计算完毕 ...		" << path << std::endl;
		return 1;
	}

	/*
	 * \function: 计算点云的属性：点云的密度（每平方米）、最大高差，地面点、非地面点
	 * \brief   : 计算点云属性
	 * \param   : vecCloud 点云数据
	 * \output  : 数据属性读取状态
	 * \author  : Neverland_LY
	 */
	int computePointCloudProps(const std::vector<std::vector<std::string>> & vecCloud){

		std::cout << " > 开始计算点云属性 ...\n";

		// 点云转换到 pcl::PointXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSub : vecCloud){
			pcl::PointXYZ point;
			point.x = std::stod(vecSub[0]);
			point.y = std::stod(vecSub[1]);
			point.z = std::stod(vecSub[2]);
			cloud->points.push_back(point);
		}

		// 构建KdTree
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; // KdTree
		kdtree.setInputCloud(cloud);

		// 开始遍历点
		float heightMax = -INFINITY; // 初始化高程最高点
		float heightMin = INFINITY; // 初始化高程最低点
		float density = 0.0f; // 初始化密度
		int NLoop = 0;
		int sumNeighbor = 0;
		float radius;
		std::cout << " > 请输入搜索半径：";
		std::cin >> radius;
		for (auto point : cloud->points){

			// 计算高差用
			heightMax = (heightMax > point.z) ? heightMax : point.z;
			heightMin = (heightMin < point.z) ? heightMin : point.z;

			// 计算密度用
			std::vector<int> vecIdx;
			std::vector<float> vecDis;
			if ((kdtree.radiusSearch(point, radius, vecIdx, vecDis)) > 0){
				sumNeighbor += vecIdx.size();
			}
			NLoop++;
		}

		std::cout << "		高程最大值： " << heightMax << std::endl;
		std::cout << "		高程最小值： " << heightMin << std::endl;
		std::cout << "		最大高程差： " << heightMax - heightMin << std::endl;
		std::cout << "		点云平均密度：" << (float)sumNeighbor / NLoop << std::endl;

		std::cout << " > 点云属性计算完毕 ...\n";

		return 1;
	}

	/*
	* \function: 从点云A中剔除点云B，用于从原始点云中去除点云B
	* \brief   : 注意是A-B
	* \param   : cloudA 点云A
	* \param   : cloudA 点云B
	* \output  : 数据属性读取状态
	* \author  : Neverland_LY
	*/
	int deleteBFromAXYZ(const std::vector<std::vector<std::string>> & vecA, const std::vector<std::vector<std::string>> & vecB){

		std::cout << " > 开始进行求异 ...\n";

		// vecA 转换到 pcl::PointXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubA : vecA){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubA[0]);
			point.y = std::stod(vecSubA[1]);
			point.z = std::stod(vecSubA[2]);
			cloudA->points.push_back(point);
		}

		// vecB 转换到 pcl::PointXYZ
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubB : vecB){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubB[0]);
			point.y = std::stod(vecSubB[1]);
			point.z = std::stod(vecSubB[2]);
			cloudB->points.push_back(point);
		}

		// 将 B 点云添加到 KdTree 中
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud(cloudB);
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);//进行1邻域点搜索
		std::vector<float> pointNKNSquaredDistance(K);

		//设置参数
		int sizeA = (int)cloudA->points.size(); int sizeB = (int)cloudB->points.size();

		// 标记位，其中0表示无重合点，1表示有重合点
		std::vector<int> flagA(sizeA, 0); std::vector<int> flagB(sizeB, 0);

		// 存放A中异于B的点
		std::vector<int> LA;
		LA.clear();  //存放A中异于B的点，初始为空, 不定义长度，因此后面要用push_back来压入数据

		// 开始重置标记位
		for (int i = 0; i < sizeA; ++i){
			if (kdtree.nearestKSearch(cloudA->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){
				if (pointNKNSquaredDistance[0] < 0.001)
					flagA[i] = 1; flagB[pointIdxNKNSearch[0]] = 1;
			}
		}

		// 记录点号
		for (int i = 0; i < sizeA; ++i){
			if (flagA[i] == 0)
				LA.push_back(i);
		}

		// 存储点云  A - B = C
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudC(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*cloudA, LA, *cloudC);//按照索引复制点云

		// 输出点云
		std::cout << " > 正在保存点云 ...\n";
		std::string path("E:/AminusB.txt");
		std::ofstream ofs(path);
		for (auto p : cloudC->points){
			ofs << std::to_string(p.x) << " " << std::to_string(p.y) << " " << std::to_string(p.z) << "\n";
		}
		ofs.close();

		std::cout << " > 点云求异完毕 ...		" << path << std::endl;

		return 1;
	}

	/*
	* \function: 以点云A为基准，对点云B进行配准
	* \brief   : 注意是A-B
	* \param   : cloudA 点云A  XYZRGB
	* \param   : cloudA 点云B  XYZ
	* \output  : 数据属性读取状态
	* \author  : Neverland_LY
	*/
	int resampleColor(const std::vector<std::vector<std::string>> & vecA, const std::vector<std::vector<std::string>> & vecB){

		std::cout << " > 开始为点云进行配色 ...\n";

		// 将点云 A 转为 pcl::cloudXYZRGB
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

		// 创建 KdTree
		// 将 A 点云添加到 KdTree 中
		pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
		kdtree.setInputCloud(cloudA);
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);//进行1邻域点搜索
		std::vector<float> pointNKNSquaredDistance(K);

		// 重新生成 B 点云
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
				// 直接输出点云，不要再添加点了
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

		std::cout << " > 点云配色完毕 ...		" << path << std::endl;

		return 1;
	}

	/*
	* \function: 在原始点云中寻找最终的建筑物点
	* \brief   : 提取最终建筑物
	* \param   : vecA 原始的建筑物点云
	* \param   : vecB 平面点云
	* \output  : 数据属性读取状态
	* \author  : Neverland_LY
	*/
	int findFinalBuilding(const std::vector<std::vector<std::string>> & vecA, const std::vector<std::vector<std::string>> & vecB){

		std::cout << " > 开始对点云进行分类 ...\n";

		// 读取平面点云构建KdTree
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto vecSubB : vecB){
			pcl::PointXYZ point;
			point.x = std::stod(vecSubB[0]);
			point.y = std::stod(vecSubB[1]);
			point.z = std::stod(vecSubB[2]);
			cloudB->points.push_back(point);
		}

		// 加入KdTree
		// 将 B 点云添加到 KdTree 中
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud(cloudB);
		int K = 1;
		std::vector<int> pointIdxNKNSearch(K);//进行1邻域点搜索
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

		std::cout << " > 点云分类完毕 ...		" << path1 << "	" << path2 << std::endl;

		return 1;
	}
}

/*
 * \function printUsage
 * \brief 打印出提示操作信息
 * \param 无
 * \output 输出返回标志
 **/
int printUsage(){

	// 提示语
	std::cout << "****************************************************\n" << std::endl;
	std::cout << "	以下功能仅用于毕业设计临时数据处理！！！\n" << std::endl;
	std::cout << "****************************************************\n" << std::endl;

	// 用于展示当前有什么功能
	std::cout << "功能列表：\n" << std::endl;
	std::vector<std::string> vecFunc{
		"合并点云字段（在点云 A 的选定字段后直接追加 B 的指定字段）", // 1
		"点云顺序纠正（以点云 B 的坐标为基础，纠正点云 A 的顺序）",
		"点云分块（沿 X 和 Y 方向取点云长和宽的最小值，然后 9 等分）",
		"点云投影到水平面（默认为 0.0m）",
		"点云密度二值化（设置点云密度阈值，将点云二等分）", // 5
		"提取点云边界（点云必须已投影到平面）",
		"计算点云属性（点云的个数、密度）",
		"点云求异（求点云 A 中不在点云 B 中的点，即 A - B）",
		"点云配色（以点云 A 为参考，采样得到点云 B 的颜色）",
		"根据投影位置将点云分成两类（点云 A 为原始点云，点云 B 为平面点云 ）", // 10
		"生成二进制信号（根据判定条件，由 XYZIcrRGBh 生成 BinarySignal((XYZ0101010101)) ）",
		"训练神经网络（由 BinarySignal(XYZ0101010101) 变为 test_tag(XYZ010101）",
		"开始给数据加标签（由 test_tag(XYZ010101) 变为 classfication-6(XYZN)）", // 13
		"未进行BP训练，直接进行分类（由 BinarySignal((XYZ0101010101)) 生成 classfication-10(XYZN)）",
		"根据点云数据(XYZ01010)生成XYZ1-5（数字由相同值的个数决定）" };
	int index = 1;
	for (auto v : vecFunc){
		std::cout << "	" << std::to_string(index++) << "、" << v << std::endl;
	}

	// 判断功能选择cincin
	while (1){
		std::cout << "\n > 请输入您选择的功能： ";
		std::string numFunc;
		std::cin >> numFunc;
		std::cout << std::endl;
		if (numFunc == "EXIT" || numFunc == "exit"){
			std::cout << "Exit program ...\n";
			exit(0); break;
		}
		if (numFunc.length() > 2 || numFunc.length() < 1){
			std::cout << "输入有误，请重新输入！\n"; continue;
		}

		return std::stoi(numFunc);
	}

	return -1;
}

/*
 * \function excuteFunc
 * \brief 根据指定顺序执行程序
 * \param numFunc 顺序号
 * \output 输出状态
 **/
int excuteFunc(int numFunc){

	// 初始化一些东西，暂且用不到
	std::vector<int> vecFunc(50);
	std::iota(vecFunc.begin(), vecFunc.end(), 0);

	// 定义点云数据
	std::vector<std::vector<std::string>> vecCloudA, vecCloudB;
	std::string pathA, pathB;
	if (numFunc == 12){ // 特定化操作
		LXT::BPTrain();
	}
	else if (numFunc == 1 || numFunc == 2 || numFunc == 8 || numFunc == 9 || numFunc == 10){
		// 需要两个点云数据
		std::cout << " > 请输入点云 A 的路径：";
		std::cin >> pathA;
		std::cout << " > 请输入点云 B 的路径：";
		std::cin >> pathB;
		std::cout << " > 正在读取点云数据 ...\n";
		std::cout << " > 正在读取点云 A ...";
		if (IO::loadTxt(pathA.c_str(), vecCloudA) < 0){
			std::cout << "点云 A 路径不正确！\n";
			return -1;
		}
		std::cout << "		点数： " << vecCloudA.size() << std::endl;
		std::cout << " > 正在读取点云 B ...";
		if (IO::loadTxt(pathB.c_str(), vecCloudB) < 0){
			std::cout << "点云 B 路径不正确！\n";
			return -1;
		}
		std::cout << "		点数： " << vecCloudB.size() << std::endl;
		std::cout << " > 点云数据读取完毕 ...\n";
	}
	else{
		// 需要一个点云数据
		std::cout << " > 请输入点云的路径：";
		std::cin >> pathA; std::cout << std::endl;
		std::cout << " > 正在读取点云数据 ...";
		if (IO::loadTxt(pathA.c_str(), vecCloudA) < 0){
			std::cout << "点云路径不正确！\n";
			return -1;
		}
		std::cout << "		点数： " << vecCloudA.size() << std::endl;
		std::cout << " > 点云数据读取完毕 ...\n";
	}


	// 按照功能执行  std::cout << " > 开始进行点云排序 ...\n";
	if (numFunc == 1) std::cout << " > 此功能暂未开放 ...\n";
	else if (numFunc == 2) return LXT::sortCloud(vecCloudA, vecCloudB);
	else if (numFunc == 3) return LY::splitCloud9Blocks(vecCloudA, 0.0f); // 第二个参数为块间隔
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

	// 设置中文环境
	setlocale(LC_ALL, "Chinese-simplified");

	// 打印辅助信息，选择功能
	int numFunc = printUsage();
	if (numFunc < 0){
		std::cout << "数据无效!\n"; exit(-1);
	}

	// 按功能执行程序
	if (excuteFunc(numFunc) < 0){
		std::cout << " > 任务执行失败！\n"; return -1;
	}

	// 执行完毕
	std::cout << "\n > 任务执行完毕！程序已经退出！\n\n"; Beep(500, 700);

	return 0;
}
