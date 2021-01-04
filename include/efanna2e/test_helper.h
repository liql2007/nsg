//
// Created by liql2007 on 2020/12/23.
//

#ifndef EFANNA2E_TEST_HELPER_H
#define EFANNA2E_TEST_HELPER_H

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <sys/stat.h>
#include <efanna2e/util.h>
#include <efanna2e/distance.h>
#include <efanna2e/neighbor.h>

template<typename T>
void print_vector(const T* vec, unsigned size) {
    for (unsigned i = 0; i < size; ++i) {
        std::cout << vec[i] << "  ";
    }
    std::cout << std::endl;
}

template<typename T>
void load_data(const char* filename, T*& data, unsigned& num,
               unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    // std::cout<<"data dimension: "<<dim<<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new T[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * sizeof(T));
    }
    in.close();
}

template<typename T>
void save_data(const char* filename, std::vector<std::vector<T>>& results) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for (unsigned i = 0; i < results.size(); i++) {
        unsigned sz = (unsigned)results[i].size();
        out.write((char*)&sz, sizeof(unsigned));
        out.write((char*)results[i].data(), sz * sizeof(T));
    }

    if (out.bad()) {
        out.close();
        std::cerr << "write to file [" << filename << "] failed" << std::endl;
        exit(-1);
    }
    out.close();
}

template<typename T>
void save_data(const char* filename, const T* data, unsigned num, unsigned dim) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    for (unsigned i = 0; i < num; i++) {
        out.write((char*)&dim, sizeof(unsigned));
        out.write((char*)(data + i * dim), dim * sizeof(T));
    }
    if (out.bad()) {
        out.close();
        std::cerr << "write to file [" << filename << "] failed" << std::endl;
        exit(-1);
    }
    out.close();
}


struct GroundTruth {
  unsigned truthItemNum;
  unsigned queryNum;
  unsigned* data;
  unsigned TOPK;

  GroundTruth(unsigned TOPK) : TOPK(TOPK) {}

  void load(const char* filename) {
    load_data(filename, data, queryNum, truthItemNum);
    std::cout << "ground truth query num: " << queryNum << std::endl;
    std::cout << "ground truth item num per query: " << truthItemNum << std::endl;
  }

  void recallRate(const std::vector<std::vector<unsigned>>& res) {
    assert(TOPK <= truthItemNum);
    assert(res.size() <= queryNum);
    float avgRecallVal = 0;
    for (unsigned qi = 0; qi < res.size(); ++qi) {
      auto truth = data + qi * truthItemNum;
      unsigned recallNum = 0;
      for (auto docId : res[qi]) {
        for (unsigned j = 0; j < TOPK; ++j) {
          if (truth[j] == docId) {
            ++recallNum;
            break;
          }
        }
      }
      auto recallRateVal = (float) recallNum / TOPK;
      // recallRate.push_back(recallRateVal);
      avgRecallVal += recallRateVal;
    }
    auto recall = avgRecallVal / res.size();
    std::cout << "recall(top" << TOPK << ") : " << recall << std::endl;
  }

  static void createPartGroundTruth(const char* queryPath, const char* groundTruthPath,
                                    const float* vecData, unsigned pointNum, unsigned dim,
                                    unsigned queryNum, unsigned topK) {
    efanna2e::DistanceL2 distance;
    std::mt19937 rng(time(nullptr));
    std::vector<unsigned> queryIds(queryNum);
    efanna2e::GenRandom(rng, queryIds.data(), queryNum, pointNum);
    std::vector<std::vector<unsigned>> topNeighbors(queryNum);
    std::vector<float> qVecs(queryNum * dim);
#pragma omp parallel for
    for (unsigned i = 0; i < queryNum; ++i) {
      auto qId = queryIds[i];
      efanna2e::Neighbor nn(qId, 0, true);
      std::vector<efanna2e::Neighbor> neighborPool;
      neighborPool.reserve(topK + 1);
      neighborPool.resize(topK);
      neighborPool[0] = std::move(nn);
      unsigned poolSize = 1;
      auto q = vecData + qId * dim;
      std::memcpy(qVecs.data() + i * dim, q, dim * sizeof(float));
      for (unsigned vId = 0; vId < pointNum; ++vId) {
        if (vId == qId) {
          continue;
        }
        auto v = vecData + vId * dim;
        float dist = distance.compare(v, q, dim);
        efanna2e::Neighbor nn(vId, dist, true);
        efanna2e::InsertIntoPool(neighborPool.data(), poolSize, nn);
        if (poolSize < topK) {
          ++poolSize;
        }
      }
      assert(poolSize == topK);
      std::sort(neighborPool.begin(), neighborPool.end(),
                [](const efanna2e::Neighbor& l, const efanna2e::Neighbor& r) {
                  return l.distance < r.distance; });
      auto& queryTopNeighbor = topNeighbors[i];
      queryTopNeighbor.reserve(topK);
      for (const auto& nn : neighborPool) {
        queryTopNeighbor.push_back(nn.id);
      }
    }
    save_data(groundTruthPath, topNeighbors);
    save_data(queryPath, qVecs.data(), queryNum, dim);
  }
};

struct PartInfo {
    unsigned vecNum;
    std::string docPath;
    std::string idPath;
    std::string nsgPath;
    std::string knnPath;
    std::string queryPath;
    std::string groundTruthPath;
};

struct Partitions {
    std::vector<PartInfo> partInfos;
    unsigned totalVecNum = 0;
    std::string dirPath;
    unsigned dim;

    std::string getMetaPath() { return dirPath + "meta.txt"; }

    std::string getMergedNsgPath() { return dirPath + "merged.nsg"; }

    std::string getMergedVecPath() { return dirPath + "merged.fvecs"; }

    void serialize() {
        auto metaPath = getMetaPath();
        std::cout << "serialize partition meta to " << metaPath << std::endl;
        std::ofstream out(metaPath.c_str());
        out << "partition num: " << partInfos.size() << std::endl;
        out << "dimension: " << dim << std::endl;
        out << "total doc num: " << totalVecNum << std::endl;
        for (unsigned i = 0; i < partInfos.size(); ++i) {
            out << "partition_" << i + 1 << " doc num: " <<
                partInfos[i].vecNum << std::endl;
        }
        out.close();
    }

    void deserialize(const char* dirPath) {
        struct stat sb;
        if (stat(dirPath, &sb) != 0 || !S_ISDIR(sb.st_mode)) {
            std::cerr << dirPath << " is not dictionary" << std::endl;
            exit(-1);
        }
        std::string metaPath;
        if (dirPath[std::strlen(dirPath) - 1] != '/') {
            metaPath = dirPath + std::string("/meta.txt");
        } else {
            metaPath = dirPath + std::string("meta.txt");
        }

        std::ifstream in(metaPath.c_str());
        if (!in.is_open()) {
            std::cout << "open file " << metaPath << " failed" << std::endl;
            exit(-1);
        }
        std::string desc;
        std::getline(in, desc, ':');
        unsigned partNum;
        in >> partNum;
        std::cout << "partition num: " << partNum << std::endl;
        init(dirPath, partNum);

        std::getline(in, desc, ':');
        in >> dim;
        std::cout << "dim: " << dim << std::endl;

        std::getline(in, desc, ':');
        in >> totalVecNum;
        std::cout << "vector num: " << totalVecNum << std::endl;

        for (auto& part : partInfos) {
            std::getline(in, desc, ':');
            in >> part.vecNum;
            std::cout << "partition vector num: " << part.vecNum << std::endl;
        }
    }

    void init(const char* dirName, unsigned partNum) {
        dirPath = dirName;
        if (dirPath[dirPath.length() - 1] != '/') {
            dirPath.append("/");
        }

        partInfos.clear();
        partInfos.reserve(partNum);
        for (unsigned i = 0; i < partNum; ++i) {
            auto docPath = dirPath + "docs_" + std::to_string(i + 1) + ".fvecs";
            auto idPath = dirPath + "ids_" + std::to_string(i + 1) + ".data";
            auto nsgPath = dirPath + "nng_" + std::to_string(i + 1) + ".nsg";
            auto knnPath = dirPath + "nng_" + std::to_string(i + 1) + ".knn";
            auto queryPath = dirPath + "query_" + std::to_string(i + 1) + ".fvecs";
            auto groundTruthPath = dirPath + "groundtruth_" + std::to_string(i + 1) + ".ivecs";
            PartInfo part{0, docPath, idPath, nsgPath, knnPath, queryPath, groundTruthPath};
            partInfos.emplace_back(std::move(part));
        }
    }

    static Partitions create(const char* dirPath, unsigned partNum) {
        struct stat sb;
        if (stat(dirPath, &sb) == 0) {
            if (!S_ISDIR(sb.st_mode)) {
                std::cerr << dirPath << " is not dictionary" << std::endl;
                exit(-1);
            }
        } else if (mkdir(dirPath, 0755) != 0) {
            std::cerr << "create dictionary [" << dirPath << "] failed" << std::endl;
            exit(-1);
        }
        Partitions ret;
        ret.init(dirPath, partNum);
        return ret;
    }
};


#endif //EFANNA2E_TEST_HELPER_H
