#include "Mesh.h"
#include <algorithm>
#include <fstream>
#include <sstream>
// #include "BaseField.hpp"
#include "Face.h"
#include <sstream>
#include <map>
#include <cassert>
#include <numeric>
#include <cstring>
#include <type_traits>
#include "Parallel/Decomposition.h"
// #include <thread>
// #include "Field.hpp"



using ULL = unsigned long long;
using LL = long long;



Mesh::Mesh(const std::string& path)
    : isValid_(false)
    , meshPath_(path)
    , dimension_(Mesh::Dimension::THREE_D)
{
    if (!par::isParallel())
    {
        // 单进程：与原串行实现完全相同
        readMesh(path);
        setupSerialParallelInfo();
        return;
    }

    // 多进程：0 号进程读取完整网格并分解，其余进程接收各自的子网格
    constexpr int MESH_TAG = 7100;
    std::unique_ptr<Mesh> global;
    std::vector<char> myPack;
    std::vector<ULL> cellPositions;
    std::vector<ULL> fluxFacePositions;

    if (par::isMaster())
    {
        global.reset(new Mesh(path, SerialReadTag{}));
        std::vector<std::vector<char>> packs =
            decompose(*global, par::size(), cellPositions, fluxFacePositions);
        for (int p = 1; p < par::size(); ++p)
        {
            par::sendBytes(packs[p], p, MESH_TAG);
            std::vector<char>().swap(packs[p]);
        }
        myPack = std::move(packs[0]);
    }
    else
    {
        myPack = par::recvBytes(0, MESH_TAG);
    }

    buildLocalMesh(myPack);

    // 通量面数：内部面中 owner 为自有单元的面 + 全部边界面
    ULL nFluxFaces = boundaryFaceIndexes_.size();
    for (ULL faceId : internalFaceIndexes_)
    {
        if (faces_[faceId].getOwnerIndex() < nOwnedCells_)
        {
            ++nFluxFaces;
        }
    }
    const ULL nGlobalFluxFaces = par::allReduceSum(nFluxFaces);

    cellOrdering_.setup(nOwnedCells_, nGlobalCells_, std::move(cellPositions));
    fluxFaceOrdering_.setup(nFluxFaces, nGlobalFluxFaces, std::move(fluxFacePositions));

    if (par::isMaster())
    {
        globalMesh_ = std::move(global);
    }

    // 输出分解信息
    const ULL nGhost = getGhostCellNumber();
    const ULL minCells = static_cast<ULL>(-par::allReduceMax(-static_cast<double>(nOwnedCells_)));
    const ULL maxCells = static_cast<ULL>(par::allReduceMax(static_cast<double>(nOwnedCells_)));
    const ULL totalGhost = par::allReduceSum(nGhost);
    std::cout << "Mesh decomposed into " << par::size() << " subdomains (RCB): "
              << "cells per process min/max = " << minCells << "/" << maxCells
              << ", total ghost cells = " << totalGhost << std::endl;
}

Mesh::Mesh(const std::string& path, SerialReadTag)
    : isValid_(false)
    , meshPath_(path)
    , dimension_(Mesh::Dimension::THREE_D)
{
    readMesh(path);
    setupSerialParallelInfo();
}

Mesh::Mesh(Mesh&&) noexcept = default;
Mesh& Mesh::operator=(Mesh&&) noexcept = default;
Mesh::~Mesh() = default;

void Mesh::readMesh(const std::string& path)
{
    std::string Mesh_points = path + "/points";
    std::string Mesh_faces = path + "/faces";
    std::string Mesh_boundary = path + "/boundary";
    std::string Mesh_neighbour = path + "/neighbour";
    std::string Mesh_owner = path + "/owner";

    readPoints(Mesh_points);
    readBoundaryPatch(Mesh_boundary);
    readFaces(Mesh_faces, Mesh_owner, Mesh_neighbour);
    buildCellsFromFaces();     // 建立网格拓扑关系

    setMeshShape();

    calculateMeshInfo();       // 计算网格信息
    isValid_ = true;

}

void Mesh::writeMeshToFile(const std::string& path) const
{
    if (distributed_)
    {
        // 并行时由 0 号进程写出完整网格
        if (par::isMaster())
        {
            globalMesh_->writeMeshToFile(path);
        }
        return;
    }

    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot write to file." << std::endl;
        throw std::runtime_error("Invalid mesh write error");
    }


    // 写入点信息，调用私有函数
    writePoints(path + "/points");
    writeFaces(path + "/faces");
    writeOwnerAndNeighbour(path + "/owner", path + "/neighbour");
    writeBoundaryPatch(path + "/boundary");
}

const std::vector<Vector<Scalar>>& Mesh::getPoints() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return points." << std::endl;
        throw std::runtime_error("Invalid mesh points error");
    }
    return points_;
}

const std::vector<Face>& Mesh::getFaces() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return faces." << std::endl;
        throw std::runtime_error("Invalid mesh faces error");
    }
    return faces_;
}

const std::vector<ULL>& Mesh::getInternalFaceIndexes() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return internal face indices." << std::endl;
        throw std::runtime_error("Invalid mesh internal face indices error");
    }
    return internalFaceIndexes_;
}

const std::vector<ULL>& Mesh::getBoundaryFaceIndexes() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return boundary face indices." << std::endl;
        throw std::runtime_error("Invalid mesh boundary face indices error");
    }
    return boundaryFaceIndexes_;
}

const std::vector<Cell>& Mesh::getCells() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return cells." << std::endl;
        throw std::runtime_error("Invalid mesh cells error");
    }
    return cells_;
}

ULL Mesh::getCellNumber() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return cell number." << std::endl;
        throw std::runtime_error("Invalid mesh cell number error");
    }
    return nOwnedCells_;
}

ULL Mesh::getFaceNumber() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return face number." << std::endl;
        throw std::runtime_error("Invalid mesh face number error");
    }
    return static_cast<ULL>(faces_.size());
}

ULL Mesh::getPointNumber() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return point number." << std::endl;
        throw std::runtime_error("Invalid mesh point number error");
    }
    return static_cast<ULL>(points_.size());
}

Mesh::Dimension Mesh::getDimension() const
{
    return dimension_;
}

Mesh::MeshShape Mesh::getMeshShape() const
{
    return meshShape_;
}

ULL Mesh::getInternalCellNumber() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return internal cell number." << std::endl;
        throw std::runtime_error("Invalid mesh internal cell number error");
    }
    return static_cast<ULL>(internalFaceIndexes_.size());
}

ULL Mesh::getBoundaryFaceNumber() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return boundary face count." << std::endl;
        throw std::runtime_error("Invalid mesh boundary face count error");
    }
    ULL count = 0;
    for (const auto& [name, patch] : boundaryPatches_)
    {
        count += patch.getNFace();
    }
    return count;
}

const std::pair<ULL, ULL>& Mesh::getEmptyFaceIndexesPair() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return empty face indexes pair." << std::endl;
        throw std::runtime_error("Invalid mesh empty face indexes pair error");
    }
    return emptyFaceIndexesPair_;
}

const std::unordered_map<std::string, BoundaryPatch>& Mesh::getBoundaryPatches() const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return boundary patches." << std::endl;
        throw std::runtime_error("Invalid mesh boundary patches error");
    }
    return boundaryPatches_;
}

void Mesh::getBoundaryMessage
() const
{
    if (distributed_)
    {
        if (par::isMaster())
        {
            globalMesh_->getBoundaryMessage();
        }
        return;
    }
    if (!isValid_)
    {
        std::cerr << "Mesh::getBoundaryNames() Error: Mesh is invalid, cannot return boundary names.";
        throw std::runtime_error("Invalid mesh boundary names error");
    }
    for (const auto& [name, patch] : boundaryPatches_)
    {
        std::cout << name << ", start face: " << patch.getStartFace() << ", n face: " << patch.getNFace() << std::endl;
    }
}

ULL Mesh::getNumber(field::FieldType type) const
{
    if (!isValid_)
    {
        std::cerr << "Error: Mesh is invalid, cannot return number." << std::endl;
        throw std::runtime_error("Invalid mesh number error");
    }

    if (type == field::FieldType::CELL_FIELD)
    {
        // 单元场存储自有单元与幽灵单元
        return getLocalCellNumber();
    }
    else if (type == field::FieldType::FACE_FIELD)
    {
        return getFaceNumber();
    }
    else if (type == field::FieldType::NODE_FIELD)
    {
        return getPointNumber();
    }
    else if (type == field::FieldType::BASE)
    {
        std::cerr << "Field type not supported." << std::endl;
        std::cerr << "The type is " << static_cast<int>(type) << std::endl;
        throw std::runtime_error("Field type not supported");
    }
    std::cerr << "Unknown field type." << std::endl;
    throw std::runtime_error("Unknown field type");
}

bool Mesh::isValid() const
{
    return isValid_;
}

void Mesh::readPoints(const std::string& pointsPath)
{
    std::ifstream ifsP(pointsPath);
    if (!ifsP.is_open())
    {
        std::cerr << "Error: Unable to open points file: " << pointsPath << std::endl;
        throw std::runtime_error("Points file open error");
    }

    // 处理points文件
    std::string line;
    std::cout << "Reading points file..." << std::endl;

    // 跳过文件头, 注释和FoamFile
    while (std::getline(ifsP, line) && line.find("FoamFile") == std::string::npos) {}
    while (std::getline(ifsP, line) && line != "}") {}

    ULL pointNum = 0;
    // 记录点数，跳过(
    while (std::getline(ifsP, line) && line != "(")
    {
        // 若能转为数字
        if (!line.empty() && std::isdigit(line[0]))
        {
            pointNum = std::stoull(line);
        }
    }

    points_.reserve(pointNum);   // 提前分配空间

    // std::getline(ifsP, line);

    // 读取坐标
    char ignore;    // 接收括号和逗号
    Scalar px, py, pz;
    std::istringstream iss;
    while (std::getline(ifsP, line) && line != ")")
    {
        iss.str(line);
        iss >> ignore >> px >> py >> pz;
        // std::cout << "Read point: (" << px << ", " << py << ", " << pz << ")" << std::endl;
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        points_.emplace_back(px, py, pz);
        iss.clear();
    }


    /* ==========测试========= */

    // for (const auto& ele : points_)
    // {
    //     std::cout << ele << std::endl;
    // }
}

void Mesh::readBoundaryPatch(const std::string& boundaryPath)
{
    std::ifstream ifsB(boundaryPath);
    if (!ifsB.is_open())
    {
        std::cerr << "Error: Unable to open boundary file: " << boundaryPath << std::endl;
        throw std::runtime_error("Boundary file open error");
    }

    // 处理boundary文件
    std::string line;
    std::cout << "Reading boundary file..." << std::endl;

    // 跳过文件头, 注释和FoamFile
    while (std::getline(ifsB, line) && line.find("FoamFile") == std::string::npos) {}
    while (std::getline(ifsB, line) && line != "}") {}

    // 找 ( 括号，并记录边界类型数量
    int boundaryTypeNum = 0;
    std::string boundaryName;                   // 边界名称
    BoundaryPatch::BoundaryType type;           // 边界类型
    std::string typeStr;
    std::string ignoreStr;                     // 边界类型字符串(临时存储)
    ULL nFaces = 0;                             // 边界面数量
    ULL startFace = 0;                          // 起始面索引
    std::istringstream iss;
    while (std::getline(ifsB, line) && line != "(")
    {
        if (!line.empty() && std::isdigit(line[0]))
        {
            boundaryTypeNum = std::stoi(line);
        }
    }
    // boundaryPatch_.reserve(boundaryTypeNum);
    // 读取边界名称：inlet、outlet、upperWall

    for (int i = 0; i < boundaryTypeNum; ++i)
    {

        // 找{括号，读取上一行的名称name
        while (std::getline(ifsB, line) && line.find("{") == std::string::npos)
        {
            iss.clear();
            iss.str(line);
            iss >> boundaryName;
        }

        // 寻找type关键字
        while (std::getline(ifsB, line) && line.find("type") == std::string::npos) {}
        line.pop_back();
        iss.clear();
        iss.str(line);
        iss >> ignoreStr >> typeStr;        // 第二个line得到类型的字符串加分号，第一个得到type字符串
        // system("pause");
        type = stringToType(typeStr);

        // 寻找nFaces关键字
        while (std::getline(ifsB, line) && line.find("nFaces") == std::string::npos) {}
        line.pop_back();        // 去掉分号
        iss.clear();
        iss.str(line);
        iss >> ignoreStr >> nFaces;

        // 寻找startFace关键字
        while (std::getline(ifsB, line) && line.find("startFace") == std::string::npos) {}
        line.pop_back();
        iss.clear();
        iss.str(line);
        iss >> ignoreStr >> startFace;

        // 添加至boundaryPatch_ map中
        patchOrder_.push_back(boundaryName);
        BoundaryPatch patch(boundaryName, nFaces, startFace, type);
        boundaryPatches_.emplace(boundaryName, std::move(patch));
    }

    // 通过是否有empty来判断网格的维度，存在empty则为二维网格
    for (const auto& [name, bP] : boundaryPatches_)
    {
        if (bP.getType() == BoundaryPatch::BoundaryType::EMPTY || bP.getName() == "frontAndBack")
        {
            // 设置为二维
            dimension_ = Mesh::Dimension::TWO_D;
            // 设置empty的起止索引(左闭右开)
            emptyFaceIndexesPair_ = std::make_pair(bP.getStartFace(), bP.getStartFace() + bP.getNFace());
        }
    }



    /* 测试用 */
    // std::cout << "Boundary Patch " << i + 1 << ": " << std::endl;
    // std::cout << "  Name: " << boundaryName << std::endl;
    // std::cout << "  Type: " << static_cast<int>(type) << std::endl;
    // std::cout << "  nFaces: " << nFaces << std::endl;
    // std::cout << "  startFace: " << startFace << std::endl;
    // getchar();

}
// 测试用
// for (const auto& [name, patch] : boundaryPatch_)
// {
//     std::cout << "Boundary Patch Name: " << name << std::endl;
//     std::cout << "  Type: " << static_cast<int>(patch.getType()) << std::endl;
//     std::cout << "  nFaces: " << patch.getNFace() << std::endl;
//     std::cout << "  startFace: " << patch.getStartFace() << std::endl;
// }
// getchar();
// }

void Mesh::readFaces(const std::string& facesPath, const std::string& ownerPath, const std::string& neighbourPath)
{
    std::ifstream ifsF(facesPath, std::ios::in);
    std::ifstream ifsO(ownerPath, std::ios::in);
    if (!ifsF.is_open())
    {
        std::cerr << "Error: Unable to open faces file: " << facesPath << std::endl;
        throw std::runtime_error("Faces file open error");
    }
    if (!ifsO.is_open())
    {
        std::cerr << "Error: Unable to open owner file: " << ownerPath << std::endl;
        throw std::runtime_error("Owner file open error");
    }

    // 处理faces文件
    std::string line;
    std::cout << "Reading faces file..." << std::endl;
    // 跳过faces文件头, 注释和FoamFile
    while (std::getline(ifsF, line) && line.find("FoamFile") == std::string::npos) {}
    while (std::getline(ifsF, line) && line != "}") {}

    // 处理owner文件
    std::cout << "Reading owner file..." << std::endl;
    // 跳过owner文件头, 注释和FoamFile
    while (std::getline(ifsO, line) && line.find("FoamFile") == std::string::npos) {}
    while (std::getline(ifsO, line) && line != "}") {}


    ULL faceNum = 0;
    while (std::getline(ifsF, line) && line != "(")
    {
        if (!line.empty() && std::isdigit(line[0]))
        {
            faceNum = std::stoull(line);
        }
    }
    while (std::getline(ifsO, line) && line != "(") {}
    faces_.reserve(faceNum);

    // 开始读取面信息
    /*
    2(2 6)
    2(3 7)
    2(6 10)
    2(7 11)
    2(10 14)
    2(11 15)
    2(5 6)
    2(6 7)
     */
    char ignore;    // 接收括号
    int pointNum;
    ULL pIndex = 0;
    std::istringstream iss;
    while (std::getline(ifsF, line) && line != ")")
    {
        // 先处理faces文件
        std::vector<ULL> pointIndexs;   // 接收读取到的点索引
        iss.str(line);
        iss >> pointNum >> ignore;
        // std::cout << pointNum << ignore;
        // system("pause");
        // 创建Face对象
        pointIndexs.reserve(pointNum);
        for (int i = 0; i < pointNum; ++i)      // 读取点索引
        {
            iss >> pIndex;
            pointIndexs.push_back(pIndex);
        }

        // owner文件同步读取
        std::getline(ifsO, line);


        Face face(pointIndexs, std::stoull(line));
        faces_.emplace_back(std::move(face));
    }



    // 开始处理neighbour文件
    std::map<ULL, ULL> boundaryIndxRange;   // 边界面的起始和终止索引
    // 获取边界面的区间范围
    for (const auto& ele : boundaryPatches_)
    {
        boundaryIndxRange[ele.second.getStartFace()] = ele.second.getNFace() + ele.second.getStartFace();
    }
    std::vector<ULL> internalFaceIndices;    // 内部面索引列表
    internalFaceIndices.reserve(faces_.size()); // 分配大小

    // 首个内部区间
    int cycleIndex = 1;     // 记录循环次数
    ULL lastEnd = 0;        // 记录上一个边界面区间的结束索引

    for (const auto& [begin, end] : boundaryIndxRange)
    {
        if (cycleIndex == 1)    // 首个内部区间为0到第一个外部区间的左值
        {
            for (ULL i = 0; i < begin; ++i)
            {
                internalFaceIndices.push_back(i);
            }
            lastEnd = end;
        }
        else    // 中间的区间
        {
            for (ULL i = lastEnd; i < begin; ++i)
            {
                internalFaceIndices.push_back(i);
            }
            lastEnd = end;
        }
        cycleIndex++;
    }
    for (ULL i = lastEnd; i < faces_.size(); ++i)    // 最后一个内部面区间
    {
        internalFaceIndices.push_back(i);
    }

    // 将所有内部面的neighbor信息写入faces_
    readNeighbour(neighbourPath, internalFaceIndices);


    // 测试用
    // for (const auto& face : faces_)
    // {
    //     face.printFaceInfo();
    //     // 睡100毫秒
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // }
}

void Mesh::readNeighbour(const std::string& neighbourPath, std::vector<ULL> internalFaceIndices)
{
    std::ifstream ifsN(neighbourPath);
    if (!ifsN.is_open())
    {
        std::cerr << "Error: Unable to open neighbour file: " << neighbourPath << std::endl;
        throw std::runtime_error("Neighbour file open error");
    }

    // 处理neighbour文件
    std::string line;
    std::cout << "Reading neighbour file..." << std::endl;
    // 跳过neighbour文件头, 注释和FoamFile
    while (std::getline(ifsN, line) && line.find("FoamFile") == std::string::npos) {}
    while (std::getline(ifsN, line) && line != "}") {}

    // 读取neighbour单元的个数
    ULL neighbourNum = 0;
    while (std::getline(ifsN, line) && line != "(")
    {
        if (!line.empty() && std::isdigit(line[0]))
        {
            neighbourNum = std::stoull(line);
        }
    }

    // assert(neighbourNum == faces_.size()- internalFaceIndices.size());     // 数量应相等
    int cycleIndex = 0;     // 循环次数
    while (std::getline(ifsN, line) && line != ")")
    {
        faces_[internalFaceIndices[cycleIndex]].setNeighbor(
            std::stoull(line));
        cycleIndex++;
    }

    // // 测试用
    // for (auto& ele : faces_)
    // {
    //     ele.printFaceInfo();
    // }
}

void Mesh::writePoints(const std::string& pointsPath) const
{
    std::ofstream ofsP(pointsPath);
    if (!ofsP.is_open())
    {
        std::cerr << "Error: Unable to open points file for writing: " << pointsPath << std::endl;
        throw std::runtime_error("Points file write error");
    }

    // 写入文件头
    ofsP << "/*--------------------------------*- C++ -*----------------------------------*\\\n";
    ofsP << "| =========                 |                                                 |\n";
    ofsP << "| \\      /  F ield          | OpenFOAM: The Open Source CFD Toolbox           |\n";
    ofsP << "|  \\    /   O peration      | Version:  v2012                                 |\n";
    ofsP << "|   \\  /    A nd            | Website:  www.openfoam.com                      |\n";
    ofsP << "|    \\/     M anipulation   |                                                 |\n";
    ofsP << "\\*---------------------------------------------------------------------------*/\n";
    ofsP << "FoamFile\n";
    ofsP << "{\n";
    ofsP << "    version     2.0;\n";
    ofsP << "    format      ascii;\n";
    ofsP << "    class       vectorField;\n";
    ofsP << "    location    \"constant/polyMesh\";\n";
    ofsP << "    object      points;\n";
    ofsP << "}\n";
    ofsP << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n";
    ofsP << std::endl;
    // 写入点数量
    ofsP << points_.size() << std::endl;
    ofsP << "(" << std::endl;
    // 写入点坐标
    for (const auto& point : points_)
    {
        ofsP << " (" << point.x() << " " << point.y() << " " << point.z() << ")" << std::endl;
    }
    ofsP << ")" << std::endl;
    ofsP << std::endl;
}

void Mesh::writeFaces(const std::string& facesPath) const
{
    std::ofstream ofsF(facesPath);
    if (!ofsF.is_open())
    {
        std::cerr << "Error: Unable to open faces file for writing: " << facesPath << std::endl;
        throw std::runtime_error("Faces file write error");
    }

    // 写入文件头
    ofsF << "/*--------------------------------*- C++ -*----------------------------------*\\\n";
    ofsF << "| =========                 |                                                 |\n";
    ofsF << "| \\      /  F ield          | OpenFOAM: The Open Source CFD Toolbox           |\n";
    ofsF << "|  \\    /   O peration      | Version:  v2012                                 |\n";
    ofsF << "|   \\  /    A nd            | Website:  www.openfoam.com                      |\n";
    ofsF << "|    \\/     M anipulation   |                                                 |\n";
    ofsF << "\\*---------------------------------------------------------------------------*/\n";
    ofsF << "FoamFile\n";
    ofsF << "{\n";
    ofsF << "    version     2.0;\n";
    ofsF << "    format      ascii;\n";
    ofsF << "    class       faceList;\n";
    ofsF << "    location    \"constant/polyMesh\";\n";
    ofsF << "    object      faces;\n";
    ofsF << "}\n";
    ofsF << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n";
    ofsF << std::endl;
    // 写入面数量
    ofsF << faces_.size() << std::endl;
    ofsF << "(" << std::endl;
    // 写入面信息
    for (const auto& face : faces_)
    {
        ofsF << " " << face.getPointNum() << "(";
        const auto& pointIndices = face.getPointIndexes();
        for (size_t i = 0; i < pointIndices.size(); ++i)
        {
            ofsF << pointIndices[i];
            if (i != pointIndices.size() - 1)
            {
                ofsF << " ";
            }
        }
        ofsF << ")" << std::endl;
    }
    ofsF << ")" << std::endl;
    ofsF << std::endl;
}

void Mesh::writeOwnerAndNeighbour(const std::string& ownerPath, const std::string& neighbourPath) const
{
    std::ofstream ofsO(ownerPath);
    std::ofstream ofsN(neighbourPath);
    // 可在循环中同时写入，只需一次循环
    if (!ofsO.is_open())
    {
        std::cerr << "Error: Unable to open owner file for writing: " << ownerPath << std::endl;
        throw std::runtime_error("Owner file write error");
    }
    if (!ofsN.is_open())
    {
        std::cerr << "Error: Unable to open neighbour file for writing: " << neighbourPath << std::endl;
        throw std::runtime_error("Neighbour file write error");
    }
    // 写入文件头
    ofsO << "/*--------------------------------*- C++ -*----------------------------------*\\\n";
    ofsO << "| =========                 |                                                 |\n";
    ofsO << "| \\      /  F ield          | OpenFOAM: The Open Source CFD Toolbox           |\n";
    ofsO << "|  \\    /   O peration      | Version:  v2012                                 |\n";
    ofsO << "|   \\  /    A nd            | Website:  www.openfoam.com                      |\n";
    ofsO << "|    \\/     M anipulation   |                                                 |\n";
    ofsO << "\\*---------------------------------------------------------------------------*/\n";
    ofsO << "FoamFile\n";
    ofsO << "{\n";
    ofsO << "    version     2.0;\n";
    ofsO << "    format      ascii;\n";
    ofsO << "    class       labelList;\n";
    ofsO << "    location    \"constant/polyMesh\";\n";
    ofsO << "    object      owner;\n";
    ofsO << "}\n";
    ofsO << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n";
    ofsO << std::endl;
    // 写入owner数量
    ofsO << faces_.size() << std::endl;
    ofsO << "(" << std::endl;
    // 写入neighbour文件头
    ofsN << "/*--------------------------------*- C++ -*----------------------------------*\\\n";
    ofsN << "| =========                 |                                                 |\n";
    ofsN << "| \\      /  F ield          | OpenFOAM: The Open Source CFD Toolbox           |\n";
    ofsN << "|  \\    /   O peration      | Version:  v2012                                 |\n";
    ofsN << "|   \\  /    A nd            |                                                 |\n";
    ofsN << "|    \\/     M anipulation   |                                                 |\n";
    ofsN << "\\*---------------------------------------------------------------------------*/\n";
    ofsN << "FoamFile\n";
    ofsN << "{\n";
    ofsN << "    version     2.0;\n";
    ofsN << "    format      ascii;\n";
    ofsN << "    class       labelList;\n";
    ofsN << "    location    \"constant/polyMesh\";\n";
    ofsN << "    object      neighbour;\n";
    ofsN << "}\n";
    ofsN << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n";
    ofsN << std::endl;
    // 写入neighbour数量
    ofsN << faces_.size() - getBoundaryFaceNumber() << std::endl;
    ofsN << "(" << std::endl;
    // 写入owner和neighbour信息
    for (const auto& face : faces_)
    {
        if (face.getNeighborIndex() != -1)
        {
            ofsN << face.getNeighborIndex() << std::endl;
        }
        ofsO << face.getOwnerIndex() << std::endl;
    }
    ofsO << ")" << std::endl;
    ofsO << std::endl;
    ofsN << ")" << std::endl;
    ofsN << std::endl;

}

void Mesh::writeBoundaryPatch(const std::string& boundaryPath) const
{
    std::ofstream ofsB(boundaryPath);
    if (!ofsB.is_open())
    {
        std::cerr << "Error: Unable to open boundary file for writing: " << boundaryPath << std::endl;
        throw std::runtime_error("Boundary file write error");
    }

    // 写入文件头
    ofsB << "/*--------------------------------*- C++ -*----------------------------------*\\\n";
    ofsB << "| =========                 |                                                 |\n";
    ofsB << "| \\      /  F ield          | OpenFOAM: The Open Source CFD Toolbox           |\n";
    ofsB << "|  \\    /   O peration      | Version:  v2012                                 |\n";
    ofsB << "|   \\  /    A nd            | Website:  www.openfoam.com                      |\n";
    ofsB << "|    \\/     M anipulation   |                                                 |\n";
    ofsB << "\\*---------------------------------------------------------------------------*/\n";
    ofsB << "FoamFile\n";
    ofsB << "{\n";
    ofsB << "    version     2.0;\n";
    ofsB << "    format      ascii;\n";
    ofsB << "    class       polyBoundaryMesh;\n";
    ofsB << "    location    \"constant/polyMesh\";\n";
    ofsB << "    object      boundary;\n";
    ofsB << "}\n";
    ofsB << "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n";
    ofsB << std::endl;

    // 写入边界补丁数量
    ofsB << boundaryPatches_.size() << std::endl;
    ofsB << "(" << std::endl;
    // 写入每个边界补丁的信息
    for (const auto& [name, patch] : boundaryPatches_)
    {
        ofsB << name << std::endl;
        ofsB << "{" << std::endl;
        ofsB << "    type            ";
        switch (patch.getType())
        {
        case BoundaryPatch::BoundaryType::PATCH:
            ofsB << "patch;" << std::endl;
            break;
        case BoundaryPatch::BoundaryType::WALL:
            ofsB << "wall;" << std::endl;
            break;
        case BoundaryPatch::BoundaryType::SYMMETRY:
            ofsB << "symmetry;" << std::endl;
            break;
        case BoundaryPatch::BoundaryType::CYCLIC:
            ofsB << "cyclic;" << std::endl;
            break;
        case BoundaryPatch::BoundaryType::WEDGE:
            ofsB << "wedge;" << std::endl;
            break;
        case BoundaryPatch::BoundaryType::EMPTY:
            ofsB << "empty;" << std::endl;
            break;
        case BoundaryPatch::BoundaryType::PROCESSOR:
            ofsB << "processor;" << std::endl;
            break;
        default:
            throw std::runtime_error("Unknown boundary type during writing.");
        }

        ofsB << "    nFaces          " << patch.getNFace() << ";" << std::endl;
        ofsB << "    startFace       " << patch.getStartFace() << ";" << std::endl;
        ofsB << "}" << std::endl;
    }
    ofsB << ")" << std::endl;
}

void Mesh::buildCellsFromFaces()
{
    // 确定单元总数（owner里的最大索引）
    ULL maxCellIndex = 0;
    for (const auto& face : faces_)
    {
        maxCellIndex = std::max(maxCellIndex, face.getOwnerIndex());

        auto faceNeighborIndex = face.getNeighborIndex();
        if (faceNeighborIndex != -1)
        {
            maxCellIndex = std::max(maxCellIndex, static_cast<ULL>(faceNeighborIndex));
        }
    }

    cells_.resize(maxCellIndex + 1);    // 索引从0开始，大小为最大索引+1

    // 遍历所有面
    if (getDimension() == Dimension::THREE_D)
    {
        for (ULL i = 0; i < faces_.size(); ++i)
        {
            ULL ownerIndex = faces_[i].getOwnerIndex();
            LL neighborIndex = faces_[i].getNeighborIndex();

            if (neighborIndex == -1)    // 说明是边界面
            {
                boundaryFaceIndexes_.emplace_back(i);
                cells_[ownerIndex].addFaceIndex(i);
                boundaryCellIndexes_.emplace(ownerIndex);
            }
            else                        // 内部面
            {
                internalFaceIndexes_.emplace_back(i);
                cells_[ownerIndex].addFaceIndex(i);
                cells_[neighborIndex].addFaceIndex(i);
                internalCellIndexes_.emplace(ownerIndex);
                internalCellIndexes_.emplace(neighborIndex);
            }
        }
    }
    else if (getDimension() == Dimension::TWO_D)
    {
        // 找到empty对应的boundaryPatch
        const auto& it = std::find_if(
            boundaryPatches_.begin(), boundaryPatches_.end(),
            [](const auto& patchMap) {
                return patchMap.second.getType() ==
                    BoundaryPatch::BoundaryType::EMPTY;
            }
        );
        if (it == boundaryPatches_.end())   // 未找到则报错
        {
            std::cerr << "Mesh::buildCellsFromFaces() Error: Cannot find empty boundary patch." << std::endl;
            throw std::runtime_error("Mesh::buildCellsFromFaces() Error: Cannot find empty boundary patch.");
        }

        // 获取empty的起始索引和终止索引
        ULL startId = it->second.getStartFace();
        ULL endId = startId + it->second.getNFace();

        for (ULL i = 0; i < faces_.size(); ++i)
        {
            ULL ownerIndex = faces_[i].getOwnerIndex();
            LL neighborIndex = faces_[i].getNeighborIndex();

            if (neighborIndex == -1)    // 边界面
            {
                if (i < startId || i >= endId)  // 非empty边界界面
                {
                    boundaryFaceIndexes_.emplace_back(i);
                }
                cells_[ownerIndex].addFaceIndex(i);
                boundaryCellIndexes_.emplace(ownerIndex);
            }
            else                        // 内部面
            {
                internalFaceIndexes_.emplace_back(i);
                cells_[ownerIndex].addFaceIndex(i);
                cells_[neighborIndex].addFaceIndex(i);
                internalCellIndexes_.emplace(ownerIndex);
                internalCellIndexes_.emplace(neighborIndex);
            }
        }
    }


    // 测试用
    // std::cout << "Internal Cell Indices: ";
    // for (const auto& ele : cells_)
    // {
    //     ele.printCellInfo();
    // }
}

/* =========测试========= */
// for (const auto& ele : faces_)
// {
//     std::cout << ele << std::endl;
// }


BoundaryPatch::BoundaryType Mesh::stringToType(const std::string& name) const
{
    if (name == "patch")
    {
        return BoundaryPatch::BoundaryType::PATCH;
    }
    else if (name == "wall")
    {
        return BoundaryPatch::BoundaryType::WALL;
    }
    else if (name == "symmetry")
    {
        return BoundaryPatch::BoundaryType::SYMMETRY;
    }
    else if (name == "cyclic")
    {
        return BoundaryPatch::BoundaryType::CYCLIC;
    }
    else if (name == "wedge")
    {
        return BoundaryPatch::BoundaryType::WEDGE;
    }
    else if (name == "empty")
    {
        return BoundaryPatch::BoundaryType::EMPTY;
    }
    else if (name == "processor")
    {
        return BoundaryPatch::BoundaryType::PROCESSOR;
    }
    throw std::invalid_argument("Unknown boundary type: " + name);
}

void Mesh::setMeshShape()
{
    // 知道维度了，判断网格形状
    if (dimension_ == Dimension::TWO_D)
    {
        // 取第一个单元用总点数除以2就可以得到形状
        // std::unordered_set<ULL> points;
        // for (ULL faceIndex : cells_[0].getFaceIndexes())
        // {
        //     const std::vector<ULL>& pointIndexes = faces_[faceIndex].getPointIndexes();
        //     points.insert(pointIndexes.begin(), pointIndexes.end());
        // }
        int faceNum = cells_[0].getFaceIndexes().size();
        if (faceNum == 5)     // 三角形
        {
            meshShape_ = MeshShape::TRIANGLE;
        }
        else if (faceNum == 6)    // 四边形
        {
            meshShape_ = MeshShape::QUADRILATERAL;
        }
        else
        {
            std::cerr << "The 2D mesh face number is " << faceNum << std::endl;
            std::cerr << "Error: Unknown mesh shape" << std::endl;
            throw std::runtime_error("Unknown mesh shape");
        }
    }
    else if (dimension_ == Dimension::THREE_D)
    {
        int faceNum = cells_[0].getFaceIndexes().size();
        if (faceNum == 4)   // 四面体
        {
            meshShape_ = MeshShape::TETRAHEDRON;
        }
        else if (faceNum == 6)
        {
            meshShape_ = MeshShape::BRICK;
        }
        else
        {
            std::cerr << "The 3D mesh face number is " << faceNum << std::endl;
            std::cerr << "Error: Unknown mesh shape" << std::endl;
            throw std::runtime_error("Unknown mesh shape");
        }
    }
}

void Mesh::calculateMeshInfo()
{
    // 计算面信息
    for (auto& face : faces_)
    {
        face.calculateFaceInfo(points_);
    }

    // 计算Cell信息
    for (auto& cell : cells_)
    {
        cell.calculateCellInfo(faces_, points_);
    }

    // 修正面的法向量由owner指向neighbor，需要用到cells_的信息，为避免参数传递，所以在本函数内实现具体细节

    // bool yesFlag = false;    // 测试用
    // bool noFlag = false;
    for (auto& face : faces_)
    {
        Point ownerCenter = cells_[face.getOwnerIndex()].getCenter();
        Point faceCenter = face.getCenter();

        Vector<Scalar> ownerToFace = faceCenter - ownerCenter;

        if ((face.getNormal() & ownerToFace) < 0)
        {

            // if (!yesFlag)        // 测试用
            // {
            //     std::cout << "face_id:" << face.getCenter() << std::endl;
            //     yesFlag = true;
            //     noFlag = false;
            //     getchar();
            // }
            face.reverseNormal();
            // std::cout << "yes" << std::endl;
        }
        // else
        // {
        //     if (!noFlag)         // 测试用
        //     {
        //         std::cout << "noface_id" << face.getCenter() << std::endl;
        //         noFlag = true;
        //         yesFlag = false;
        //         getchar();
        //     }
        //     std::cout << "no" << std::endl;
        // }
    }
}



/* =============================================================== */
/*                         并行：区域分解                           */
/* =============================================================== */

namespace
{
    using ULL = unsigned long long;
    using LL = long long;

    // 简单的二进制打包/解包工具（仅用于可平凡拷贝的类型）
    class Packer
    {
    public:
        template<typename T>
        void put(const T& value)
        {
            static_assert(std::is_trivially_copyable_v<T>);
            const char* p = reinterpret_cast<const char*>(&value);
            buffer.insert(buffer.end(), p, p + sizeof(T));
        }

        template<typename T>
        void putVector(const std::vector<T>& values)
        {
            static_assert(std::is_trivially_copyable_v<T>);
            put<ULL>(values.size());
            const char* p = reinterpret_cast<const char*>(values.data());
            buffer.insert(buffer.end(), p, p + values.size() * sizeof(T));
        }

        void putString(const std::string& value)
        {
            put<ULL>(value.size());
            buffer.insert(buffer.end(), value.begin(), value.end());
        }

        std::vector<char> buffer;
    };

    class Unpacker
    {
    public:
        explicit Unpacker(const std::vector<char>& buffer) : buffer_(buffer) {}

        template<typename T>
        T get()
        {
            static_assert(std::is_trivially_copyable_v<T>);
            check(sizeof(T));
            T value;
            std::memcpy(&value, buffer_.data() + pos_, sizeof(T));
            pos_ += sizeof(T);
            return value;
        }

        template<typename T>
        std::vector<T> getVector()
        {
            const ULL n = get<ULL>();
            check(n * sizeof(T));
            std::vector<T> values(n);
            std::memcpy(values.data(), buffer_.data() + pos_, n * sizeof(T));
            pos_ += n * sizeof(T);
            return values;
        }

        std::string getString()
        {
            const ULL n = get<ULL>();
            check(n);
            std::string value(buffer_.data() + pos_, buffer_.data() + pos_ + n);
            pos_ += n;
            return value;
        }

    private:
        void check(std::size_t n) const
        {
            if (pos_ + n > buffer_.size())
            {
                throw std::runtime_error("Mesh: corrupted decomposition data");
            }
        }

        const std::vector<char>& buffer_;
        std::size_t pos_ = 0;
    };

    void putPoint(std::vector<double>& out, const Vector<double>& p)
    {
        out.push_back(p.x());
        out.push_back(p.y());
        out.push_back(p.z());
    }

    Vector<double> getPoint(const std::vector<double>& in, ULL i)
    {
        return Vector<double>(in[3 * i], in[3 * i + 1], in[3 * i + 2]);
    }
}

void Mesh::setupSerialParallelInfo()
{
    distributed_ = false;
    nOwnedCells_ = cells_.size();
    nGlobalCells_ = cells_.size();
    nGlobalFaces_ = faces_.size();
    nGlobalPoints_ = points_.size();
    globalCellIds_.clear();
    globalFaceIds_.clear();
    halo_ = par::HaloExchange();
    cellOrdering_.setupSerial(nOwnedCells_);
    fluxFaceOrdering_.setupSerial(internalFaceIndexes_.size() + boundaryFaceIndexes_.size());
}

std::vector<std::vector<char>> Mesh::decompose(
    const Mesh& global,
    int nProcs,
    std::vector<ULL>& cellPositions,
    std::vector<ULL>& fluxFacePositions)
{
    const std::vector<Face>& gFaces = global.faces_;
    const std::vector<Cell>& gCells = global.cells_;
    const ULL nCells = gCells.size();
    const ULL nFaces = gFaces.size();
    const ULL nPoints = global.points_.size();

    // 1. 按单元中心做 RCB 分解
    std::vector<Vector<double>> centers(nCells);
    for (ULL c = 0; c < nCells; ++c)
    {
        centers[c] = gCells[c].getCenter();
    }
    const std::vector<int> part = par::decomposeRCB(centers, nProcs);

    // 2. 各进程自有单元（全局编号升序）
    std::vector<std::vector<ULL>> owned(nProcs);
    for (ULL c = 0; c < nCells; ++c)
    {
        owned[part[c]].push_back(c);
    }

    // 3. 各进程的局部面（与自有单元相邻的面，全局编号升序）与幽灵单元
    std::vector<std::vector<ULL>> localFaces(nProcs);
    std::vector<std::vector<ULL>> ghosts(nProcs);
    for (ULL f = 0; f < nFaces; ++f)
    {
        const ULL o = gFaces[f].getOwnerIndex();
        const LL n = gFaces[f].getNeighborIndex();
        const int po = part[o];
        localFaces[po].push_back(f);
        if (n >= 0)
        {
            const int pn = part[static_cast<ULL>(n)];
            if (pn != po)
            {
                localFaces[pn].push_back(f);
                ghosts[po].push_back(static_cast<ULL>(n));
                ghosts[pn].push_back(o);
            }
        }
    }
    for (int p = 0; p < nProcs; ++p)
    {
        std::sort(ghosts[p].begin(), ghosts[p].end());
        ghosts[p].erase(std::unique(ghosts[p].begin(), ghosts[p].end()), ghosts[p].end());
    }

    // 4. halo 发送关系：ghostsFrom[q][r] = 进程 q 从进程 r 接收的幽灵单元（全局编号升序）
    std::vector<std::vector<std::vector<ULL>>> ghostsFrom(nProcs, std::vector<std::vector<ULL>>(nProcs));
    for (int q = 0; q < nProcs; ++q)
    {
        for (ULL g : ghosts[q])
        {
            ghostsFrom[q][part[g]].push_back(g);
        }
    }

    // 5. 结果收集顺序：各进程自有单元依次拼接后的全局位置
    cellPositions.clear();
    cellPositions.reserve(nCells);
    for (int p = 0; p < nProcs; ++p)
    {
        cellPositions.insert(cellPositions.end(), owned[p].begin(), owned[p].end());
    }

    // 6. 通量面顺序：串行程序先遍历内部面、再遍历（非 empty）边界面
    std::vector<LL> fluxSequence(nFaces, -1);
    {
        LL pos = 0;
        for (ULL f : global.internalFaceIndexes_)
        {
            fluxSequence[f] = pos++;
        }
        for (ULL f : global.boundaryFaceIndexes_)
        {
            fluxSequence[f] = pos++;
        }
    }
    fluxFacePositions.clear();
    for (int p = 0; p < nProcs; ++p)
    {
        // 与局部循环顺序一致：先内部面（owner 属于本进程），后边界面
        for (ULL f : localFaces[p])
        {
            if (gFaces[f].getNeighborIndex() >= 0 && part[gFaces[f].getOwnerIndex()] == p)
            {
                fluxFacePositions.push_back(static_cast<ULL>(fluxSequence[f]));
            }
        }
        for (ULL f : localFaces[p])
        {
            if (gFaces[f].getNeighborIndex() < 0 && fluxSequence[f] >= 0)
            {
                fluxFacePositions.push_back(static_cast<ULL>(fluxSequence[f]));
            }
        }
    }

    // 7. 为每个进程打包局部网格
    std::vector<LL> cellG2L(nCells, -1);
    std::vector<LL> pointG2L(nPoints, -1);
    std::vector<std::vector<char>> packs(nProcs);

    for (int p = 0; p < nProcs; ++p)
    {
        // 局部单元编号：自有单元在前，幽灵单元在后，各自按全局编号升序
        std::vector<ULL> localCells = owned[p];
        localCells.insert(localCells.end(), ghosts[p].begin(), ghosts[p].end());
        for (ULL i = 0; i < localCells.size(); ++i)
        {
            cellG2L[localCells[i]] = static_cast<LL>(i);
        }

        // 局部点：局部面与局部单元用到的点
        std::vector<ULL> localPoints;
        for (ULL f : localFaces[p])
        {
            const auto& pts = gFaces[f].getPointIndexes();
            localPoints.insert(localPoints.end(), pts.begin(), pts.end());
        }
        for (ULL c : localCells)
        {
            const auto& pts = gCells[c].getPointIndexes();
            localPoints.insert(localPoints.end(), pts.begin(), pts.end());
        }
        std::sort(localPoints.begin(), localPoints.end());
        localPoints.erase(std::unique(localPoints.begin(), localPoints.end()), localPoints.end());
        for (ULL i = 0; i < localPoints.size(); ++i)
        {
            pointG2L[localPoints[i]] = static_cast<LL>(i);
        }

        Packer pk;
        pk.put<int>(static_cast<int>(global.dimension_));
        pk.put<int>(static_cast<int>(global.meshShape_));
        pk.put<ULL>(nCells);
        pk.put<ULL>(nFaces);
        pk.put<ULL>(nPoints);
        pk.put<ULL>(owned[p].size());
        pk.putVector(localCells);
        pk.putVector(localFaces[p]);

        // 点坐标
        std::vector<double> pointCoords;
        pointCoords.reserve(3 * localPoints.size());
        for (ULL gp : localPoints)
        {
            putPoint(pointCoords, global.points_[gp]);
        }
        pk.putVector(pointCoords);

        // 面：拓扑 + 几何
        std::vector<ULL> facePointOffsets{ 0 };
        std::vector<ULL> facePoints;
        std::vector<ULL> faceOwners;
        std::vector<LL> faceNeighbours;
        std::vector<double> faceNormals;
        std::vector<double> faceAreas;
        std::vector<double> faceCenters;
        for (ULL f : localFaces[p])
        {
            const Face& face = gFaces[f];
            for (ULL gp : face.getPointIndexes())
            {
                facePoints.push_back(static_cast<ULL>(pointG2L[gp]));
            }
            facePointOffsets.push_back(facePoints.size());
            faceOwners.push_back(static_cast<ULL>(cellG2L[face.getOwnerIndex()]));
            const LL n = face.getNeighborIndex();
            faceNeighbours.push_back(n >= 0 ? cellG2L[static_cast<ULL>(n)] : -1);
            putPoint(faceNormals, face.getNormal());
            faceAreas.push_back(face.getArea());
            putPoint(faceCenters, face.getCenter());
        }
        pk.putVector(facePointOffsets);
        pk.putVector(facePoints);
        pk.putVector(faceOwners);
        pk.putVector(faceNeighbours);
        pk.putVector(faceNormals);
        pk.putVector(faceAreas);
        pk.putVector(faceCenters);

        // 单元几何
        std::vector<ULL> cellPointOffsets{ 0 };
        std::vector<ULL> cellPoints;
        std::vector<double> cellVolumes;
        std::vector<double> cellCenters;
        for (ULL c : localCells)
        {
            const Cell& cell = gCells[c];
            for (ULL gp : cell.getPointIndexes())
            {
                cellPoints.push_back(static_cast<ULL>(pointG2L[gp]));
            }
            cellPointOffsets.push_back(cellPoints.size());
            cellVolumes.push_back(cell.getVolume());
            putPoint(cellCenters, cell.getCenter());
        }
        pk.putVector(cellPointOffsets);
        pk.putVector(cellPoints);
        pk.putVector(cellVolumes);
        pk.putVector(cellCenters);

        // 边界 patch（按文件顺序，局部起止面）
        const auto localFaceBound = [&](ULL globalFace) {
            return static_cast<ULL>(
                std::lower_bound(localFaces[p].begin(), localFaces[p].end(), globalFace) -
                localFaces[p].begin());
        };
        pk.put<ULL>(global.patchOrder_.size());
        for (const std::string& name : global.patchOrder_)
        {
            const BoundaryPatch& patch = global.boundaryPatches_.at(name);
            const ULL begin = localFaceBound(patch.getStartFace());
            const ULL end = localFaceBound(patch.getStartFace() + patch.getNFace());
            pk.putString(name);
            pk.put<int>(static_cast<int>(patch.getType()));
            pk.put<ULL>(begin);
            pk.put<ULL>(end - begin);
        }
        pk.put<ULL>(localFaceBound(global.emptyFaceIndexesPair_.first));
        pk.put<ULL>(localFaceBound(global.emptyFaceIndexesPair_.second));

        // halo 交换列表（局部编号）
        std::vector<int> neighbours;
        for (int r = 0; r < nProcs; ++r)
        {
            if (r != p && (!ghostsFrom[p][r].empty() || !ghostsFrom[r][p].empty()))
            {
                neighbours.push_back(r);
            }
        }
        pk.putVector(neighbours);
        for (int r : neighbours)
        {
            std::vector<ULL> sendList;
            for (ULL g : ghostsFrom[r][p])
            {
                sendList.push_back(static_cast<ULL>(cellG2L[g]));
            }
            std::vector<ULL> recvList;
            for (ULL g : ghostsFrom[p][r])
            {
                recvList.push_back(static_cast<ULL>(cellG2L[g]));
            }
            pk.putVector(sendList);
            pk.putVector(recvList);
        }

        packs[p] = std::move(pk.buffer);

        // 复位映射
        for (ULL c : localCells)
        {
            cellG2L[c] = -1;
        }
        for (ULL gp : localPoints)
        {
            pointG2L[gp] = -1;
        }
    }

    return packs;
}

void Mesh::buildLocalMesh(const std::vector<char>& pack)
{
    Unpacker up(pack);

    distributed_ = true;
    dimension_ = static_cast<Dimension>(up.get<int>());
    meshShape_ = static_cast<MeshShape>(up.get<int>());
    nGlobalCells_ = up.get<ULL>();
    nGlobalFaces_ = up.get<ULL>();
    nGlobalPoints_ = up.get<ULL>();
    nOwnedCells_ = up.get<ULL>();
    globalCellIds_ = up.getVector<ULL>();
    globalFaceIds_ = up.getVector<ULL>();

    // 点
    const std::vector<double> pointCoords = up.getVector<double>();
    points_.clear();
    points_.reserve(pointCoords.size() / 3);
    for (ULL i = 0; i < pointCoords.size() / 3; ++i)
    {
        points_.push_back(getPoint(pointCoords, i));
    }

    // 面
    const std::vector<ULL> facePointOffsets = up.getVector<ULL>();
    const std::vector<ULL> facePoints = up.getVector<ULL>();
    const std::vector<ULL> faceOwners = up.getVector<ULL>();
    const std::vector<LL> faceNeighbours = up.getVector<LL>();
    const std::vector<double> faceNormals = up.getVector<double>();
    const std::vector<double> faceAreas = up.getVector<double>();
    const std::vector<double> faceCenters = up.getVector<double>();
    faces_.clear();
    faces_.reserve(faceOwners.size());
    for (ULL f = 0; f < faceOwners.size(); ++f)
    {
        std::vector<ULL> pts(facePoints.begin() + facePointOffsets[f],
                             facePoints.begin() + facePointOffsets[f + 1]);
        Face face(pts, faceOwners[f], faceNeighbours[f]);
        face.setGeometry(getPoint(faceNormals, f), faceAreas[f], getPoint(faceCenters, f));
        faces_.emplace_back(std::move(face));
    }

    // 单元几何
    const std::vector<ULL> cellPointOffsets = up.getVector<ULL>();
    const std::vector<ULL> cellPoints = up.getVector<ULL>();
    const std::vector<double> cellVolumes = up.getVector<double>();
    const std::vector<double> cellCenters = up.getVector<double>();

    // 边界 patch：按与串行相同的顺序插入，保证 unordered_map 遍历顺序一致
    boundaryPatches_.clear();
    patchOrder_.clear();
    const ULL nPatches = up.get<ULL>();
    for (ULL i = 0; i < nPatches; ++i)
    {
        const std::string name = up.getString();
        const auto type = static_cast<BoundaryPatch::BoundaryType>(up.get<int>());
        const ULL start = up.get<ULL>();
        const ULL n = up.get<ULL>();
        patchOrder_.push_back(name);
        boundaryPatches_.emplace(name, BoundaryPatch(name, n, start, type));
    }
    const ULL emptyFirst = up.get<ULL>();
    const ULL emptySecond = up.get<ULL>();
    emptyFaceIndexesPair_ = std::make_pair(emptyFirst, emptySecond);

    // 由局部面构造单元拓扑、内部面/边界面列表（与串行相同的算法）
    buildCellsFromFaces();
    const ULL nLocalCells = globalCellIds_.size();
    if (cells_.size() != nLocalCells)
    {
        cells_.resize(nLocalCells);
    }
    for (ULL c = 0; c < nLocalCells; ++c)
    {
        std::vector<ULL> pts(cellPoints.begin() + cellPointOffsets[c],
                             cellPoints.begin() + cellPointOffsets[c + 1]);
        cells_[c].setGeometry(cellVolumes[c], getPoint(cellCenters, c), std::move(pts));
    }

    // halo
    std::vector<int> neighbours = up.getVector<int>();
    std::vector<std::vector<ULL>> sendLists;
    std::vector<std::vector<ULL>> recvLists;
    for (std::size_t i = 0; i < neighbours.size(); ++i)
    {
        sendLists.push_back(up.getVector<ULL>());
        recvLists.push_back(up.getVector<ULL>());
    }
    halo_.setup(std::move(neighbours), std::move(sendLists), std::move(recvLists));

    isValid_ = true;
}

/* ---------------- 并行查询接口 ---------------- */

ULL Mesh::getLocalCellNumber() const
{
    return static_cast<ULL>(cells_.size());
}

ULL Mesh::getGhostCellNumber() const
{
    return static_cast<ULL>(cells_.size()) - nOwnedCells_;
}

ULL Mesh::getGlobalCellNumber() const
{
    return nGlobalCells_;
}

ULL Mesh::getGlobalFaceNumber() const
{
    return nGlobalFaces_;
}

ULL Mesh::getGlobalPointNumber() const
{
    return nGlobalPoints_;
}

bool Mesh::isDistributed() const
{
    return distributed_;
}

ULL Mesh::getGlobalCellIndex(ULL localCell) const
{
    return distributed_ ? globalCellIds_[localCell] : localCell;
}

ULL Mesh::getGlobalFaceIndex(ULL localFace) const
{
    return distributed_ ? globalFaceIds_[localFace] : localFace;
}

LL Mesh::findLocalCell(ULL globalCell, bool includeGhost) const
{
    if (!distributed_)
    {
        return globalCell < cells_.size() ? static_cast<LL>(globalCell) : -1;
    }
    // 自有单元、幽灵单元各自按全局编号升序存放
    const auto search = [&](ULL first, ULL last) -> LL {
        const auto begin = globalCellIds_.begin() + static_cast<std::ptrdiff_t>(first);
        const auto end = globalCellIds_.begin() + static_cast<std::ptrdiff_t>(last);
        const auto it = std::lower_bound(begin, end, globalCell);
        if (it != end && *it == globalCell)
        {
            return static_cast<LL>(it - globalCellIds_.begin());
        }
        return -1;
    };
    const LL owned = search(0, nOwnedCells_);
    if (owned >= 0 || !includeGhost)
    {
        return owned;
    }
    return search(nOwnedCells_, globalCellIds_.size());
}

const par::HaloExchange& Mesh::getHalo() const
{
    return halo_;
}

const par::GlobalOrdering& Mesh::getCellOrdering() const
{
    return cellOrdering_;
}

const par::GlobalOrdering& Mesh::getFluxFaceOrdering() const
{
    return fluxFaceOrdering_;
}

const Mesh& Mesh::getGlobalMesh() const
{
    if (!distributed_)
    {
        return *this;
    }
    if (!globalMesh_)
    {
        throw std::runtime_error("Mesh::getGlobalMesh(): the global mesh is only available on the master process");
    }
    return *globalMesh_;
}
