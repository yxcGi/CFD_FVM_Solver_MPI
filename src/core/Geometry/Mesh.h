#ifndef MESH_H_
#define MESH_H_


// #include <iostream>
#include <vector>
#include <unordered_map>
#include "Vector.hpp"
#include "Face.h"
#include "Cell.h"
#include "BoundaryPatch.h"
#include <unordered_set>
#include <memory>
#include <string>
// #include "BaseField.hpp"
#include "FieldType.hpp"
#include "Parallel/Parallel.h"

// using field::FieldType;




/**
 * @brief 网格
 *
 * 串行运行（单进程）时与原实现完全相同。
 *
 * MPI 并行运行时，0 号进程读取完整网格并计算全部几何量，然后做区域分解，
 * 把各子区域发送给对应进程。每个进程上的 Mesh 为局部网格：
 *
 *   单元：[0, getCellNumber())                 自有单元（参与计算）
 *         [getCellNumber(), getLocalCellNumber()) 幽灵单元（相邻进程的单元副本）
 *   面：  所有与自有单元相邻的面，按全局面编号升序排列；
 *         进程间的面（一侧为幽灵单元）作为内部面处理
 *
 * 所有索引均为局部索引，可通过 getGlobalCellIndex()/getGlobalFaceIndex() 映射到全局编号。
 * 局部编号保持全局编号的相对顺序，因此各种按面/按单元的循环顺序与串行程序一致，
 * 这是并行结果与串行结果逐位一致的基础。
 */
class Mesh
{
    using ULL = unsigned long long;
    using LL = long long;
    using Scalar = double;
public:
    // 网格维度
    enum class Dimension
    {
        TWO_D,
        THREE_D
    };
    // 网格形状（tecplot里的非结构网格类型）
    enum class MeshShape
    {
        TRIANGLE,   // 三角形单元
        QUADRILATERAL,  // 四边形单元
        TETRAHEDRON,   // 四面体单元
        BRICK,      // 六面体单元
    };

public:
    Mesh() = delete;
    Mesh(const std::string& path);
    Mesh(const Mesh&) = delete;
    Mesh(Mesh&&) noexcept;
    Mesh& operator=(const Mesh&) = delete;
    Mesh& operator=(Mesh&&) noexcept;
    ~Mesh();

public:
    // 常用接口
    void readMesh(const std::string& path);
    void writeMesh(const std::string& path);
    void printMeshInfo() const;
    void writeMeshToFile(const std::string& path) const;

    // 获取器
    const std::vector<Vector<Scalar>>& getPoints() const;
    const std::vector<Face>& getFaces() const;
    const std::vector<ULL>& getInternalFaceIndexes() const;
    const std::vector<ULL>& getBoundaryFaceIndexes() const;
    const std::vector<Cell>& getCells() const;     // 局部单元（自有 + 幽灵）
    ULL getCellNumber() const;      // 自有单元数（串行时即全部单元数）
    ULL getFaceNumber() const;      // 局部面数
    ULL getPointNumber() const;     // 局部点数
    Dimension getDimension() const;
    MeshShape getMeshShape() const;

    ULL getInternalCellNumber() const;
    ULL getBoundaryFaceNumber() const;
    const std::pair<ULL, ULL>& getEmptyFaceIndexesPair() const;

    const std::unordered_map<std::string, BoundaryPatch>&
        getBoundaryPatches() const;
    void getBoundaryMessage() const;    // 获取边界信息

    // 用于获取不同类型场的数量的统一接口
    ULL getNumber(field::FieldType type) const;

    bool isValid() const;

    /* ================= 并行相关 ================= */
    // 局部存储单元数 = 自有单元 + 幽灵单元（单元场的数组长度）
    ULL getLocalCellNumber() const;
    ULL getGhostCellNumber() const;
    // 全局数量
    ULL getGlobalCellNumber() const;
    ULL getGlobalFaceNumber() const;
    ULL getGlobalPointNumber() const;
    // 是否为分布式（多进程）网格
    bool isDistributed() const;
    // 局部编号 -> 全局编号
    ULL getGlobalCellIndex(ULL localCell) const;
    ULL getGlobalFaceIndex(ULL localFace) const;
    // 全局单元编号 -> 局部编号（不在本进程时返回 -1）
    // includeGhost = false 时只查找自有单元；true 时也查找幽灵单元
    LL findLocalCell(ULL globalCell, bool includeGhost = false) const;
    // 幽灵单元数据交换
    const par::HaloExchange& getHalo() const;
    // 自有单元的全局排序（结果收集、按串行顺序求和）
    const par::GlobalOrdering& getCellOrdering() const;
    // 通量面的全局排序：内部面（仅计 owner 为自有单元的面）在前，边界面在后，
    // 与串行程序"先内部面、后边界面"的循环顺序一致
    const par::GlobalOrdering& getFluxFaceOrdering() const;
    // 完整网格：串行时为自身；并行时仅 0 号进程可用（用于输出）
    const Mesh& getGlobalMesh() const;





private:
    // 私有处理接口
    void readPoints(const std::string& pointsPath);
    void readBoundaryPatch(const std::string& boundaryPath);
    void readFaces(
        const std::string& facesPath,
        const std::string& ownerPath,
        const std::string& neighbourPath
    );
    void readNeighbour(const std::string& neighbourPath, std::vector<ULL> internalFaceIndices);

    void writePoints(const std::string& pointsPath) const;
    void writeFaces(const std::string& facesPath) const;
    void writeOwnerAndNeighbour(const std::string& ownerPath, const std::string& neighbourPath) const;
    void writeBoundaryPatch(const std::string& boundaryPath) const;

    void buildCellsFromFaces();

    // 并行：串行读取标记
    struct SerialReadTag {};
    Mesh(const std::string& path, SerialReadTag);
    // 并行：0 号进程分解网格，生成各进程的数据包
    static std::vector<std::vector<char>> decompose(const Mesh& global, int nProcs,
                                                    std::vector<ULL>& cellPositions,
                                                    std::vector<ULL>& fluxFacePositions);
    // 并行：由数据包构造局部网格
    void buildLocalMesh(const std::vector<char>& pack);
    // 设置串行情形下的并行信息（单进程）
    void setupSerialParallelInfo();
    BoundaryPatch::BoundaryType stringToType(const std::string& name) const;
    void setMeshShape();
    void calculateMeshInfo();




private:
    std::vector<Vector<Scalar>> points_;                                    // 点列表
    std::vector<Face> faces_;                                               // 面列表
    std::vector<ULL> internalFaceIndexes_;                                 // 内部面索引列表
    std::vector<ULL> boundaryFaceIndexes_;                                 // 边界面索引列表
    std::vector<Cell> cells_;                                               // 单元列表
    std::unordered_set<ULL> internalCellIndexes_;                                 // 内部单元索引列表
    std::unordered_set<ULL> boundaryCellIndexes_;                                 // 边界面索引列表
    std::unordered_map<std::string, BoundaryPatch> boundaryPatches_;     // 边界条件映射
    std::pair<ULL, ULL> emptyFaceIndexesPair_;                              // empty边界面的起止索引（二维专属，左闭右开）
    bool isValid_;                                                          // 网格是否有效
    std::string meshPath_;                                                  // 网格路径
    Dimension dimension_;                                                   // 网格维度，默认三维
    MeshShape meshShape_;
    std::vector<std::string> patchOrder_;                                   // 边界 patch 在文件中的顺序

    /* 并行数据 */
    bool distributed_ = false;
    ULL nOwnedCells_ = 0;
    ULL nGlobalCells_ = 0;
    ULL nGlobalFaces_ = 0;
    ULL nGlobalPoints_ = 0;
    std::vector<ULL> globalCellIds_;                                        // 局部单元 -> 全局单元
    std::vector<ULL> globalFaceIds_;                                        // 局部面 -> 全局面
    par::HaloExchange halo_;
    par::GlobalOrdering cellOrdering_;
    par::GlobalOrdering fluxFaceOrdering_;
    std::unique_ptr<Mesh> globalMesh_;                                      // 0 号进程保存的完整网格
};



#endif // MESH_H_