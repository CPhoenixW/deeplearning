package gcn.data;

import gcn.core.SparseMatrix;

import java.io.*;
import java.util.*;

/**
 * 构建邻接矩阵（从 .cites 文件），并返回归一化矩阵 A_hat
 */
public class GraphPreprocessor {

    /**
     * 构建邻接矩阵并进行对称归一化
     *
     * @param filePath   .cites 文件路径
     * @param paperIdMap paperId -> index 映射（由 DatasetLoader 提供）
     * @param numNodes   总节点数（即 paper 数）
     */
    public static SparseMatrix buildNormalizedAdj(String filePath, Map<String, Integer> paperIdMap, int numNodes) throws IOException {
        SparseMatrix adj = new SparseMatrix(numNodes, numNodes);
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] tokens = line.split("\t");
            String cited = tokens[0];
            String citing = tokens[1];

            Integer from = paperIdMap.get(citing);
            Integer to = paperIdMap.get(cited);
            if (from != null && to != null) {
                adj.addEdge(from, to);
                adj.addEdge(to, from);
            }
        }
        reader.close();
        return adj.normalizeSymmetric();
    }
}
