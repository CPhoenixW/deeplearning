package gcn.data;

import java.io.*;
import java.util.*;

/**
 * 加载 Cora 数据集：
 * - .content 文件 → 特征矩阵、标签
 * - paperId -> index 映射
 */
public class DatasetLoader {
    public double[][] features;
    public int[] labels;
    public Map<String, Integer> paperIdMap; // paperID -> index
    public String[] indexToPaperId;         // index -> paperID
    public int numClasses;

    private static final Map<String, Integer> labelToIndex = new HashMap<>();

    public void loadContent(String filePath) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        List<double[]> featureList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        paperIdMap = new HashMap<>();
        List<String> paperIdList = new ArrayList<>();

        int row = 0;
        String line;
        while ((line = reader.readLine()) != null) {
            String[] tokens = line.split("\t");
            String paperId = tokens[0];
            paperIdMap.put(paperId, row);
            paperIdList.add(paperId);

            double[] feat = new double[tokens.length - 2];
            for (int i = 1; i < tokens.length - 1; i++) {
                feat[i - 1] = Double.parseDouble(tokens[i]);
            }
            featureList.add(feat);

            String labelStr = tokens[tokens.length - 1];
            if (!labelToIndex.containsKey(labelStr)) {
                labelToIndex.put(labelStr, labelToIndex.size());
            }
            labelList.add(labelToIndex.get(labelStr));

            row++;
        }
        reader.close();

        int n = featureList.size();
        int dim = featureList.get(0).length;
        features = new double[n][dim];
        labels = new int[n];
        for (int i = 0; i < n; i++) {
            features[i] = featureList.get(i);
            labels[i] = labelList.get(i);
        }
        indexToPaperId = paperIdList.toArray(new String[0]);
        numClasses = labelToIndex.size();
    }
}
